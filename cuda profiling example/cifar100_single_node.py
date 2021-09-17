"""
CIFAR-100 code for training a DNN.

Adjusts only the problem size and runs on a single GPU (if available) or CPU.

run the code with the profiler using:

srun -n1 -N1 nsys profile -f true -o my_report ./executable JOBNAME PROBLEM_SIZE REPETITION

analyze the profile with:

nsys stats my_report.qdrep

"""

import os
import time
import json
import tensorflow as tf
import numpy as np
import random
import sys
import nvtx

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from create_datasets import DataLoader
from scale_problemsize import ProblemScaler

# Read command line options for the script
if len(sys.argv)>3:
    JOBNAME = sys.argv[1]
    PROBLEM_SIZE = int(sys.argv[2])
    REPETITION = int(sys.argv[3])
else:
    JOBNAME = "test"
    PROBLEM_SIZE = 100
    REPETITION = 1
JOBNAME = JOBNAME + "_" + str(PROBLEM_SIZE) + "_" + str(REPETITION)

# Definition off some constants used in the code
BATCH_SIZE = 128
# since nothing changes in code, 5 epochs are enough for profiling with nsys
EPOCHS = 5
VERBOSE = 1
LEARNING_RATE = 0.4
MOMENTUM = 0.9
WARMUP = 20

# Define the checkpoint directory to store the checkpoints
checkpoint_dir = 'training_checkpoints/'+JOBNAME+'/'

# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# Set weight decay
WEIGHT_DECAY = 0.000125 * BATCH_SIZE

# DEBUG CODE
print("Tensorflow version:", tf.__version__)
print("Number of GPUs available:", len(tf.config.experimental.list_physical_devices("GPU")))
print("JOBNAME:", JOBNAME)
print("PROBLEM_SIZE:", PROBLEM_SIZE)
print("REPETITION:", REPETITION)

#---- START: Load the CIFAR-100 data ----
START_LOAD_DATA_PROC = time.process_time()

data_loader = DataLoader()
classes = data_loader.get_classes()
testX, testY = data_loader.get_evaluation_data()
trainX, trainY = data_loader.get_training_data()

END_LOAD_DATA_PROC = time.process_time()
LOAD_DATA_PROC_RUNTIME = END_LOAD_DATA_PROC - START_LOAD_DATA_PROC
LOAD_DATA_PROC_RUNTIME = "{:.3f}".format(LOAD_DATA_PROC_RUNTIME)
#---- END: Load CIFAR-100 data ----

#---- START: Adjust the problem size ----
START_ADJUST_PROBLEM_SIZE_PROC = time.process_time()

scaler = ProblemScaler(classes, trainX, trainY, testX, testY)
trainX, trainY, testX, testY = scaler.scale_problem(PROBLEM_SIZE)

END_ADJUST_PROBLEM_SIZE_PROC = time.process_time()
ADJUST_PROBLEM_SIZE_PROC_RUNTIME = END_ADJUST_PROBLEM_SIZE_PROC - START_ADJUST_PROBLEM_SIZE_PROC
ADJUST_PROBLEM_SIZE_PROC_RUNTIME = "{:.3f}".format(ADJUST_PROBLEM_SIZE_PROC_RUNTIME)
#---- END: Adjust the problem size ----

#---- START: preprocessing ----
START_PREPROCESSING_PROC = time.process_time()

# One-hot encode the labels
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# Convert from integers to floats
trainX = trainX.astype('float32')
testX = testX.astype('float32')

# Normalize to range 0-1
X_train_mean = np.mean(trainX, axis=(0,1,2))
X_train_std = np.std(trainX, axis=(0,1,2))
trainX = (trainX - X_train_mean) / X_train_std
testX = (testX - X_train_mean) / X_train_std

END_PREPROCESSING_PROC = time.process_time()
PREPROCESSING_PROC_RUNTIME = END_PREPROCESSING_PROC - START_PREPROCESSING_PROC
PREPROCESSING_PROC_RUNTIME = "{:.3f}".format(PREPROCESSING_PROC_RUNTIME)
#---- END: preprocessing ----

# Function for decaying the learning rate
@nvtx.annotate("decay()", color="red")
def decay(epoch):
    max_lr = LEARNING_RATE / BATCH_SIZE
    min_lr = 1e-4
    if epoch < WARMUP:
        m = (min_lr-max_lr)/(WARMUP-0)
        b = min_lr-(m*0)
        x = epoch
        lr = -m*x+b
        return lr
    elif epoch > WARMUP:
        m = (max_lr-min_lr)/(EPOCHS-WARMUP)
        b = -1*((-m*EPOCHS)-min_lr)
        x = epoch
        lr = -m*x+b
        return lr
    else:
        return max_lr

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()))

# Callbacks for performance profiler, learning rate adjustment and model checkpoints
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs/'+JOBNAME+'/', update_freq="epoch", profile_batch='100, 200'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]

# Create the Model
@nvtx.annotate("define_model()", color="purple")
def define_model(PROBLEM_SIZE):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same", kernel_regularizer=l2(WEIGHT_DECAY), input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same", kernel_regularizer=l2(WEIGHT_DECAY)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same", kernel_regularizer=l2(WEIGHT_DECAY)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same", kernel_regularizer=l2(WEIGHT_DECAY)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same", kernel_regularizer=l2(WEIGHT_DECAY)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same", kernel_regularizer=l2(WEIGHT_DECAY)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(PROBLEM_SIZE, activation="softmax"))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

#---- START: define model ----
START_DEFINE_MODEL_PROC = time.process_time()

model = define_model(PROBLEM_SIZE)

END_DEFINE_MODEL_PROC = time.process_time()
DEFINE_MODEL_PROC_RUNTIME = END_DEFINE_MODEL_PROC - START_DEFINE_MODEL_PROC
DEFINE_MODEL_PROC_RUNTIME = "{:.3f}".format(DEFINE_MODEL_PROC_RUNTIME)
#---- END: define model ----

# Create an Image data generator for augmenting and sampling new training data for the epochs
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
it_train = datagen.flow(trainX, trainY, batch_size=BATCH_SIZE)

# Calculate the number of training steps per epoch
STEPS = int(trainX.shape[0] / BATCH_SIZE)

#---- START: training loop ----
START_TRAINING_LOOP_PROC = time.process_time()

history = model.fit(it_train, steps_per_epoch=STEPS, epochs = EPOCHS,
    validation_data = (testX, testY), verbose=VERBOSE, callbacks=callbacks)

END_TRAINING_LOOP_PROC = time.process_time()
TRAINING_LOOP_PROC_RUNTIME = END_TRAINING_LOOP_PROC - START_TRAINING_LOOP_PROC
TRAINING_LOOP_PROC_RUNTIME = "{:.3f}".format(TRAINING_LOOP_PROC_RUNTIME)
#---- END: training loop ----

#---- START: Evaluate the trained model over the test data set ----
START_EVALUATION_PROC = time.process_time()

TEST_LOSS, TEST_ACC = model.evaluate(testX, testY, verbose=VERBOSE)
TEST_LOSS = "{:.3f}".format(TEST_LOSS)
TEST_ACC = "{:.3f}".format(TEST_ACC)

END_EVALUATION_PROC = time.process_time()
EVALUATION_PROC_RUNTIME = END_EVALUATION_PROC - START_EVALUATION_PROC
EVALUATION_PROC_RUNTIME = "{:.3f}".format(EVALUATION_PROC_RUNTIME)
#---- END: Evaluate the trained model over the test data set ----

# DEBUG CODE
print ("Final test loss: ", TEST_LOSS)
print ("Final test accuracy: ", TEST_ACC)

# Create the performance profile
def create_performance_profile():
    performance_profile = {}
    performance_profile["ACCURACY"] = TEST_ACC
    performance_profile["LOSS"] = TEST_LOSS
    performance_profile["TRAINING_LOOP_PROC_RUNTIME"] = TRAINING_LOOP_PROC_RUNTIME
    performance_profile["EVALUATION_PROC_RUNTIME"] = EVALUATION_PROC_RUNTIME
    performance_profile["DEFINE_MODEL_PROC_RUNTIME"] = DEFINE_MODEL_PROC_RUNTIME
    performance_profile["LOAD_DATA_PROC_RUNTIME"] = LOAD_DATA_PROC_RUNTIME
    performance_profile["PREPROCESSING_PROC_RUNTIME"] = PREPROCESSING_PROC_RUNTIME
    performance_profile["ADJUST_PROBLEM_SIZE_PROC_RUNTIME"] = ADJUST_PROBLEM_SIZE_PROC_RUNTIME
    return performance_profile

performance_profile = create_performance_profile()

# Save the performance profile
def save_performance_profile(performance_profile):
    try:
        os.mkdir("profiles/"+JOBNAME+"/")
    except OSError as error:
        pass
    with open("profiles/"+JOBNAME+"/profile.json", "w", encoding="utf-8") as f:
        json.dump(performance_profile, f, ensure_ascii=False, indent=4)

save_performance_profile(performance_profile)

# Get the data from the training history and save it to a file
def save_training_history():
    acc = history.history["accuracy"]
    loss = history.history["loss"]
    val_acc = history.history["val_accuracy"]
    val_loss = history.history["val_loss"]
    try:
        os.mkdir("data/"+JOBNAME+"/")
    except OSError as error:
        pass
    f = open("data/"+JOBNAME+"/training_data.txt", "w")
    text = "["
    for i in range(len(acc)):
        text += str(acc[i])+","
    text = text[:-1]
    text += "]\n"
    f.write(text)
    text = "["
    for i in range(len(loss)):
        text += str(loss[i])+","
    text = text[:-1]
    text += "]\n"
    f.write(text)
    text = "["
    for i in range(len(val_acc)):
        text += str(val_acc[i])+","
    text = text[:-1]
    text += "]\n"
    f.write(text)
    text = "["
    for i in range(len(val_loss)):
        text += str(val_loss[i])+","
    text = text[:-1]
    text += "]\n"
    f.write(text)
    f.close()

save_training_history()

# Save the final model as SavedModel
path = "models/"+JOBNAME+"/saved_model"
model.save(path, save_format="tf")
