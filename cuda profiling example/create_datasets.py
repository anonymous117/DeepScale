import pickle
import numpy as np

"""
This class handles loading the cifar100 data from the files in this directoy.
It loads the data and transfers it from dictonary format into numpy arrays.
The methods do not do any data preprocessing.
"""
class DataLoader:

    """
    Open the pickeled files containing the training and test data.
    """
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    """
    Get the classes of the classification data set.
    """
    def get_classes(self):
        file = "cifar-100-data/meta"
        dict = self.unpickle(file)
        label_names = dict[b"fine_label_names"]
        classes = []
        for i in range(len(label_names)):
            classes.append(label_names[i].decode("utf-8"))
        return classes

    """
    Get the evaluation data as numpy arrays.
    """
    def get_evaluation_data(self):
        file = "cifar-100-data/test"
        dict = self.unpickle(file)
        evaluation_data_labels = dict[b"fine_labels"]
        testY = np.array(evaluation_data_labels)
        data = dict[b"data"]
        evaluation_data = []
        for i in range(len(data)):
            image_data = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            image_data = image_data.astype("float32") / 255.0
            evaluation_data.append(image_data)
        testX = np.array(evaluation_data)
        #evaluation_data_labels = tf.one_hot(evaluation_data_labels, 10)
        #evaluation_dataset = tf.data.Dataset.from_tensor_slices((evaluation_data, evaluation_data_labels))
        return testX, testY

    """
    Get the training data as numpy arrays.
    """
    def get_training_data(self):
        file = "cifar-100-data/train"
        dict = self.unpickle(file)
        training_data_labels = dict[b"fine_labels"]
        trainY = np.array(training_data_labels)
        data = dict[b"data"]
        training_data = []
        for i in range(len(data)):
            image_data = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            image_data = image_data.astype("float32") / 255.0
            training_data.append(image_data)
        trainX = np.array(training_data)
        #training_data_labels = tf.one_hot(training_data_labels, 10)
        #training_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_data_labels))
        return trainX, trainY
