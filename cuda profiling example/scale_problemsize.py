import numpy as np

"""
This class handles the scaling of the problem size based on the configuration specified by the user.
The class does not do any preprocessing. 
"""
class ProblemScaler:

    """
    Load the data for training and evaluation when initializing the class.
    """
    def __init__(self, classes, trainX, trainY, testX, testY):
        self.classes = classes
        self.testX = testX
        self.testY = testY
        self.trainX = trainX
        self.trainY = trainY

    """
    Scales the problem size of train and test data based on the user specified problem size.
    Problem size must be integer!
    """
    def scale_problem(self, PROBLEM_SIZE):
        # for test data
        if PROBLEM_SIZE != 100:
            new_testX = []
            new_testY = []
            classes = []
            for i in range(PROBLEM_SIZE):
                classes.append(i)
            for i in range(len(self.testY)):
                for j in range(PROBLEM_SIZE):
                    if self.testY[i] == classes[j]:
                        value = self.testY[i]
                        image = self.testX[i]
                        new_testY.append(value)
                        new_testX.append(image)
                        break
            self.testY = new_testY
            self.testX = new_testX
            self.testY = np.asarray(self.testY)
            self.testX = np.asarray(self.testX)

        # for training data
        if PROBLEM_SIZE != 100:
            new_trainY = []
            new_trainX = []
            classes = []
            for i in range(PROBLEM_SIZE):
                classes.append(i)
            for i in range(len(self.trainY)):
                for j in range(PROBLEM_SIZE):
                    if self.trainY[i] == classes[j]:
                        value = self.trainY[i]
                        image = self.trainX[i]
                        new_trainY.append(value)
                        new_trainX.append(image)
                        break
            self.trainY = new_trainY
            self.trainX = new_trainX
            self.trainY = np.asarray(self.trainY)
            self.trainX = np.asarray(self.trainX)
        
        return self.trainX, self.trainY, self.testX, self.testY
