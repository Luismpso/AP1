import numpy as np

class LossFunction:
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    def derivative(self, y_true, y_pred):
        raise NotImplementedError

class CategoricalCrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / len(y_true) 

    def derivative(self, y_true, y_pred):
        return (y_pred - y_true) / len(y_true)