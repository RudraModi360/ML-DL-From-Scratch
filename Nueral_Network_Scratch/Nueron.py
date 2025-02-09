import nnfs.datasets
import numpy as np
import nnfs

X, y = nnfs.datasets.spiral_data(100, 3)


class Dense:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forwad(self, n_inputs):
        self.outputs = np.dot(n_inputs, self.weights) + self.biases


class ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)


class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = exp_values / np.sum(exp_values, axis=1, keepdims=True)


class Loss:
    def calculate(self, outputs, y):
        return np.mean(self.forward(outputs, y))

class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-9, 1 - 1e-9)
        if len(y_true.shape) == 1:  
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else: 
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(correct_confidences)


dense_1 = Dense(2, 5)
activation1 = ReLU()
dense_2 = Dense(5, 3)
activation2 = Softmax()

dense_1.forwad(X)
activation1.forward(dense_1.outputs)
dense_2.forwad(activation1.outputs)
activation2.forward(dense_2.outputs)

loss = CategoricalCrossEntropy()
print(loss.calculate(activation2.outputs, y))
