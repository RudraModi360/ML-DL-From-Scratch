import nnfs.datasets
import numpy as np
import matplotlib.pyplot as plt
import nnfs


class Dense:
    def __init__(self, n_inputs, n_nueorns):
        self.weigths = 0.01 * np.random.randn(n_inputs, n_nueorns)
        self.biases = np.zeros((1, n_nueorns))

    def forward(self, inputs) -> np.array:
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weigths) + self.biases

    def backward(self, dvalues):
        self.dweigths = np.dot(self.inputs.T, dvalues)
        self.biases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weigths.T)


class ReLU:
    def forward(self, x):
        self.inputs = x
        self.outputs = np.maximum(0, x)

    def backward(self, dvalues: list):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class SoftMax:
    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.outputs = prob


class Loss:
    def calculate(self, y_pred, y_true):
        loss = self.forward(y_pred, y_true)
        return np.mean(loss)


class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        y_clip = np.clip(y_pred, 1e-9, 1 - 1e-9)
        if len(y_true.shape) == 1:
            confidence_val = y_clip[
                range(len(y_pred)), y_true
            ]  ## for Sparse_categoricalCrossEntropy case
        else:
            confidence_val = np.sum(y_pred * y_true)  ## for Categorical Outputs
        return -np.log(confidence_val)

    def backward(self, dvalues, y_true):
        samples = len(y_true)
        labels = len(len(dvalues)[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class SotMax_Loss_CategoricalCrossEntropy:
    def __init__(self):
        self.activation = SoftMax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        return self.loss.calculate(self.activation.outputs, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


# softmax_op = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
# y_true=np.array([0,1,1])
# softmax_loss=SotMax_Loss_CategoricalCrossEntropy()
# softmax_loss.backward(softmax_op,y_true)
# print("Gradients Combined Loss and Activation ")
# print(softmax_loss.dinputs)


# X, y = nnfs.datasets.spiral_data(samples=100, classes=3)
# # plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
# # plt.show()
# dense_1 = Dense(X.shape[1], 5000)
# dense_2 = Dense(5000, 3)
# relu_1 = ReLU()
# loss = CategoricalCrossEntropy()
# dense_1.forward(X)
# relu_1.forward(dense_1.outputs)
# dense_2.forward(relu_1.outputs)
# softmax_1 = SoftMax()
# softmax_1.forward(dense_2.outputs)
# print(loss.calculate(softmax_1.outputs,y))
# predictions = np.argmax(softmax_1.outputs, axis=1)
# if len(y.shape) == 2:
#     softmax_1.outputs = np.argmax(softmax_1.outputs, axis=1)

# acc = np.mean(predictions == y)
# print("Accuracy : ", acc)

# X, y = nnfs.datasets.vertical_data(100, 3)
# dense_1 = Dense(X.shape[1], 3)
# dense_2 = Dense(3, 3)
# relu_1 = ReLU()
# loss = CategoricalCrossEntropy()
# softmax_1 = SoftMax()

# lowest_loss = 999999
# best_dense_1_weigths = dense_1.weigths.copy()
# best_dense_1_biases = dense_1.biases.copy()
# best_dense_2_weigths = dense_2.weigths.copy()
# best_dense_2_biases = dense_2.biases.copy()
# for i in range(10000):
#     dense_1.weigths += 0.05 * np.random.randn(2, 3)
#     dense_1.biases += 0.05 * np.random.randn(1, 3)
#     dense_2.weigths += 0.05 * np.random.randn(3, 3)
#     dense_2.biases += 0.05 * np.random.randn(1, 3)
#     dense_1.forward(X)
#     relu_1.forward(dense_1.outputs)
#     dense_2.forward(relu_1.outputs)
#     softmax_1.forward(dense_2.outputs)
#     loss_val = loss.calculate(softmax_1.outputs, y)
#     predictions = np.argmax(softmax_1.outputs, axis=1)
#     if len(y.shape) == 2:
#         softmax_1.outputs = np.argmax(softmax_1.outputs, axis=1)
#     acc = np.mean(predictions == y)
#     if loss_val < lowest_loss:
#         print(f"Iteration {i} , Loss :{loss_val} , accuracy : {acc}")
#         lowest_loss = loss_val
#         best_dense_1_weigths = dense_1.weigths.copy()
#         best_dense_1_biases = dense_1.biases.copy()
#         best_dense_2_weigths = dense_2.weigths.copy()
#         best_dense_2_biases = dense_2.biases.copy()
#     else:
#         dense_1.weigths = best_dense_1_weigths.copy()
#         dense_1.biases = best_dense_1_biases.copy()
#         dense_2.weigths = best_dense_2_weigths.copy()
#         dense_2.biases = best_dense_2_biases.copy()


