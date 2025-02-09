import numpy as np

weigths = np.array([1, 2, -1],dtype=float)
biases = 1
inputs = np.array([2, 1, 3],dtype=float)
target_op = 0
learning_rate = 0.001


def relu(x):
    return max(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


for iteration in range(100):
    linear_op = np.dot(weigths, inputs) + biases
    outputs = relu(linear_op)
    loss = (outputs - target_op) ** 2

    dloss_d_relu = 2 * outputs
    drelu_dsum = relu_derivative(linear_op)
    dlinear_dweigths = inputs
    dlinear_dbias = 1

    dloss_dlinear = dloss_d_relu * drelu_dsum
    dloss_dweigths = dloss_dlinear * dlinear_dweigths
    dloss_dbias = dloss_dlinear * dlinear_dbias

    weigths -= learning_rate * dloss_dweigths
    biases -= learning_rate * dloss_dbias

    print(f"Iteration : {iteration+1} , Loss : {loss}")
print(f"Final Weigths : {weigths} , Bias : {biases}")
