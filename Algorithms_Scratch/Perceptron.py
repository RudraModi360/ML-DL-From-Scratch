import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr_rate = learning_rate
        self.epochs = iterations
        self.weigths = None
        self.bias = None

    def fit(self, X, y):
        self.weigths = np.zeros((X.shape[1]))
        self.bias = 0
        X, y = np.array(X), np.array(y)
        for j in range(self.epochs):
            for i in range(X.shape[0]):
                y_pred = self.activations(np.dot(self.weigths, X[i]) + self.bias)
                self.weigths += self.lr_rate * (y[i] - y_pred) * X[i]
                self.bias += self.lr_rate * (y[i] - y_pred)
        return [self.weigths, self.bias]

    def activations(self, y):
        if y >= 0:
            return 1
        else:
            return 0

    def predict(self, X):
        X = np.array(X)
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self.activations(np.dot(self.weigths, X[i]) + self.bias))
        return np.array(y_pred)


df = pd.read_csv("Datasets\\placement-dataset.csv")
df = df.dropna()
df = df.drop(["Unnamed: 0"], axis=1)

X, y = df.drop(["placement"], axis=1), df["placement"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = Perceptron()
print(model.fit(X_train, y_train))
print(model.predict(X_test))
print(accuracy_score(y_test, model.predict(X_test)))
