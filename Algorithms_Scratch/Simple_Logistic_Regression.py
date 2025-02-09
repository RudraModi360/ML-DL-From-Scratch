import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression


class Simple_Logistic_Regression:

    def __init__(self):
        self.weights = None
        self.epochs = 2500
        self.lr_rate = 0.0001

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.ones(X.shape[1])
        for j in range(self.epochs):
            y_pred = self.sigmoid(np.dot(X, self.weights))
            self.weights += self.lr_rate * np.dot((y - y_pred), X) / X.shape[0]
        return self.weights[1:], self.weights[0]

    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))

    def predict(self, X):
        X = np.array(X)
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self.sigmoid(np.dot(self.weights, X[i])))
        return np.array(y_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score


df = pd.read_csv("Datasets\\placement-dataset.csv")
df = df.dropna()
df = df.drop(["Unnamed: 0"], axis=1)

X, y = df.drop(["placement"], axis=1), df["placement"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = Simple_Logistic_Regression()
print(model.fit(X_train, y_train))

mdl = LogisticRegression()
mdl.fit(np.array(X_train), np.array(y_train))
print(mdl.coef_, mdl.intercept_)
