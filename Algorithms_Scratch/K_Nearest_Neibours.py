import operator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        distance = {
            i: np.dot((self.X[j] - X_test[i]), (self.X[j] - X_test[i]))
            for i in range(len(X_test))
            for j in range(len(self.X))
        }
        distance = sorted(distance.items(), key=operator.itemgetter(1))
        return self.classify(distance[: self.k])

    def classify(self, distance: list):
        label = [self.y[x[0]] for x in distance]
        return Counter(label).most_common()[0][0]


df = pd.read_csv("Datasets\\placement-dataset.csv")
df = df.dropna()
df = df.drop(["Unnamed: 0"], axis=1)

X, y = np.array(df.drop(["placement"], axis=1)), np.array(df["placement"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = KNN(k=5)
model.fit(X_train, y_train)
print(model.predict(X_test))
