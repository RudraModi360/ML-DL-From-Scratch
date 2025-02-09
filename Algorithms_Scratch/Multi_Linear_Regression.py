import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


class Multi_Linear_Regression:
    def __init__(self):
        self.m = None
        self.c = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        X = np.c_[np.ones(X.shape[0]), X]
        coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
        self.c = coefficients[0]
        self.m = coefficients[1:]
        return coefficients

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.m) + self.c

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score


df = pd.read_csv("Datasets\\Real_Combine.csv")
df = df.dropna()
X, y = df.drop(["PM 2.5"], axis=1), df["PM 2.5"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = Multi_Linear_Regression()
model.fit(X_train, y_train)
print("R² score:", model.score(X_test, y_test))
