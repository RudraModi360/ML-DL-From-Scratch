import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go


class Simple_Linear_Regression:

    def __init__(self):
        self.m = None
        self.c = None

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        num = 0
        den = 0
        for i in range(X.shape[0]):
            num += (X[i] - X.mean()) * (y[i] - y.mean())
            den += (X[i] - X.mean()) * (X[i] - X.mean())
        self.m = num / den
        self.c = y.mean() - self.m * (X.mean())
        return np.array([self.m, self.c])

    def predict(self, X):
        return np.array(self.m * X + self.c)

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score


df = pd.read_csv("Datasets\\placement.csv")

X, y = df["cgpa"], df["package"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = Simple_Linear_Regression()
print(model.fit(X_train, y_train))
print(model.score(X_test, y_test))

fig = px.scatter(df, x="cgpa", y="package")
fig.show()

X_range = np.linspace(X.min(), X.max(), 100)
y_pred = model.predict(X_range)

line_fig = px.scatter(
    df, x="cgpa", y="package", title="CGPA vs Package with Regression Line"
)
line_fig.add_trace(
    go.Scatter(
        x=X_range,
        y=y_pred,
        mode="lines",
        name="Regression Line",
        line=dict(color="red"),
    )
)
line_fig.show()
