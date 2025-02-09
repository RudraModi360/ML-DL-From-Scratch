import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class Clustering:
    def __init__(self, clusters=2, epochs=200):
        self.epochs = epochs
        self.clusters = clusters
        self.centroids = None

    def fit(self, X):
        self.centroids = X[random.sample(range(0, X.shape[0]), self.clusters)]
        for i in range(self.epochs):
            cluster_group = self.assign_centroids(X)
            old_centroids = self.centroids
            self.centroids = self.update_clusters(X, cluster_group)
            if np.array_equal(self.centroids, old_centroids):
                break
        return self.centroids

    def predict(self, X):
        return self.assign_centroids(X)

    def assign_centroids(self, X):
        centroids_group = []
        distance = []
        for row in X:
            for centroid in self.centroids:
                distance.append(np.dot(row - centroid, row - centroid))
            centroids_group.append(distance.index(min(distance)))
            distance.clear()
        return centroids_group

    def update_clusters(self, X, cluster_group):
        centroids = []
        clusters = np.unique(cluster_group)
        for cluster in clusters:
            centroids.append(np.array(X[cluster_group == cluster].mean(axis=0)))
        return centroids


df = pd.read_csv("Datasets\\placement-dataset.csv")
df = df.dropna()
df = df.drop(["Unnamed: 0"], axis=1)

X, y = np.array(df.drop(["placement"], axis=1)), np.array(df["placement"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = Clustering(clusters=3)
print(model.fit(X_train))
y_means = np.array(model.predict(X))

# Plotting the clusters
plt.figure(figsize=(10, 7))

# Assign unique colors for each cluster
colors = ["red", "blue", "green"]
for i in range(model.clusters):
    plt.scatter(
        X[y_means == i, 0], X[y_means == i, 1], color=colors[i], label=f"Cluster {i}"
    )

# Plot centroids
for i, centroid in enumerate(model.centroids):
    plt.scatter(
        centroid[0],
        centroid[1],
        color="black",
        marker="x",
        s=200,
        label=f"Centroid {i}",
    )

# Add titles and labels
plt.title("Clustering Results Visualization", fontsize=14)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
