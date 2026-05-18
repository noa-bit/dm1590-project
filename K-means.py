import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class KMeansClustering:
    def __init__(self, n_clusters=3):
        self.scaler = None
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=42
        )

    def train(self, X):

        self.model.fit(X)

        print("Cluster Centers:")
        print(self.model.cluster_centers_)

    def predict(self, X):
        return self.model.predict(X)


def plot_clusters(X, labels, centers=None, title="K-Means Clusters"):

    fig, ax = plt.subplots(figsize=(6, 5))

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap="tab10",
        edgecolor="k",
        s=40
    )

    # draw cluster centers
    if centers is not None:
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            s=250,
            marker="X",
            c="red",
            edgecolor="black",
            label="Centroids"
        )

    ax.set_title(title)
    ax.legend()
    plt.show()


if __name__ == "__main__":

    df = pd.read_csv("features.csv")

    # remove labels if they exist
    X_raw = df.drop('Label', axis=1).values

    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # PCA to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # train KMeans
    kmeans = KMeansClustering(n_clusters=3)
    kmeans.train(X_pca)

    # get cluster assignments
    cluster_labels = kmeans.predict(X_pca)

    # plot
    plot_clusters(
        X_pca,
        cluster_labels,
        centers=kmeans.model.cluster_centers_,
        title="K-Means on PCA-transformed data"
    )