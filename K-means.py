import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utils.feature_selection import FeatureSelector
from utils.tSNE import TSNEAnalyzer

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
    df = pd.read_csv("features.csv").dropna()

    X_df = df.drop('Label', axis=1)
    y = df['Label']

    featureSelector = FeatureSelector()
    X_selected, selected_features, selector = featureSelector.select_k_best(X_df, y, k=15)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    kmeans = KMeansClustering(n_clusters=3)
    kmeans.train(X_scaled)
    cluster_labels = kmeans.predict(X_scaled)

    tsne = TSNEAnalyzer(n_components=2, standardize=False)
    X_tsne_2d = tsne.fit_transform(X_scaled, perplexity=30)

    plot_clusters(
        X_tsne_2d,
        cluster_labels,
        centers=None, 
        title="K-Means (15 Features) visualized via t-SNE"
    )