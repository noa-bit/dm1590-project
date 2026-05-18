import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn import metrics

class HDBSCAN_Analyzer:
    def __init__(self, min_cluster_size=15, min_samples=None):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        
        self.model = None
        self.labels = None
        self.probabilities = None
        self.n_clusters = 0

    def fit(self, X):
        self.model = HDBSCAN(
            min_cluster_size=self.min_cluster_size, 
            min_samples=self.min_samples
        ).fit(X)
        
        self.labels = self.model.labels_
        self.probabilities = self.model.probabilities_

        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        return self.labels

    def plot_clusters(self, X):
        if self.labels is None:
            raise ValueError("Model has not been fitted")

        unique_labels = set(self.labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1] 

            class_member_mask = (self.labels == k)

            confident_mask = (self.probabilities >= 0.5) if k != -1 else np.zeros_like(self.labels, dtype=bool)

            xy_confident = X[class_member_mask & confident_mask]
            plt.scatter(xy_confident[:, 0], xy_confident[:, 1], marker='o', 
                        facecolor=tuple(col), edgecolor='k', s=50)

            xy_uncertain = X[class_member_mask & ~confident_mask]
            plt.scatter(xy_uncertain[:, 0], xy_uncertain[:, 1], marker='X', 
                        facecolor=tuple(col), edgecolor='k', s=30)

        plt.title(f'HDBSCAN Estimated number of clusters: {self.n_clusters}')
        plt.show()

    def evaluate(self, X, y_true=None):
        if self.labels is None:
            raise ValueError("Model has not been fitted")

        unique_labels = set(self.labels)
        silhouette_score = None

        if len(unique_labels) < 2:
            print("Less than 2 clusters found")
        else:
            silhouette_score = metrics.silhouette_score(X, self.labels)
            print(f"Silhouette Coefficient: {silhouette_score:.2f}")

        if y_true is not None:
            adjusted_rand_score = metrics.adjusted_rand_score(y_true, self.labels)
            print(f"Adjusted Rand Index: {adjusted_rand_score:.2f}")
            return silhouette_score, adjusted_rand_score
            
        return silhouette_score