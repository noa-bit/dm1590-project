import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics

class DBSCAN_Analyzer:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        
        self.model = None
        self.labels = None
        self.core_samples_mask = None
        self.n_clusters = 0

    def fit(self, X):
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        self.labels = self.model.labels_

        self.core_samples_mask = np.zeros_like(self.labels, dtype=bool)
        self.core_samples_mask[self.model.core_sample_indices_] = True

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

            xy_core = X[class_member_mask & self.core_samples_mask]
            plt.plot(xy_core[:, 0], xy_core[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=8)

            xy_edge = X[class_member_mask & ~self.core_samples_mask]
            plt.plot(xy_edge[:, 0], xy_edge[:, 1], 'X', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)

        plt.title(f'Estimated number of clusters: {self.n_clusters}')
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