import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from util.pca import PCA
from util.knn import KNN, plot_knn_boundary
from utils.feature_selection import FeatureSelector
from utils.tSNE import TSNEAnalyzer

from util.hdbscan import HDBSCAN_Analyzer
from util.SVM import SVM, plot_svm_boundary
from utils.feature_extraction import FeatureExtractor
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np


class Main():

    def run(self):
        print("started")
        featureExtractor = FeatureExtractor()
        featureExtractor.read_from_file("features.csv")
        df = featureExtractor.get_data_frame()
        df = df.dropna()

        X_raw = df.drop('Label', axis=1).values
        y_labels = df['Label'].values

        x = df.drop("Label", axis=1)
        y = df["Label"]

        featureSelector = FeatureSelector()
        X_selected, selected_features, selector = featureSelector.select_k_best(x, y, k=15)

        scaler = StandardScaler()
        X_scaled_for_hdbscan = scaler.fit_transform(X_selected)
        hdbscan = HDBSCAN_Analyzer(min_cluster_size=50, min_samples=5)
        hdbscan_labels = hdbscan.fit(X_scaled_for_hdbscan)

        tsne_plotter = TSNEAnalyzer(n_components=2, standardize=False)

        X_tSNE_ = tsne_plotter.fit_transform(X_scaled_for_hdbscan, perplexity=30)

        hdbscan.plot_clusters(X_tSNE_)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_labels)

        tsne_analyzer = TSNEAnalyzer(n_components=2, standardize=True)
        search_result = tsne_analyzer.find_optimal_perplexity(X_selected, plot=True)
        print(tsne_analyzer.interpret_divergence(search_result))

        X_transform_tsne = tsne_analyzer.fit_transform(X_selected, perplexity=search_result.best_perplexity)

        tsne_analyzer.plot_embedding(X_transform_tsne, labels=y_encoded, title="Master t-SNE Drum Embeddings")

        X_train, X_test, y_train, y_test = train_test_split(
            X_transform_tsne, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
               
        acc_train = []
        acc_test = []
        
        K_range = range(1, 11)
        for kk in K_range:
            clf = KNeighborsClassifier(n_neighbors=kk)
            clf.fit(X_train, y_train)
            acc_train.append(clf.score(X_train, y_train))
            acc_test.append(clf.score(X_test, y_test))

        fig, ax = plt.subplots(figsize=(8, 5))
        plt.plot(K_range, acc_train, marker='o', label='Training Accuracy')
        plt.plot(K_range, acc_test, marker='s', label='Validation Accuracy')
        plt.xlabel('Number of Neighbors (K)')
        plt.ylabel('Accuracy Score')
        plt.title('KNN: Performance vs. K-Value')
        plt.legend()
        plt.grid(True)
        plt.show()

        best_k = K_range[np.argmax(acc_test)]
        print(f"Optimal K identified: {best_k}")



        knn_raw = KNN(n_neighbors=5)
        knn_raw.fit(X_train, y_train)
        knn_raw.evaluate(X_test, y_test)

        knn_visualization = KNN(n_neighbors=5)
        knn_visualization.fit(X_train, y_train)

        svm = SVM()
        svm.svm_training(X_train, X_test, y_train, y_test)

        plot_knn_boundary(
            knn_visualization.model, 
            X_train, 
            y_train, 
            title=f"KNN (k={knn_visualization.n_neighbors}) Decision Boundaries"
        )

        plot_svm_boundary(
            svm.model, 
            X_train, 
            y_train, 
            title=f"SVM Decision Boundaries (PCA Space)\nLabels: {list(le.classes_)}"
        )
        

if __name__ == "__main__":
    main = Main()
    main.run()