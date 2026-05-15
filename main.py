import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from util.pca import PCA
from util.knn import KNN, plot_knn_boundary

from util.dbscan import DBSCAN_Analyzer 
from util.SVM import SVM, plot_svm_boundary
from utils.feature_extraction import FeatureExtractor
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.feature_selection import FeatureSelector


class Main():

    def run(self):
        print("started")
        featureExtractor = FeatureExtractor()
        featureExtractor.read_from_file("features.csv")
        df = featureExtractor.get_data_frame()
        df = df.dropna()

        X = df.drop("Label", axis=1)
        y = df["Label"]
        y_labels = df['Label'].values
        
        featureSelector = FeatureSelector()
        X_selected, selected_features, selector = featureSelector.select_k_best(X, y, 5)

        scaler = StandardScaler()
        X_selected_scaled = scaler.fit_transform(X_selected)

        pca = PCA(good_stuff=2)
        pca.fit(X_selected_scaled)
        x_proj = pca.transform(X_selected_scaled)

        label_names = sorted(set(y_labels))
        label_to_color = {label: index for index, label in enumerate(label_names)}
        colors = [label_to_color[label] for label in y_labels]
        scatter = plt.scatter(
            x_proj[:, 0],
            x_proj[:, 1],
            alpha=0.3,
            c=colors,
            cmap='tab10',
            marker='.',
        )
        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=label_names,
            title='Label',
        )
        plt.show()
        return


        y_labels = df['Label'].values
        pca = PCA(good_stuff=2)
        
        pca.fit(x_scaled)
        x_proj = pca.transform(x_scaled)
        print(x_proj[:, 0], x_proj[:, 1])
        plt.figure(figsize=(12, 10))
        print("trying to plot")

        plt.plot(range(1, len(pca.eigenvalues) + 1), pca.eigenvalues, marker='o', linestyle='-', color='b')

        plt.show()


        label_names = sorted(set(y_labels))
        label_to_color = {label: index for index, label in enumerate(label_names)}
        colors = [label_to_color[label] for label in y_labels]
        scatter = plt.scatter(
            x_proj[:, 0],
            x_proj[:, 1],
            alpha=0.3,
            c=colors,
            cmap='tab10',
            marker='.',
        )
        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=label_names,
            title='Label',
        )
        """
        for d in range(24):
            x_val = pca.eigenvectors[:, :2][d, 0]
            y_val = pca.eigenvectors[:, :2][d, 1]

        scaler = StandardScaler()
        X_scaled_for_dbscan = scaler.fit_transform(X_raw)
        dbscan = DBSCAN_Analyzer(eps=4.0, min_samples=5)
        dbscan_labels = dbscan.fit(X_scaled_for_dbscan)
        clean_mask = (dbscan_labels != -1)

        X_clean = X_raw[clean_mask]
        y_clean = y_labels[clean_mask]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_clean)

        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)       
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
        
        pca = PCA(n_components=2)
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        knn_raw = KNN(n_neighbors=5)
        knn_raw.fit(X_train, y_train)
        knn_raw.evaluate(X_test, y_test)

        knn = KNN(n_neighbors=5)
        knn.fit(X_train_pca, y_train)

        svm_inst = SVM()
        svm_inst.svm_training(X_train_pca, X_test_pca, y_train, y_test)

        plot_knn_boundary(
            knn.model, 
            X_train_pca, 
            y_train, 
            title=f"KNN (k={knn.n_neighbors}) Decision Boundaries"
        )

        plot_svm_boundary(
            svm_inst.model, 
            X_train_pca, 
            y_train, 
            title=f"SVM Decision Boundaries (PCA Space)\nLabels: {list(le.classes_)}"
        )
        

if __name__ == "__main__":
    main = Main()
    main.run()