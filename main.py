import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from util import pca
from util.pca import PCA

from util.dbscan import DBSCAN_Analyzer 
from util.SVM import SVM, plot_svm_boundary
from utils.feature_extraction import FeatureExtractor
import matplotlib.pyplot as plt
class Main():

    def run(self):
        print("started")
        featureExtractor = FeatureExtractor()
        featureExtractor.read_from_file("features.csv")
        df = featureExtractor.get_data_frame()
        df = df.dropna()

        X_raw = df.drop('Label', axis=1).values
        y_labels = df['Label'].values

        scaler = StandardScaler()
        X_scaled_for_dbscan = scaler.fit_transform(X_raw)

        dbscan = DBSCAN_Analyzer(eps=4.0, min_samples=5)
        dbscan_labels = dbscan.fit(X_scaled_for_dbscan)

        clean_mask = (dbscan_labels != -1)

        X_clean = X_raw[clean_mask]
        y_clean = y_labels[clean_mask]

        pca = PCA(n_components=2)

        pca.fit(X_clean) 
        X_train_pca = pca.transform(X_clean)
        X_reduced_clean = pca.transform(X_clean)
        print(X_reduced_clean[:, 0], X_reduced_clean[:, 1])
        plt.figure(figsize=(12, 10))
        label_names = sorted(set(y_clean))
        label_to_color = {label: index for index, label in enumerate(label_names)}
        colors = [label_to_color[label] for label in y_clean]
        
        scatter = plt.scatter(
            X_reduced_clean[:, 0],
            X_reduced_clean[:, 1],
            alpha=0.7,
            c=colors,
            cmap='tab10',
            marker='o',
            edgecolors='k'
        )
        
        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=label_names,
            title='Label',
        )

        svm = SVM() 
        X_train, y_train_encoded = svm.svm_training(X_train_pca, y_clean)

        plot_svm_boundary(
            svm.model, 
            X_train, 
            y_train_encoded, 
            title="SVM on PCA-transformed data"
        )
        

if __name__ == "__main__":
    main = Main()
    main.run()