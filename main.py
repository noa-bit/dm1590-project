import pandas as pd
from util.pca import PCA

from util.dbscan import DBSCAN_Analyzer 

from utils.feature_extraction import FeatureExtractor
import matplotlib.pyplot as plt
class Main():

    def run(self):
        print("started")
        featureExtractor = FeatureExtractor()
        featureExtractor.set_test_data()
        featureExtractor.set_first_n(30)
        featureExtractor.extract_all()
        df = featureExtractor.get_data_frame()
        df = df.dropna()

        X_raw = df.drop('Label', axis=1).values
        y_labels = df['Label'].values
        pca = PCA(good_stuff=2)

        pca.fit(X_raw)
        x_proj = pca.compute_project(2, X_raw)
        print(x_proj[:, 0], x_proj[:, 1])
        plt.figure(figsize=(12, 10))
        print("trying to plot")
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
        X_reduced = pca.transform(X_raw)
        dbscan = DBSCAN_Analyzer(eps=0.5, min_samples=5)

        dbscan.fit(X_reduced)

        dbscan.evaluate(X_reduced, y_true=y_labels)

        dbscan.plot_clusters(X_reduced)
        """
        for d in range(24):
            x_val = pca.eigenvectors[:, :2][d, 0]
            y_val = pca.eigenvectors[:, :2][d, 1]

            plt.arrow(0, 0, x_val, y_val, color='r', alpha=0.8)"""
        plt.show()



if __name__ == "__main__":
    main = Main()
    main.run()
