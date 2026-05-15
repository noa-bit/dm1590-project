import pandas as pd
from util.pca import PCA
from utils.feature_extraction import FeatureExtractor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils.feature_selection import FeatureSelector


class Main():

    def run(self):
        print("started")
        featureExtractor = FeatureExtractor()
        featureExtractor.read_from_file("features.csv")
        df = featureExtractor.get_data_frame()

        X = df.drop("Label", axis=1)
        y = df["Label"]
        y_labels = df['Label'].values

        featureSelector = FeatureSelector()
        X_selected, selected_features, selector = featureSelector.select_k_best(X, y, 10)

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

            plt.arrow(0, 0, x_val, y_val, color='r', alpha=0.8)"""
        plt.show()



if __name__ == "__main__":
    main = Main()
    main.run()