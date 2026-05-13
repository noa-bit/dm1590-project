from utils.feature_extraction import FeatureExtractor
from util.pca import PCA
import matplotlib.pyplot as plt
class Main():

    def run(self):
        print("started")
        featureExtractor = FeatureExtractor()
        featureExtractor.set_test_data()
        featureExtractor.extract_all()
        df = featureExtractor.get_data_frame()

        X_raw = df.drop('Label', axis=1).values
        y_labels = df['Label'].values
        pca = PCA(good_stuff=2)

        pca.fit(X_raw)
        x_proj = pca.compute_project(2, X_raw)

        plt.figure(figsize=(12, 10))

        plt.scatter(x_proj[:, 0], x_proj[:, 1], alpha=0.3, color=lambda x: print(x), marker='.', )
        """
        for d in range(24):
            x_val = pca.eigenvectors[:, :2][d, 0]
            y_val = pca.eigenvectors[:, :2][d, 1]

            plt.arrow(0, 0, x_val, y_val, color='r', alpha=0.8)"""
        plt.show()







if __name__ == "__main__":
    print("Test")
    main = Main()
    main.run()





