import pandas as pd
from sklearn.preprocessing import StandardScaler
from util.pca import PCA
import numpy as np

from util.dbscan import DBSCAN_Analyzer 

from utils.feature_extraction import FeatureExtractor
from utils.feature_selector import FeatureSelector
from utils.file_reader import ManageTestData
from utils.training import Training
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
class Main():

    def run(self):
        scaler = StandardScaler()
        feature_extractor = FeatureExtractor()
        feature_selector = FeatureSelector()
        pca = PCA()
        models = Training(feature_selector, scaler)

        models.train()


        SVM, kNN, randomForest = models.get_trained_models()




        manage_test_data = ManageTestData(feature_extractor=feature_extractor)
        manage_test_data.handleAndExtract()
        features_data_frame = manage_test_data.feature_extractor.get_data_frame()


        df_scaled:pd.DataFrame = manage_test_data.standardize(scaler=scaler, X =features_data_frame)

        df_extracted: np.ndarray = feature_selector.transform_select_k_best(df_scaled)
        df_pca: np.ndarray = pca.transform(df_extracted)

        for row in df_pca:
            svm.transform(row)


if __name__ == "__main__":
    main = Main()
    main.run()