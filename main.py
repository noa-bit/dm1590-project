import pandas as pd
from sklearn.preprocessing import StandardScaler

import numpy as np

from utils.feature_extraction import FeatureExtractor
from utils.feature_selection import FeatureSelector
from utils.file_reader import TestDataManager
from utils.knn import KNN
from utils.pca import PCA
from utils.svm import SVM
from utils.random_forest import RandomForest

from training import Training
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
class Main():

    def run(self):
        scaler = StandardScaler()
        feature_extractor = FeatureExtractor()
        feature_selector = FeatureSelector()
        knn = KNN()
        pca = PCA(good_stuff=6)
        svm = SVM()
        rf = RandomForest()

        models = Training(
            kNN=knn, 
            SVM=svm, 
            RF=rf, 
            feature_extractor=feature_extractor, 
            feature_selector=feature_selector,
            scaler = scaler,
            pca= pca
        )
        models.run()

        k, s, r = models.get_trained_models()
        k:KNN
        s:SVM
        r:RandomForest

        
        manage_test_data = TestDataManager()

        manage_test_data.handle_and_extract()

        
        test_dataframe:pd.DataFrame = manage_test_data.get_new_df()
        test_dataframe = test_dataframe.dropna()
        print(test_dataframe)

        X = test_dataframe.drop("Label", axis=1)

        df_scaled:pd.DataFrame = manage_test_data.standardize(scaler=scaler, X =X)

        df_extracted: np.ndarray = feature_selector.transform(df_scaled)
        df_pca: np.ndarray = pca.transform(df_extracted)

        decode_dict = {
            0: "hi-hat",
            1: "kick",
            2: "snare"
        }
        
        for index, row in enumerate(df_pca):
            row = np.array([row])
            print("index: ", index, " with row transposed: ", row)
            res_knn = k.model.predict(row)[0]
            res_svm = s.model.predict(row)[0]
            res_rf = r.model.predict(row)[0]
            manage_test_data.sort_file(manage_test_data.feature_extractor.data["unclassified"][index], decode_dict[res_svm] )
            """ if (res_knn == res_svm) or (res_knn == res_rf):
                #kalla på knn rätt
                manage_test_data.sort_file(manage_test_data.feature_extractor.data["unclassified"][index], decode_dict[res_knn] )
            elif (res_svm == res_rf):
                manage_test_data.sort_file(manage_test_data.feature_extractor.data["unclassified"][index], decode_dict[res_svm] )
            else:
                manage_test_data.sort_file(manage_test_data.feature_extractor.data["unclassified"][index], decode_dict[res_knn] )
            """



if __name__ == "__main__":
    main = Main()
    main.run()