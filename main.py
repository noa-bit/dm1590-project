import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

import numpy as np
import json
import re

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


def infer_label_from_filename(filename: str) -> str | None:
    label_patterns = {
        "hi-hat": re.compile(r"hi[-_\s]?hat", re.IGNORECASE),
        "kick": re.compile(r"kick", re.IGNORECASE),
        "snare": re.compile(r"snare", re.IGNORECASE),
    }

    for label, pattern in label_patterns.items():
        if pattern.search(filename):
            return label

    return None


class Main():

    def run(self):
        scaler = StandardScaler()
        feature_extractor = FeatureExtractor()
        feature_selector = FeatureSelector()
        knn = KNN()
        pca = PCA(good_stuff=6)
        svm = SVM()
        rf = RandomForest()
        LE = LabelEncoder()

        models = Training(
            kNN=knn, 
            SVM=svm, 
            RF=rf, 
            feature_extractor=feature_extractor, 
            feature_selector=feature_selector,
            scaler = scaler,
            pca= pca,
            label_encoder=LE
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

        results = []

        for index, row in enumerate(df_pca):
            row = np.array([row])
            print("index: ", index, " with row transposed: ", row)
            res_knn = k.model.predict(row)[0]
            res_svm = s.model.predict(row)[0]
            res_rf = r.model.predict(row)[0]

            if (res_knn == res_svm) or (res_knn == res_rf):
                majority_result = res_knn
            elif res_svm == res_rf:
                majority_result = res_svm
            else:
                # If all three models disagree, there is no majority.
                # Keep SVM as the tie-breaker for now, matching the old behavior.
                majority_result = res_rf
            majority_result = res_rf

            decoded_result = LE.inverse_transform([majority_result])[0]
            filename = manage_test_data.feature_extractor.data["unclassified"][index]
            expected_result = infer_label_from_filename(filename)

            results.append(
                {
                    "filename": filename,
                    "predicted_label": decoded_result,
                    "expected_label_from_filename": expected_result,
                    "correct": decoded_result == expected_result if expected_result else None,
                }
            )

            manage_test_data.sort_file(
                filename,
                decoded_result,
            )

        with open("classification_results.json", "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)

        evaluable_results = [result for result in results if result["expected_label_from_filename"]]
        correct_results = [result for result in evaluable_results if result["correct"]]

        if evaluable_results:
            accuracy = len(correct_results) / len(evaluable_results)
            print(
                f"Regex filename accuracy: {accuracy:.2%} "
                f"({len(correct_results)}/{len(evaluable_results)})"
            )
        else:
            print("Regex filename accuracy: no evaluable filenames found.")



if __name__ == "__main__":
    main = Main()
    main.run()
