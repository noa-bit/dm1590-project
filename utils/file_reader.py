import os
import shutil
from feature_extraction import FeatureExtractor
import pandas as pd
from pathlib import Path
from processing import save_filenames_json
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

@dataclass
class Result:
    raw: str
    category: str


class ManageTestData():
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
    
    def handleAndExtract(self) -> None:
        """
        Organizes all data into a 
        """
        while True:
            try:
                source_folder = input("Enter the name of the sample folder:")
                destination_folder = r"/data/sorted/unclassified"

                os.makedirs(destination_folder, exist_ok=True)
                results = []

                for filename in os.listdir(source_folder):
                    source_path = os.path.join(source_folder, filename)
                    dest_path = os.path.join(destination_folder, filename)

                    # Move only files (skip directories)
                    if os.path.isfile(source_path):
                        results.append(Result(raw=filename, category="unclassified"))
                        shutil.move(source_path, dest_path)

                save_filenames_json(results, "unclassified_sounds.json")

                print("All files moved successfully.")
                break     

            except Exception as error:
                print(f"Something went wrong. Details: {error}")

        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.set_test_data("unclassified_sounds.json")
        self.feature_extractor.extract_all()

    def standardize(self, scaler: StandardScaler, X: pd.DataFrame):
        X_scaled = pd.DataFrame(
                    scaler.fit_transform(X),
                    columns=X.columns,
                    index=X.index,
                )

        return scaler.transform(self.feature_extractor.get_data_frame(  ))