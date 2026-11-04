import os
import shutil
from utils.feature_extraction import FeatureExtractor
import pandas as pd
from pathlib import Path
from processing import save_filenames_json
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

@dataclass
class Result:
    raw: str
    category: str



class TestDataManager():

    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def move_to_folder(self, target_file, folder_name):
        source_path = os.path.join("data/sorted/unclassified/", target_file)
        os.makedirs(folder_name, exist_ok=True)

        dest_path = os.path.join(folder_name, target_file)
        
        shutil.move(source_path, dest_path)

    def sort_file(self, target_file, result):
        print("result: ", result, " targetfile: ", target_file)

        return
        if result == "kick":
            self.move_to_folder(target_file, "data/sorted/kicks-sorted/")
        elif result == "snare":
            self.move_to_folder(target_file, "data/sorted/snares-sorted/")
        elif result == "hi-hat":
            self.move_to_folder(target_file, "data/sorted/hi-hats-sorted/")
        else:
            self.move_to_folder(target_file, "data/sorted/unclassified/")

    

    def handle_and_extract(self):
        while True:
            try:
                source_folder = input("Enter the name of the sample folder:")
                destination_folder = r"data/sorted/unclassified"

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


        self.feature_extractor.set_test_data("unclassified_sounds.json")
        self.feature_extractor.extract_all()

    def get_new_df(self):
        return self.feature_extractor.get_data_frame()
    #här sker massa skit med df som vi ska lösa
    def standardize(self, scaler: StandardScaler, X: pd.DataFrame):
        
        print(self.feature_extractor.get_data_frame())
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )
        return X_scaled