import os
import shutil
from utils.feature_extraction import FeatureExtractor
import pandas as pd
from pathlib import Path
from processing import save_filenames_json
from dataclasses import dataclass

@dataclass
class Result:
    raw: str
    category: str

def find_folder(target_name, search_root):
    search_root = Path(search_root)
    for path in search_root.rglob("*"):
        if path.is_dir() and path.name == target_name:
            return path
    return None

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

feature_extractor = FeatureExtractor()
feature_extractor.set_test_data("unclassified_sounds.json")
feature_extractor.extract_all()
df = feature_extractor.get_data_frame()