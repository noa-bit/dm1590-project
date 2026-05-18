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

def move_to_folder(target_file, folder_name):
    source_path = os.path.join("/data/sorted/unclassified", target_file)
    os.makedirs(folder_name, exist_ok=True)

    dest_path = os.path.join(folder_name, target_file)
    
    shutil.move(source_path, dest_path)

def weighting(data):
    
    #testA


    #testB
    
    
    #testC
    
    
    
    """
    Om a==b eller a==c eller b==c eller a==b==c så vinner det alternativet
    returnera majoritets val
    
    om a != b != c så vinner a
    returnera a's resultat
    
    
    """




def sort_file(target_file, result):
    if result == "kick":
        move_to_folder(target_file, "/data/sorted/kicks-sorted")
    elif result == "snare":
        move_to_folder(target_file, "/data/sorted/snares-sorted")
    elif result == "hi-hat":
        move_to_folder(target_file, "/data/sorted/hi-hats-sorted")
    else:
        move_to_folder(target_file, "/data/sorted/unclassified")

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

#här sker massa skit med df som vi ska lösa
for row in df.itertuples(index=False):
    

    result = weighting(row)
    sort_file(LABEL FÖR FIL, result)