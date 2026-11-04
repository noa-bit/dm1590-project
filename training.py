import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils.feature_extraction import FeatureExtractor
from utils.feature_selection import FeatureSelector
from utils.pca import PCA
from utils.svm import SVM, plot_svm_boundary
from utils.knn import KNN, plot_knn_boundary
from utils.random_forest import RandomForest
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from standardize import find_audio_files, standardize
from processing import classify_samples, save_filenames_json


class Training():
    def __init__(
                self,
                kNN: KNN,
                SVM: SVM,
                RF: RandomForest,
                feature_extractor: FeatureExtractor,
                feature_selector: FeatureSelector,
                scaler: StandardScaler,
                pca: PCA,
                label_encoder: LabelEncoder
            ):
        self.kNN = kNN
        self.SVM = SVM
        self.RF = RF
        self.feature_extractor = feature_extractor
        self.feature_selector = feature_selector
        self.scaler = scaler
        self.pca = pca
        self.label_encoder = label_encoder

    def pca_transform(self, X, good_stuff=5):
            self.pca.fit(X)
            return self.pca.transform(X)

    def train_models(self, X_train, X_test, y_train, y_test):
        print("kNN:")
        self.kNN.training(X_train, X_test, y_train, y_test)

        print("SVM:")
        self.SVM.training(X_train, X_test, y_train, y_test)

        print("RF:")
        self.RF.training(X_train, X_test, y_train, y_test)

    def get_trained_models(self):
        return self.kNN, self.SVM, self.RF

    def features_file_has_rows(self, path: str = "features.csv") -> bool:
        if not os.path.exists(path):
            return False

        try:
            return not pd.read_csv(path).empty
        except pd.errors.EmptyDataError:
            return False

    def run(self) -> None:
        print("started")

        # **Preprocessing / Pretraining**
        features_ready = self.features_file_has_rows()

        if not features_ready:
                
            # Standardize audio files
            src_dir = os.path.join("data", "drums")
            out_dir = os.path.join("data", "processed")

            audio_files = list(find_audio_files(src_dir))
            print(f"Found {len(audio_files)} audio files in '{src_dir}'")
            processed_files = []

            for i, src_path in enumerate(audio_files, 1):
                try:
                    out_path = standardize(src_path, out_dir, src_dir)
                    processed_files.append(out_path)
                    print(f"[{i}/{len(audio_files)}] {src_path} -> {out_path}")
                except Exception as e:
                    print(f"[{i}/{len(audio_files)}] SKIP {src_path}: {e}")

            with open("file_registry.txt", "w", encoding="utf-8") as file:
                for processed_path in processed_files:
                    file.write(f"{processed_path}\n")

            print("Done.")

            if not os.path.exists("drum_classes.json"):
                print("drum_classes.json not found. Creating it instead...")
                file_paths = []
                with open("file_registry.txt", encoding="utf-8") as file:
                    for line in file.readlines():
                        file_paths.append(line.strip())

                print(file_paths)
                results = classify_samples(file_paths, verbose=False)

                save_filenames_json(results)

        # Feature extraction
        featureExtractor = self.feature_extractor

        if features_ready:
            featureExtractor.read_from_file("features.csv")
        else:
            # Run feature extraction if the file is missing or has no rows yet.
            print("features.csv missing or empty. Running feature extraction.")
            try:
                featureExtractor.set_test_data("drum_classes.json")
                featureExtractor.extract_all()
                featureExtractor.write_to_file("features.csv")
            except Exception as error:
                print(f"Error setting test data: {error}")
                return

        df = featureExtractor.get_data_frame()

        X = df.drop("Label", axis=1)
        y = df["Label"]
        y_labels = df['Label'].values
        y_labels_encoded = self.label_encoder.fit_transform(y_labels)

        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_labels_encoded, test_size=0.2, random_state=42, stratify=y)

        # Feature selection
        X_selected, selected_features, selector = self.feature_selector.fit_select_k_best(X_train, y_train, 15)


        # Training

        self.train_models(X_train, X_test, y_train, y_test)

        self.pca.fit(X_selected)
        X_pca = self.pca.transform(X_selected)

        X_test_selected = selector.transform(X_test)
        X_test_pca = self.pca.transform(X_test_selected)
        self.train_models(X_pca, X_test_pca, y_train, y_test)
        print("finished")


if __name__ == "__main__":
    training = Training(
        KNN(),
        SVM(),
        RandomForest(),
        FeatureExtractor(),
        FeatureSelector(),
        StandardScaler(),
        PCA(good_stuff=6),
        LabelEncoder(),
    )
    training.run()

# **TRAINING.PY**

# **Data**
# Preprocessing, standardize
# Organize data into folders
# Feature extraction, feature selection, dimensionality reduction

# **Training**
# Train all (supervised, unsupervised) models on data

# **TESTING.PY**

# **Testing**
# Test all models on data, evaluate

# file_reader.py
