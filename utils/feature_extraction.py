import librosa
import pandas as pd
import json
import numpy as np

#Returns a line that can be appended to a pandas dataframe


class MissingDataError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"




class FeatureExtractor():

    def __init__(self):
        self.data = None
        self.dataframe = {}
        pass

    #Expects a dictionary with the label as key
    # and a list of filenames with that label as objects

    def set_test_data(self):
        with open("drum_classes.json") as file:
            self.data = json.load(file)


    def set_data(self, data) -> None:
        self.data = data
    

    def extract_all(self) -> None:
        """
        Builds the dict of extracted features
        
        """

        if (self.data == None):
            raise MissingDataError("Data not defined")
       

        # Base feature labels (without _mean / _std)
        feature_labels = [
            "chroma_stft",
            "chroma_cqt",
            "chroma_cens",
            "chroma_vqt",
            "melspectrogram",
            "mfcc",
            "rms",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_contrast",
            "spectral_flatness",
            "spectral_rolloff",
            "poly_features",
            "tonnetz",
            "zero_crossing_rate"
        ]

        # Reset dataframe dictionary
        self.dataframe = {}

        # Create the Label column
        self.dataframe["Label"] = []

        # For every feature, create:
        # feature_mean
        # feature_std
        #
        # Example:
        # chroma_stft_mean
        # chroma_stft_std
        for label in feature_labels:
            self.dataframe[f"{label}_mean"] = []
            self.dataframe[f"{label}_std"] = []

        # Extract features from every file
        for key in self.data:
            if not (key == "undefined"):
                
                for index, path in enumerate(self.data[key]):
                    print(f"key: {key}, index: {index}\n")
                    if index >= 30:
                        break
                    row = self.__extract_features(path, key)
                    if row:
                        # Append each value in the row to the matching key
                        # The row order must be:
                        # [Label, feature1_mean, feature1_std, feature2_mean, feature2_std, ...]
                        for column_name, value in zip(self.dataframe.keys(), row):
                            self.dataframe[column_name].append(value)

        
    
    def __extract_features(self, path, key):
        """
        Extracts audio features and returns a flat row containing:
        - Label
        - mean and std for each feature

        Example column names:
        chroma_stft_mean, chroma_stft_std, mfcc_mean, mfcc_std, etc.
        """

        # Load audio
        try:
            y, sr = librosa.load(f"data/sorted/{key}/{path}")

            # FFT size
            n_fft = 256

            # Trim signal so length is divisible by n_fft
            sample_len = len(y)
            max_samples = sample_len - (sample_len % n_fft)
            if max_samples == 0:
                return None
            y = y[:max_samples]

            # Dictionary containing all extracted features
            features = [
                librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft),
                librosa.feature.chroma_cqt(y=y, sr=sr),
                librosa.feature.chroma_cens(y=y, sr=sr),
                librosa.feature.chroma_vqt(y=y, sr=sr, intervals="equal"),
                librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft),
                librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft),
                librosa.feature.rms(y=y),
                librosa.feature.spectral_centroid(
                    y=y, sr=sr, n_fft=n_fft
                ),
                librosa.feature.spectral_bandwidth(
                    y=y, sr=sr, n_fft=n_fft
                ),
                librosa.feature.spectral_contrast(
                    y=y, sr=sr, n_fft=n_fft
                ),
                librosa.feature.spectral_flatness(
                    y=y, n_fft=n_fft
                ),
                librosa.feature.spectral_rolloff(
                    y=y, sr=sr, n_fft=n_fft
                ),
                librosa.feature.poly_features(
                    y=y, sr=sr, n_fft=n_fft
                ),
                librosa.feature.tonnetz(y=y, sr=sr),
                librosa.feature.zero_crossing_rate(y=y),
            ]

            # Start row with the label
            row = [key]

            # Add mean and std for every feature
            for value in features:
                row.append(self.__get_mean(value))
                row.append(self.__get_std(value))

            return row
        except:
            return None
    
    def __get_mean(self, arr):
        return np.asarray(arr).mean()
    

    def __get_std(self, arr):
        return np.asarray(arr).std()

    #returns a pandas dataframe with all the data
    def get_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.dataframe)


if __name__ =="__main__":
    test = FeatureExtractor()
    test.set_test_data()
    test.extract_all()
    print(test.get_data_frame())


    features = {}

# Spectral centroid

    