import librosa
import pandas as pd
import json
import numpy as np

#Returns a line that can be appended to a pandas dataframe




class FeatureExtractor():

    def __init__(self):
        self.data = None
        pass

    #Expects a dictionary with the label as key
    # and a list of filenames with that label as objects
    def set_data(self, data):
        self.data = data
    
    def set_sample_rate(self, rate):
        self.sample_rate = rate
    
    def extract_all(self):
        
        with open("dummy_data.json") as file:
            self.data = json.load(file)
        
        df = pd.DataFrame(columns= [
            "Label",
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
        ])
        
        for key in self.data:
            for path in self.data[key]:
                row = self.__extract_features(path, key)
                df[len(df)] = row
        self.dataframe = df
        
    
    def __extract_features(self, path, key):
        
        y, sr = librosa.load(f"data/raw/{path}")
        V = np.abs(librosa.vqt(y=y, sr=sr))
            
        row = {
            "Label": f"{key}",
            "chroma_stft": librosa.feature.chroma_stft(y=y, sr=sr),
            "chroma_cqt": librosa.feature.chroma_cqt(y=y, sr=sr ),
            "chroma_cens": librosa.feature.chroma_cens(y=y, sr=sr),
            "chroma_vqt": librosa.feature.chroma_vqt(y=y, sr=sr, intervals="equal"),
            "melspectrogram": librosa.feature.melspectrogram(y=y, sr=sr),
            "mfcc": librosa.feature.mfcc(y=y, sr=sr),
            "rms": librosa.feature.rms(y=y),
            "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
            "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr),
            "spectral_contrast": librosa.feature.spectral_contrast(y=y, sr=sr),
            "spectral_flatness": librosa.feature.spectral_flatness(y=y),
            "spectral_rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr),
            "poly_features": librosa.feature.poly_features(y=y, sr=sr),
            "tonnetz": librosa.feature.tonnetz(y=y, sr=sr),
            "zero_crossing_ratve": librosa.feature.zero_crossing_rate(y=y)
        }
        return row
    
    
    def getDataFrame(self):
        return self.dataframe


if __name__ =="__main__":
    test = FeatureExtractor()
    test.extract_all()
    print(test.getDataFrame())
    