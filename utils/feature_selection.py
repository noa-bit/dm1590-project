# utils/feature_selection.py
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

class FeatureSelector:
    def __init__(self):
        self.selector = None
    def fit_select_k_best(self, X, y, k: int=15):
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)

        selected_features = X.columns[selector.get_support()].tolist()
        print("Features selected are:")
        for feature in selected_features:
            print(feature)

        self.selector = selector

        return X_selected, selected_features, selector
    
    def transform(self, X):
        return self.selector.transform(X)
