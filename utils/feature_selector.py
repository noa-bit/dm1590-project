# utils/feature_selection.py
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

class FeatureSelector:
    def __init__(self):
        self.selector: SelectKBest
        self.selected_features = list[str]

    def fit_select_k_best(self, X, y, k: int):
        selector = SelectKBest(score_func=f_classif, k=k)

        self.selector = selector
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        print("Features selected are:")
        for feature in selected_features:
            print(feature)


        self.selected_features = selected_features
        
        return X_selected, selected_features, selector
    
    def transform_select_k_best(self, X) ->  np.ndarray:
        return self.selector.transform(X)
