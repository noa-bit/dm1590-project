import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

from utils.feature_selection import FeatureSelector
from utils.tSNE import TSNEAnalyzer

class RandomForest:
    def __init__(self):
        self.scaler = None
        self.model = None
        self.label_encoder = LabelEncoder()

    def rf_training(self, X, y):
        y = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def plot_feature_importance(self, feature_names=None):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(importances)), importances[indices])
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
        ax.set_title("Random Forest Feature Importances")
        ax.set_ylabel("Importance")
        plt.tight_layout()
        plt.show()

df = pd.read_csv("features.csv", index_col=0)

X_raw = df.drop('Label', axis=1).values
y_labels = df['Label'].values
feature_names = df.drop('Label', axis=1).columns.tolist()

x = df.drop("Label", axis=1)
y = df["Label"]
featureSelector = FeatureSelector()
X_selected, selected_features, selector = featureSelector.select_k_best(x, y, k=5)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)

tsne_analyzer = TSNEAnalyzer(n_components=2, standardize=True)
search_result = tsne_analyzer.find_optimal_perplexity(X_selected, plot=True)
print(tsne_analyzer.interpret_divergence(search_result))

X_transform_tsne = tsne_analyzer.fit_transform(X_selected, perplexity=search_result.best_perplexity)

tsne_analyzer.plot_embedding(X_transform_tsne, labels=y_encoded, title="Master t-SNE Drum Embeddings")

X_train, X_test, y_train, y_test = train_test_split(
    X_transform_tsne, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

rf = RandomForest()
rf.rf_training(X_train, y_train)
rf.plot_feature_importance(feature_names)