import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from util.pca import PCA
import matplotlib.pyplot as plt


class SVM:
    def __init__(self):
        self.scaler = None
        self.model = None
        self.label_encoder = None

    def svm_training(self, X, y):

        # Encode labels (IMPORTANT for plotting + sklearn stability)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train SVM
        self.model = SVC(kernel='rbf')
        self.model.fit(X_train, y_train)

        # Predict
        y_pred = self.model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))


df = pd.read_csv("features.csv")

X_raw = df.drop('Label', axis=1).values
y_labels = df['Label'].values

scaler = StandardScaler()
X_scaled_for_dbscan = scaler.fit_transform(X_raw)

svm = SVM()
svm.svm_training(X_scaled_for_dbscan, y_labels)

pca = PCA(good_stuff=2)
pca.fit(X_scaled_for_dbscan)
X_pca = pca.transform(X_scaled_for_dbscan)
svm.svm_training(X_pca, y_labels)

print(X_pca)