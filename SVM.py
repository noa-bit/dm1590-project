import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from util.pca import PCA
import matplotlib.pyplot as plt

class SVM:
    def __init__(self):
        self.scaler = None
        self.model = None

    def svm_training(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.scaler = StandardScaler()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = SVC(kernel='rbf')

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    @staticmethod
    def plot_svm(model, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3)

        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("SVM Decision Boundary on PCA Data")
        plt.show()

df = pd.read_csv("features.csv")

X_raw = df.drop('Label', axis=1).values
y_labels = df['Label'].values

# PCA
pca = PCA(good_stuff=2)
pca.fit(X_raw)
X_pca = pca.transform(X_raw)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_labels,
    test_size=0.2,
    random_state=42,
    stratify=y_labels
)

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train SVM
model = SVC(kernel="rbf")
model.fit(X_train, y_train)

# plot
SVM.plot_svm(model, X_train, y_train)