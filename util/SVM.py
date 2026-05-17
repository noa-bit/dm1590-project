import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt


class SVM:
    def __init__(self):
        self.scaler = None
        self.model = None
        self.label_encoder = LabelEncoder()

    def svm_training(self, X_train, X_test, y_train, y_test):

        self.model = SVC(kernel='rbf')
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print("SVM Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        return acc
    

def plot_svm_boundary(clf, X, y, title="SVM Decision Boundary"):

    X = np.asarray(X, dtype=float)

    # FIX: ensure numeric labels (NOT strings)
    y = np.asarray(y).astype(int)

    fig, ax = plt.subplots(figsize=(10, 8))

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.25,
        ax=ax,
    )

    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors="k",
        ax=ax,
    )

    # FIX: safe coloring (encoded labels)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", edgecolor="k", s=30)

    ax.set_title(title)
    plt.show()
