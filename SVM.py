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

    def svm_training(self, X, y):

        # IMPORTANT: encode ONCE consistently
        y = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = SVC(kernel='rbf')
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

def plot_svm_boundary(clf, X, y, title="SVM Decision Boundary"):

    X = np.asarray(X, dtype=float)

    # FIX: ensure numeric labels (NOT strings)
    if isinstance(y[0], str):
        raise ValueError("y is still string labels — encode them first!")

    fig, ax = plt.subplots(figsize=(6, 5))

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


df = pd.read_csv("features.csv")

X_raw = df.drop('Label', axis=1).values
y_labels = df['Label'].values

# scale original data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

svm = SVM() 

svm.svm_training(X_scaled, y_labels)

y_encoded = svm.label_encoder.transform(y_labels)


from util.pca import PCA

pca = PCA(good_stuff=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)


clf = SVC(kernel="rbf", gamma=2)
clf.fit(X_pca, y_encoded)


plot_svm_boundary(clf, X_pca, y_encoded, title="SVM on PCA-transformed data")