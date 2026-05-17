from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

class KNN:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.model = None

    def fit(self, X, y):
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(X, y)

    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model has not been fitted")
        
        y_prediction = self.model.predict(X_test)
        
        acc = accuracy_score(y_test, y_prediction)
        
        print(f"KNN Test Accuracy: {acc:.2f}")
        print(classification_report(y_test , y_prediction))
        
        return acc

def plot_knn_boundary(clf, X, y, title="KNN Decision Boundary"):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)

    fig, ax = plt.subplots(figsize=(10, 8))

    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.25,
        ax=ax,
        cmap="tab10"
    )
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", edgecolor="k", s=30)
    
    ax.set_title(title)
    plt.show()