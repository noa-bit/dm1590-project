from sklearn.neighbors import KNeighborsClassifier


class KNN_Analyzer:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.model = None

    def fit(self, X, y):
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(X, y)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted")
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        if self.model is None:
            raise ValueError("Model has not been fitted")
        y_pred = self.predict(X)
        accuracy = (y_pred == y_true).mean()
        print(f"Accuracy: {accuracy:.2f}")