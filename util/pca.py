import numpy as np 


class PCA:
    def __init__(self, good_stuff):
        self.good_stuff = good_stuff
        self.components = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean # centering the data

        matrix_cov= np.cov(X_centered, rowvar=False) # covariance

        eigenvalues, eigenvectors = np.linalg.eig(matrix_cov)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        print(eigenvalues)
        index = np.argsort(eigenvalues)[::-1] # Sorts and reverses the eigenvalues
        eigenvalues = eigenvalues[index]
        eigenvectors = eigenvectors[:, index]

        # Select the top good_stuff
        self.components = eigenvectors[:, :self.good_stuff]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    

    def compute_project(self, dims: int, data) -> np.ndarray:
        data_centered = data - np.mean(data, axis=0)

        print(data_centered.shape)
        vecs_to_display = self.eigenvectors[:, :dims]
        print(len(vecs_to_display.T))
        return np.array(np.matmul(vecs_to_display.T, data_centered.T).T)