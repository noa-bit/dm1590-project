import numpy as np 

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0

        X_scaled = (X - self.mean) / self.std

        matrix_cov= np.cov(X_scaled, rowvar=False) # covariance

        eigenvalues, eigenvectors = np.linalg.eig(matrix_cov)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        print(eigenvalues)
        index = np.argsort(eigenvalues)[::-1] # Sorts and reverses the eigenvalues
        eigenvalues = eigenvalues[index]
        eigenvectors = eigenvectors[:, index]

        # Select the top n_components
        self.components = eigenvectors[:, :self.n_components]
    
    def transform(self, X):
        if self.components is None:
            raise ValueError("PCA has not been fitted yet.")
        X_centered = (X - self.mean) / self.std
        return np.dot(X_centered, self.components)
