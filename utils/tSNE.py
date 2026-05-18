from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


@dataclass
class PerplexitySearchResult:
    """Container for the output of a t-SNE perplexity search."""

    perplexities: list
    divergences: list
    best_perplexity: float
    best_divergence: float


class TSNEAnalyzer:
    """Small helper for tuning, fitting, and plotting t-SNE embeddings."""

    def __init__(
        self,
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        standardize=True,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.init = init
        self.learning_rate = learning_rate
        self.standardize = standardize
        self.scaler = None
        self.embedding_ = None
        self.model_ = None

    @staticmethod
    def features_from_extractor(feature_extractor, label_column="Label"):
        """Return numeric features and labels from a FeatureExtractor instance."""
        df = feature_extractor.get_data_frame().dropna()
        if label_column not in df.columns:
            raise ValueError(f"Expected label column {label_column!r} in extracted dataframe")
        return df.drop(columns=[label_column]), df[label_column].to_numpy()

    def _prepare_features(self, features):
        """Convert dataframe/array-like input to a numeric matrix."""
        X = np.asarray(features, dtype=float)
        if X.ndim != 2:
            raise ValueError("features must be a 2D matrix")
        if X.shape[0] < 3:
            raise ValueError("t-SNE needs at least 3 samples")

        if self.standardize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        return X

    @staticmethod
    def _valid_perplexities(perplexities, n_samples):
        """Keep only values allowed by sklearn: perplexity must be < n_samples."""
        valid = sorted({float(p) for p in perplexities if 0 < float(p) < n_samples})
        if not valid:
            raise ValueError(
                f"No valid perplexities. Every perplexity must be > 0 and < n_samples ({n_samples})."
            )
        return valid

    def find_optimal_perplexity(self, features, perplexities=None, plot=False):
        """
        Fit several t-SNE models and choose the perplexity with the lowest KL divergence.

        Lower KL divergence means the 2D map preserves the high-dimensional neighborhood
        relationships more faithfully. It is useful for comparison across perplexities,
        but it is not an absolute quality score by itself.
        """
        X = self._prepare_features(features)
        if perplexities is None:
            upper = min(50, X.shape[0] - 1)
            perplexities = range(5, upper + 1, 5)

        valid_perplexities = self._valid_perplexities(perplexities, X.shape[0])
        divergences = []

        for perplexity in valid_perplexities:
            model = TSNE(
                n_components=self.n_components,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=self.random_state,
            )
            model.fit_transform(X)
            divergences.append(float(model.kl_divergence_))

        best_index = int(np.argmin(divergences))
        result = PerplexitySearchResult(
            perplexities=valid_perplexities,
            divergences=divergences,
            best_perplexity=valid_perplexities[best_index],
            best_divergence=divergences[best_index],
        )

        if plot:
            self.plot_divergence(result)

        return result

    @staticmethod
    def interpret_divergence(result):
        """Return a short human-readable interpretation of the search result."""
        best = result.best_perplexity
        best_div = result.best_divergence
        spread = max(result.divergences) - min(result.divergences)

        if spread < 0.05:
            stability = "The curve is fairly flat, so several perplexities behave similarly."
        else:
            stability = "The curve changes noticeably, so perplexity meaningfully affects the embedding."

        return (
            f"Best tested perplexity: {best:g} with KL divergence {best_div:.4f}. "
            "Lower KL divergence means the 2D embedding better preserves local structure "
            "from the original feature space among the values tested. "
            f"{stability}"
        )

    def fit_transform(self, features, perplexity):
        """Run the actual t-SNE reduction using a chosen perplexity."""
        X = self._prepare_features(features)
        valid_perplexity = self._valid_perplexities([perplexity], X.shape[0])[0]

        self.model_ = TSNE(
            n_components=self.n_components,
            perplexity=valid_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=self.random_state,
        )
        self.embedding_ = self.model_.fit_transform(X)
        return self.embedding_

    @staticmethod
    def plot_divergence(result):
        """Plot perplexity against KL divergence using matplotlib."""
        plt.figure(figsize=(8, 5))
        plt.plot(result.perplexities, result.divergences, marker="o")
        plt.scatter(
            [result.best_perplexity],
            [result.best_divergence],
            color="red",
            label=f"best = {result.best_perplexity:g}",
            zorder=3,
        )
        plt.xlabel("Perplexity")
        plt.ylabel("KL divergence")
        plt.title("t-SNE perplexity search")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_embedding(embedding, labels=None, title="t-SNE embedding"):
        """Plot a 2D t-SNE embedding with optional class labels."""
        if embedding.ndim != 2 or embedding.shape[1] != 2:
            raise ValueError("plot_embedding expects an array with shape (n_samples, 2)")

        plt.figure(figsize=(8, 6))

        if labels is None:
            plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
        else:
            labels = np.asarray(labels)
            unique_labels = sorted(set(labels.tolist()))
            for label in unique_labels:
                mask = labels == label
                plt.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    alpha=0.7,
                    label=str(label),
                )
            plt.legend(title="Label")

        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


tSNE = TSNEAnalyzer


if __name__ == "__main__":
    from feature_extraction import FeatureExtractor

    feature_extractor = FeatureExtractor()
    feature_extractor.read_from_file("features.csv")

    tsne = TSNEAnalyzer()
    X, labels = tsne.features_from_extractor(feature_extractor)

    search_result = tsne.find_optimal_perplexity(X, plot=True)
    print(tsne.interpret_divergence(search_result))

    embedding = tsne.fit_transform(X, search_result.best_perplexity)
    tsne.plot_embedding(embedding, labels)