import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.feature_selection import FeatureSelector
from utils.pca import PCA


LABEL_COLORS = {
    "hi-hat": "tab:blue",
    "kick": "tab:orange",
    "snare": "tab:green",
}


def load_features(path: str = "features.csv") -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # features.csv is currently written with the dataframe index included.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")

    X = df.drop(columns="Label")
    y = df["Label"]
    return X, y


def project_train_and_test_to_same_pca_space():
    X, y = load_features()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split before fitting any preprocessing step so the test set stays genuinely unseen.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=X_train_raw.columns,
        index=X_train_raw.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=X_test_raw.columns,
        index=X_test_raw.index,
    )

    feature_selector = FeatureSelector()
    X_train_selected, selected_features, selector = feature_selector.fit_select_k_best(
        X_train_scaled,
        y_train,
        k=15,
    )
    X_test_selected = selector.transform(X_test_scaled)

    # One fitted PCA object = one shared set of axes for both datasets.
    pca = PCA(good_stuff=2)
    pca.fit(X_train_selected)
    X_train_pca = pca.transform(X_train_selected)
    X_test_pca = pca.transform(X_test_selected)

    return (
        X_train_pca,
        X_test_pca,
        y_train,
        y_test,
        label_encoder,
        selected_features,
    )


def plot_overlay():
    (
        X_train_pca,
        X_test_pca,
        y_train,
        y_test,
        label_encoder,
        selected_features,
    ) = project_train_and_test_to_same_pca_space()

    fig, ax = plt.subplots(figsize=(10, 7))

    for encoded_label, class_name in enumerate(label_encoder.classes_):
        train_mask = y_train == encoded_label
        test_mask = y_test == encoded_label
        color = LABEL_COLORS.get(class_name, "tab:gray")

        ax.scatter(
            X_train_pca[train_mask, 0],
            X_train_pca[train_mask, 1],
            color=color,
            alpha=0.45,
            s=26,
            marker="o",
        )
        ax.scatter(
            X_test_pca[test_mask, 0],
            X_test_pca[test_mask, 1],
            color=color,
            alpha=0.95,
            s=42,
            marker="x",
            linewidths=1.2,
        )

    class_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=LABEL_COLORS.get(class_name, "tab:gray"),
            label=class_name,
        )
        for class_name in label_encoder.classes_
    ]
    split_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="black",
            alpha=0.45,
            label="train",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="",
            color="black",
            label="test",
        ),
    ]

    first_legend = ax.legend(
        handles=class_legend,
        title="Class",
        loc="upper right",
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=split_legend,
        title="Dataset",
        loc="lower right",
    )

    ax.set_title("Train and test samples in the same 2D PCA space")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.grid(alpha=0.2)

    ax.text(
        0.01,
        0.01,
        "Selected features:\n" + "\n".join(selected_features),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    plt.tight_layout()
    plt.savefig("pca_train_test_overlay.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    plot_overlay()
