import pandas as pd
from util.pca import PCA

def main():
    df = pd.read_csv('data.csv')

    X_raw = df.drop('Label', axis=1).values
    y_labels = df['Label'].values

    pca = PCA(good_stuff=2)

    pca.fit(X_raw)
    X_reduced = pca.transform(X_raw)

    pca_df = pd.DataFrame(data=X_reduced, columns=['PC1', 'PC2'])
    pca_df['Label'] = y_labels

    print(pca_df.head())


    if __name__ == "__main__":
        main()