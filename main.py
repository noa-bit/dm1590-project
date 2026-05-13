import pandas as pd
from util.pca import PCA
from util.dbscan import DBSCAN_Analyzer 
def main():
    df = pd.read_csv('data.csv')

    X_raw = df.drop('Label', axis=1).values
    y_labels = df['Label'].values

    pca = PCA(good_stuff=2)
    pca.fit(X_raw)
    X_reduced = pca.transform(X_raw)

    dbscan = DBSCAN_Analyzer(eps=0.5, min_samples=5)

    dbscan.fit(X_reduced)

    dbscan.evaluate(X_reduced, y_true=y_labels)

    dbscan.plot_clusters(X_reduced)

    pca_df = pd.DataFrame(data=X_reduced, columns=['PC1', 'PC2'])
    pca_df['Label'] = y_labels

    print(pca_df.head())


    if __name__ == "__main__":
        main()