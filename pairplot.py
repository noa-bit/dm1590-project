import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    df = pd.read_csv("features.csv")

    

    full_dataset = df.iloc[1:, 2:]

    sns.pairplot(full_dataset)
    plt.show()