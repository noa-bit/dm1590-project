import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

class SVM:
    def __init__(self, csv_name):
        # Load CSV
        self.df = pd.read_csv(csv_name)

        # Features and target
        self.X = self.df.drop('Label', axis=1).values 
        self.y = self.df["Label"].values

        # Placeholders
        self.scaler = None
        self.model = None

    def svm_training(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        self.scaler = StandardScaler()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = SVC(kernel='rbf')

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))


svm = SVM("features.csv")
svm.svm_training()

svm2 = PCA("features.csv")