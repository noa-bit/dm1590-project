from utils.feature_extraction import FeatureExtractor
class Main():

    def run(self):
        print("started")
        featureExtractor = FeatureExtractor()
        featureExtractor.set_test_data()
        featureExtractor.extract_all()
        df = featureExtractor.get_data_frame()

        X_raw = df.drop('Label', axis=1).values
        y_labels = df['Label'].values
        pca = PCA(good_stuff=2)





if __name__ == "__main__":
    print("Test")
    main = Main()
    main.run()





