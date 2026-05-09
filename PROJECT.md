We will build a drum sound classification system that automatically sorts electronic drum samples into categories such as kick, snare, hi-hat, and other percussion types. Before training we will preprocess and segment drum audio samples, extract audio features using Librosa apply feature engineering (e.g., spectral and temporal descriptors), extract the meaningful features using backward feature selection and then performing further dimensionality reduction with PCA. With the processed data we will train supervised machine learning models to classify drum sounds. The final output will be a tool that can take unknown drum samples and correctly label them by type. The primary dataset used will be the 200 Drummachines dataset (https://www.hexawe.net/mess/200.Drum.Machines from https://ismir.net/resources/datasets) This dataset contains a large collection of electronic drum machine sounds. Additional test data will be created using Logic Pro X’s stock electronic drum kits to test accuracy on data outside of the original dataset. All audio data will be: standardized (sample rate, mono conversion) segmented into individual drum hits and where necessary split into training, validation, and test sets.






DATA

Segment audio files into 4 categories: kick, snare, hi-hat and percussion (all other). We will write a processing script to classify these by names first, then by manual review.