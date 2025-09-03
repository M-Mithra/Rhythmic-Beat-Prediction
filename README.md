# Rhythmic Beat Prediction
Computational model to predict the rhythmic beat pattern of the song and fit them into a tala

This project focuses on predicting rhythmic beat patterns from audio files using both traditional machine learning models and deep learning (CNN). The workflow involves extracting features from audio, training models, and generating predictions.

## Features
- Audio feature extraction using `librosa` (Amplitude, Spectrograms, MFCCs, Chroma).
- Traditional models: Decision Tree and Random Forest classifiers.
- Deep learning: Convolutional Neural Network (CNN).
- Label encoding and decoding for human-readable class predictions.
- Dataset and utilities provided for further experimentation.

## Requirements
- Python 3.x
- Install dependencies:
  ```bash
  pip install torch librosa scikit-learn joblib numpy
````
````
## How to Run

1. Clone this repository and ensure the dataset and model files are in place.
2. Run the main script:

   ```bash
   python scripts/test.py
   ```

   or use the top-level `test.py` for testing.
3. Provide the path to an audio file when prompted.
4. The script will:

   * Extract features
   * Run predictions using Decision Tree, Random Forest, and CNN
   * Print the predicted labels

## Project Structure

```
├── CMR_subset_1.0/                 # Dataset
│   ├── audio/                      # Audio files
│   ├── annotations/                # Annotations
│   ├── CMRdataset.csv              # Dataset in CSV format
│   └── CMRdataset.xlsx             # Dataset in Excel format
│
├── scripts/
│   ├── dt_amp_mag_classifier.pkl   # Decision Tree model (pickle)
│   ├── dt_amp_mag_classifier.joblib# Decision Tree model (joblib)
│   ├── rf_amp_mag_classifier.joblib# Random Forest model
│   ├── label_encoder.pkl           # Label encoder
│   ├── cnn_model.pt                # Trained CNN model
│   ├── plot_spectrogram.py         # Utility script for spectrogram visualization
│   ├── data_processing.ipynb       # Jupyter notebook for preprocessing
│   └── test.py                     # Main script for predictions
│
├── test.py                         # Top-level test script
└── README.md                       # Project documentation
```

## Notes

* Audio files are padded or truncated to ensure uniform feature length.
* CNN requires reshaped input of the form `(1, 1, 40, 120)`.
* Ensure the pretrained models are placed inside the `scripts/` directory before running the prediction scripts.
