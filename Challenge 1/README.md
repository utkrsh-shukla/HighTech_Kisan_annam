# HighTech_Kisan_annam

# 🌱 Soil Classification Challenge

This repository contains a Google Colab notebook that demonstrates a complete machine learning workflow for classifying images of soil into different types using a Convolutional Neural Network (CNN).

## Overview

The notebook guides you through the following steps:

1.  **Importing Required Libraries:** Setting up the environment by importing necessary Python libraries for data handling, visualization, image processing, and building the CNN model.
2.  **Loading Dataset:** Loading the training labels from a CSV file.
3.  **Viewing a Sample Image:** Visualizing an example image to understand the input data format.
4.  **Data Preprocessing:**
    *   Encoding categorical soil labels into numerical values.
    *   Resizing and normalizing all images.
    *   Storing the preprocessed images and labels as NumPy arrays.
5.  **Train-Validation Split:** Dividing the dataset into training and validation sets to evaluate the model's performance.
6.  **Building the CNN Model:** Constructing a simple CNN architecture using TensorFlow/Keras with convolutional, pooling, and dense layers.
7.  **Model Training:** Training the CNN model on the training data for a specified number of epochs and monitoring validation performance.
8.  **Training Performance:** Visualizing the training and validation accuracy and loss over the epochs.
9.  **Validation Evaluation:** Evaluating the final performance of the trained model on the validation set.
10. **Test Data Prediction & Submission:**
    *   Loading and preprocessing the test images.
    *   Making predictions on the test data using the trained model.
    *   Creating a submission CSV file in the required format.
11. **Visualizing Test Predictions:** Displaying some test images with their predicted soil types.
12. **Conclusion:** A brief summary of the project's outcome.

## Dataset

The dataset is expected to be located in the `/content/soil_classification-2025/` directory within the Colab environment. It should contain:

*   `train_labels.csv`: A CSV file with image IDs and corresponding soil type labels for the training data.
*   `train/`: A directory containing the training images.
*   `test/`: A directory containing the test images.

## Setup

To run this notebook:

1.  Open the provided `.ipynb` file in Google Colab.
2.  Ensure your dataset is correctly placed in the `/content/soil_classification-2025/` directory or update the file paths in the notebook accordingly.
3.  Run all cells in the notebook sequentially.

## Dependencies

The notebook uses the following libraries:

*   pandas
*   matplotlib
*   opencv-python (`cv2`)
*   os
*   numpy
*   tqdm
*   sklearn
*   tensorflow
*   PIL (Pillow)

These dependencies are standard and should be available in the Colab environment.

## Results

The notebook will output the following:

*   Information about the loaded data.
*   A sample training image with its label.
*   The label mapping used for encoding.
*   The shapes of the preprocessed data.
*   The train and validation set sizes.
*   A summary of the CNN model architecture.
*   Plots showing the training and validation accuracy and loss over epochs.
*   The final validation accuracy.
*   A `submission.csv` file containing the predictions for the test data.
*   Visualizations of some test images with their predicted labels.

## Contributing

If you find any issues or have suggestions for improvement, feel free to open an issue or submit a pull request.
