# 🌱 Soil Image Classification Challenge - Annam.ai, IIT Ropar 🌱

This notebook presents a solution for the Soil Image Classification Challenge. The goal is to classify images based on whether they contain soil (label 1) or not (label 0) using a pre-trained ResNet18 model in PyTorch.

## 🚀 Project Overview

The notebook follows these main steps:

1.  **Import Libraries**: Import necessary libraries for data handling, model building, and visualization.
2.  **Set Up Paths**: Define and verify dataset directories and CSV files.
3.  **Define Data Preprocessing**: Define image transformations for training and testing.
4.  **Create Dataset Classes**: Implement custom PyTorch `Dataset` classes for loading image data and labels.
5.  **Load Data**: Split data into training and validation sets and create data loaders.
6.  **Visualize Sample Images**: Display example images from the training set.
7.  **Define the Model**: Load a pre-trained ResNet18 model and modify the final layer for binary classification.
8.  **Define Loss and Optimizer**: Set up the criterion and optimizer for training.
9.  **Training Loop**: Train the model and track loss.
10. **Plot Loss Curves**: Visualize the training and validation loss over epochs.
11. **Generate Test Predictions**: Make predictions on the test dataset.
12. **Create Submission File**: Generate a `submission.csv` file in the required format.

## 🛠️ Setup and Usage

### Prerequisites

*   Google Colab environment or a local environment with Python and necessary libraries installed.


### Installation

The required libraries are commonly available in Google Colab. If running locally, you can install them using pip:

### Running the Notebook

1.  Upload the notebook file (`.ipynb`) to your Google Drive if using Colab.
2.  Open the notebook in Google Colab.
3.  Mount your Google Drive (if applicable) to access the dataset.
4.  Ensure the dataset is placed in the correct directory structure as defined in the notebook (e.g., `/content/soil_competition-2025-2`).
5.  Run the cells sequentially.

The notebook will perform the following actions:

*   Load and preprocess the data.
*   Train the ResNet18 model.
*   Generate predictions for the test set.
*   Save a `submission.csv` file in the working directory.

## 📂 Dataset Details

*   `train_labels.csv`: Contains `image_id` and `soil_label` for the training data.
*   `test_ids.csv`: Contains `image_id` for the test data.
*   Images are expected to be in the `train` and `test` subdirectories.

## 🧠 Model

The model used is a pre-trained ResNet18 from the `torchvision.models` library. The final fully connected layer is modified to output probabilities for two classes.

## ⚙️ Configuration

*   **Batch Size**: 32
*   **Optimizer**: Adam with a learning rate of 0.001
*   **Loss Function**: CrossEntropyLoss
*   **Epochs**: 5 (can be increased for better performance)

## 📈 Results

The notebook outputs a `submission.csv` file containing the predicted `soil_label` for each `image_id` in the test set. A plot of the training and validation loss curves is also generated and saved as `loss_curves.png`.

## 📝 Notes

*   Transfer learning with a pre-trained ResNet18 is used for efficiency.
*   The number of training epochs is kept small in the notebook for demonstration purposes; increasing this value may improve performance.
*   Ensure that the dataset paths in the notebook match the location of your dataset.
*   The generated `submission.csv` is ready for submission to the challenge platform (e.g., Kaggle).
