#!/bin/bash

KAGGLE_DATASET="annam-ai/soilclassification"
TARGET_DIR="./soil_classification-2025"

echo "Downloading dataset: $KAGGLE_DATASET"
mkdir -p "$TARGET_DIR"
kaggle datasets download -d "$KAGGLE_DATASET" -p "$TARGET_DIR" --unzip

echo "Download complete. Files saved to $TARGET_DIR"
