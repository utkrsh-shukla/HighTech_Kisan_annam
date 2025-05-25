#!/bin/bash

# Create data directory if not exists
TARGET_DIR="./data"
mkdir -p "$TARGET_DIR"

# Download dataset
KAGGLE_COMPETITION="soil-classification-2"
echo "Downloading dataset: $KAGGLE_COMPETITION"
kaggle competitions download -c "$KAGGLE_COMPETITION" -p "$TARGET_DIR"

echo "Download complete. Files saved to $TARGET_DIR"
