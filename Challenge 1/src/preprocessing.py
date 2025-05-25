import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class SoilPreprocessor:
    def __init__(self, img_size=128):
        self.img_size = img_size
        self.label_encoder = LabelEncoder()
        self.label_mapping = None

    def load_and_prepare_data(self, csv_path, image_folder):
        """Load and prepare dataset with progress tracking"""
        # Load and encode labels
        df = pd.read_csv(csv_path)
        df['label'] = self.label_encoder.fit_transform(df['soil_type'])
        self.label_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
        
        # Load images with error handling
        X, y = [], []
        valid_indices = []
        
        print("Loading and preprocessing images...")
        for idx in tqdm(range(len(df))):
            row = df.iloc[idx]
            img_path = os.path.join(image_folder, row['image_id'])
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                y.append(row['label'])
                valid_indices.append(idx)
            except Exception as e:
                print(f"Skipping {row['image_id']}: {str(e)}")
        
        # Filter dataframe to only valid entries
        df = df.iloc[valid_indices]
        
        # Convert to arrays and normalize
        X = np.array(X) / 255.0
        y = np.array(y)
        
        return X, y, df

    def train_val_split(self, X, y, test_size=0.2, random_state=42):
        """Create stratified train/validation split"""
        return train_test_split(
            X, y, 
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

    def get_label_encoder(self):
        return self.label_encoder

    def get_label_mapping(self):
        return self.label_mapping
