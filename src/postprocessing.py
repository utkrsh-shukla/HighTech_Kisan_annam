import pandas as pd
import numpy as np

class SoilPostprocessor:
    def __init__(self, label_encoder):
        self.label_encoder = label_encoder
        self.label_mapping = {v: k for k, v in enumerate(label_encoder.classes_)}

    def decode_predictions(self, predictions):
        """Convert model predictions to soil type labels"""
        predicted_classes = np.argmax(predictions, axis=1)
        return [self.label_mapping[cls] for cls in predicted_classes]

    def create_submission(self, image_ids, predicted_labels, output_path='submission.csv'):
        """Create submission file with proper formatting"""
        submission_df = pd.DataFrame({
            'image_id': image_ids,
            'soil_type': predicted_labels
        })
        
        # Ensure correct order of classes
        submission_df['soil_type'] = submission_df['soil_type'].astype(
            pd.CategoricalDtype(categories=self.label_encoder.classes_)
        )
        
        submission_df.to_csv(output_path, index=False)
        return submission_df

    def visualize_predictions(self, images, predictions, n=9):
        """Visualize predictions with matplotlib"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        for i in range(min(n, len(images))):
            plt.subplot(3, 3, i+1)
            plt.imshow(images[i])
            plt.title(f"Pred: {predictions[i]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
