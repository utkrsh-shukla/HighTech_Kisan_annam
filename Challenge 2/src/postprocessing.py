import pandas as pd
import torch

def generate_submission(model, test_loader, device, submission_file="submission.csv"):
    """
    Generates a submission file in the format required for the Kaggle competition.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (str): Device to run the model on ('cuda' or 'cpu').
        submission_file (str): Name of the output submission CSV file.
    """
    model.eval()
    predictions = []
    image_ids = []
    with torch.no_grad():
        for images, img_ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            image_ids.extend(img_ids)

    # Remove .jpg from image_ids
    image_ids = [img_id.replace('.jpg', '') for img_id in image_ids]

    # Create Submission File
    submission = pd.DataFrame({
        "image_id": image_ids,
        "soil_label": predictions
    })
    submission.to_csv(submission_file, index=False)
    print(f"\nSubmission file created: {submission_file}")
    print(submission.head())
