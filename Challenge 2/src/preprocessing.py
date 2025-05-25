import os
import pandas as pd
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader


class SoilDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx]['image_id'])
        image = Image.open(img_name).convert('RGB')
        label = self.labels.iloc[idx]['soil_label']
        if self.transform:
            image = self.transform(image)
        return image, label

class TestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.test_data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.test_data.iloc[idx]['image_id'])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.test_data.iloc[idx]['image_id']

# Define data transformations
def create_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, test_transforms

# Load and split the data
def load_data(base_dir, batch_size=32):
    train_csv = os.path.join(base_dir, "train_labels.csv")
    test_csv = os.path.join(base_dir, "test_ids.csv")
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    train_transforms, test_transforms = create_transforms()

    train_dataset = SoilDataset(train_csv, train_dir, train_transforms)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    test_dataset = TestDataset(test_csv, test_dir, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
