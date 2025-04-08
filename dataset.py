import os
import scipy.io
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CarDataset(Dataset):
    def __init__(self, annos, img_dir, transform=None):
        self.annos = annos
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        item = self.annos[idx]
        img_name = item["fname"][0]
        label = int(item["class"][0][0]) - 1  # метки начинаются с 1 → сдвиг на 0
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
