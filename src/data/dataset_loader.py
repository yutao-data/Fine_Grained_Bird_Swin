from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

class BirdDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.dataset = ImageFolder(root=main_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, label = self.dataset.samples[idx]
        image = Image.open(image_path).convert("RGB")  # Ensure it is a PIL image

        # Apply Transformations Safely
        if self.transform:
            image = self.transform(image)

        return image, label
