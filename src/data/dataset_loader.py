from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class BirdDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        """
        Custom dataset loader using torchvision's ImageFolder.
        
        Args:
        - main_dir (str): Path to dataset (e.g., train or val directory)
        - transform (torchvision.transforms.Compose): Image transformations
        
        Returns:
        - image: Transformed image
        - label: Corresponding class label
        """
        self.dataset = ImageFolder(root=main_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label