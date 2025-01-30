import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from src.configs.config import Config
from src.models.vit_model import load_vit_model

class CompetitionTestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.image_paths = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(self.image_paths[idx])

def generate_submission(test_dir, model_path, output_csv="submission.csv"):
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Image Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Dataset and DataLoader
    test_dataset = CompetitionTestDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Load Model
    model = load_vit_model(config.num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    filenames = []
    predictions = []

    # Run Inference
    with torch.no_grad():
        for images, paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            filenames.extend(paths)
            predictions.extend(batch_preds.tolist())

    # Save Predictions to CSV
    submission_df = pd.DataFrame({"path": filenames, "class_idx": predictions})
    submission_df.to_csv(output_csv, index=False)
    print(f"Submission file saved as `{output_csv}`")
