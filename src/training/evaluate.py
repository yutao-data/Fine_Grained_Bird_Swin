import torch
from src.configs.config import Config
from src.models.vit_model import load_vit_model
from torch.utils.data import DataLoader
from src.data.dataset_loader import BirdDataset

def evaluate():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BirdDataset(config.val_dir, transform=config.val_transform)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    model = load_vit_model(config.num_classes)
    model.load_state_dict(torch.load(config.best_model_path))
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate()