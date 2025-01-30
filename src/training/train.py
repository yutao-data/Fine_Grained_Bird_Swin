import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from src.utils.utils import train_epoch, validate
from src.configs.config import Config
from src.data.dataset_loader import BirdDataset
from src.models.vit_model import load_vit_model
from src.data.augmentations import get_train_augmentations, get_val_augmentations

def train_and_evaluate():
    # Load Configurations
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()  # Enable AMP only if CUDA is available
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Load Dataset with Augmentations (Torchvision Transforms)
    train_dataset = BirdDataset(config.train_dir, transform=get_train_augmentations())  
    val_dataset = BirdDataset(config.val_dir, transform=get_val_augmentations())

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Load Model
    model = load_vit_model(config.num_classes)
    model.to(device)

    # ❄️ Freeze Feature Extraction Layers
    for param in model.vit.parameters():
        param.requires_grad = False

    # Train Only Output Layer
    optimizer = AdamW(model.classifier.parameters(), lr=3e-4, weight_decay=0.01)

    # Training
    print("Starting training (With Torchvision Augmentations)...")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # Save Best Model
    if best_model_state:
        torch.save(best_model_state, config.best_model_path)
        print(f"Best model saved at {config.best_model_path}")

    return model
