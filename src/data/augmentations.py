import torchvision.transforms as T

# Fix Train Augmentations
def get_train_augmentations():
    return T.Compose([
        T.Resize((256, 256)),  
        T.RandomCrop((224, 224)),  
        T.RandomHorizontalFlip(p=0.5),  
        T.RandomRotation(degrees=15),  
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        T.ToTensor(),  # Ensure Tensor Conversion
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  
    ])

# Fix Validation Transformations
def get_val_augmentations():
    return T.Compose([
        T.Resize((224, 224)),  
        T.ToTensor(),  
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  
    ])
