from torchvision import transforms
from transformers import ViTImageProcessor

class Config:
    def __init__(self):
        self.train_dir = "datasets/train_images"
        self.val_dir = "datasets/val_images"
        self.test_dir = "datasets/test_images/mistery_cat"
        self.batch_size = 32
        self.num_epochs = 100
        self.num_classes = 20
        self.model_name = "google/vit-large-patch16-224-in21k"
        self.best_model_path = "saved_models/best_vit_model.pth"

        # Initialize image processor
        processor = ViTImageProcessor.from_pretrained(self.model_name)

        # Define Train and Validation Transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),  
            transforms.RandomCrop(224),  
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
        ])

        # Ensure this exists for mapping predictions correctly
        self.class_order = [
            'Groove_billed_Ani', 'Red_winged_Blackbird', 'Rusty_Blackbird',
            'Gray_Catbird', 'Brandt_Cormorant', 'Eastern_Towhee', 'Indigo_Bunting',
            'Brewer_Blackbird', 'Painted_Bunting', 'Bobolink', 'Lazuli_Bunting',
            'Yellow_headed_Blackbird', 'American_Crow', 'Fish_Crow', 'Brown_Creeper',
            'Yellow_billed_Cuckoo', 'Yellow_breasted_Chat', 'Black_billed_Cuckoo',
            'Gray_crowned_Rosy_Finch', 'Bronzed_Cowbird'
        ]