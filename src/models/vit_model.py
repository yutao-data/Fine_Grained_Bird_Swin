from transformers import ViTForImageClassification

def load_vit_model(num_classes=10):
    model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224-in21k", num_labels=num_classes)
    return model