import os
import torch
from datetime import datetime
import pandas as pd
from src.configs.config import Config
from src.data.dataset_loader import BirdDataset
from src.models.vit_model import load_vit_model
from src.training.train import train_and_evaluate
from src.utils.utils import save_model
from src.inference.predict import generate_submission
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor

def main():
    print(f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load Configuration
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train & Evaluate Model
    model = train_and_evaluate()

    # Generate Predictions
    print("Generating predictions on test set...")
    generate_submission(config.test_dir, config.best_model_path, output_csv="submission.csv")

if __name__ == "__main__":
    main()