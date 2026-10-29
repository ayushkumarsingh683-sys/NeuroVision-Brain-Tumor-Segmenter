import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import MedicalDataset
from models.vit_segmenter import get_vit_segmenter_model  # Chahe toh unet_baseline import kar lo

def train_segmentation_model():
    # 1. Hyperparameters & Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 2
    EPOCHS = 5
    LEARNING_RATE = 0.0001
    IMAGE_DIR = "dataset/images"
    MASK_DIR = "dataset/masks"
    
    print(f"Training on Device: {DEVICE}")

    # 2. Initialize Model (Using ViT SegFormer here; change to get_unet_model() for U-Net baseline)
    model = get_vit_segmenter_model().to(DEVICE)

    # 3. Dataset and DataLoader setup
    # Note: Ensure you have added sample images & masks in dataset/images and dataset/masks before running!
    try:
        train_dataset = MedicalDataset(IMAGE_DIR, MASK_DIR)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"Dataset loaded successfully with {len(train_dataset)} samples.")
    except Exception as e:
        print(f"Dataset loading error (Make sure folders are not empty): {e}")
        return

    # 4. Loss Function and Optimizer
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass & optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss / len(train_loader):.4f}")

    print("🎉 Model Training Completed Successfully!")
    
    # Save the trained model weights
    os.makedirs("models/saved_weights", exist_ok=True)
    torch.save(model.state_dict(), "models/saved_weights/vit_segmenter_weights.pth")
    print("Model weights saved to models/saved_weights/vit_segmenter_weights.pth")

if __name__ == "__main__":
    train_segmentation_model()