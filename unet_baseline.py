import torch
import segmentation_models_pytorch as smp

def get_unet_model():
    """
    Initializes a standard U-Net model with a ResNet34 encoder backbone
    pretrained on ImageNet, configured for binary medical image segmentation.
    """
    model = smp.Unet(
        encoder_name="resnet34",      # Backbone encoder
        encoder_weights="imagenet",   # Pre-trained weights for faster convergence
        in_channels=3,                # Number of input channels (RGB medical images)
        classes=1,                    # Binary output mask (tumor / non-tumor)
    )
    return model

if __name__ == "__main__":
    # Test model instantiation and forward pass
    model = get_unet_model()
    print("U-Net Baseline Model Loaded Successfully!")
    
    # Dummy input tensor to check output shape: [Batch_Size, Channels, Height, Width]
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print(f"Output Shape: {output.shape}")  # Expected: torch.Size([2, 1, 256, 256])