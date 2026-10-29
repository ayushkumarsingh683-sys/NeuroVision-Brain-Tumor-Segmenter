import torch
import segmentation_models_pytorch as smp

def get_vit_segmenter_model():
    """
    Initializes a Vision Transformer based segmentation model (SegFormer with MiT backbone)
    for high-precision medical image segmentation.
    """
    model = smp.Segformer(
        encoder_name="mit_b0",        # Mix Transformer (ViT-based encoder backbone)
        encoder_weights="imagenet",   # Pre-trained weights for fast convergence
        in_channels=3,                # RGB medical scans
        classes=1,                    # Binary output mask (tumor / non-tumor)
    )
    return model

if __name__ == "__main__":
    model = get_vit_segmenter_model()
    print("Vision Transformer (SegFormer) Model Loaded Successfully!")
    
    # Test forward pass with dummy tensor
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print(f"ViT Output Shape: {output.shape}")  # Expected: torch.Size([2, 1, 256, 256])