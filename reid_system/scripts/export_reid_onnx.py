import torch
import torchreid
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def export_model_to_onnx():
    """
    Downloads a pre-trained OSNet model and converts it to the universal ONNX format.
    """
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/osnet_x1_0.onnx"

    if os.path.exists(onnx_path):
        logging.info(f"{onnx_path} already exists. Skipping export.")
        return

    logging.info("Loading pre-trained OSNet model from torchreid...")
    
    # Load OSNet architecture pre-trained on Market1501 (a massive dataset of pedestrians)
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=1000,
        loss='softmax',
        pretrained=True
    )
    
    # Set model to evaluation mode (turns off training-specific layers like dropout)
    model.eval()

    # Create a dummy input tensor matching the size ReID models expect: 
    # (Batch Size=1, Channels=3, Height=256, Width=128)
    dummy_input = torch.randn(1, 3, 256, 128)

    logging.info("Exporting model to ONNX format...")
    
    # Export to ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    logging.info(f"Success! Model exported to {onnx_path}")

if __name__ == "__main__":
    export_model_to_onnx()