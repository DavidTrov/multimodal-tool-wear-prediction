import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.sensor_cnn_model import SensorCNNRegressor

def main():
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to ONNX format.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pt model file")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the .onnx model file")
    
    args = parser.parse_args()
    model_path = Path(args.model_path)
    
    if args.output_path is None:
        output_path = model_path.with_suffix('.onnx')
    else:
        output_path = Path(args.output_path)
        
    print(f"Loading model from {model_path}...")
    model = SensorCNNRegressor()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Dummy input based on scalogram shape: batch_size=1, channels=5, height=64, width=64
    dummy_input = torch.randn(1, 5, 64, 64)
    
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        str(output_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    # Merge external data if PyTorch created a .onnx.data file and scrub unsupported attributes
    import onnx
    import os
    
    data_file = str(output_path) + ".data"
    if os.path.exists(data_file):
        print(f"Merging external data from {data_file} into standalone ONNX file...")
        onnx_model = onnx.load(str(output_path))
    else:
        onnx_model = onnx.load(str(output_path))

    print("Scrubbing 'allowzero' from Reshape nodes for ST Edge AI compatibility...")
    for node in onnx_model.graph.node:
        if node.op_type == "Reshape":
            # Filter out the allowzero attribute as ST Edge AI doesn't support it
            new_attrs = [attr for attr in node.attribute if attr.name != "allowzero"]
            if len(new_attrs) < len(node.attribute):
                del node.attribute[:]
                node.attribute.extend(new_attrs)

    # Save it back as a single standalone file
    onnx.save(onnx_model, str(output_path))
    
    if os.path.exists(data_file):
        os.remove(data_file)

    print("Done!")

if __name__ == "__main__":
    main()
