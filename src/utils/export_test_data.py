import argparse
from pathlib import Path
import sys
import numpy as np

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.sensor_scalogram_dataset import MATWISensorScalogramDataset

SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"

def main():
    parser = argparse.ArgumentParser(description="Export test data to .npz format for ONNX evaluation.")
    parser.add_argument("--output_path", type=str, default="data/processed/phase4_test_data.npz", help="Path to save the .npz file")
    
    args = parser.parse_args()
    output_path = Path(ROOT / args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading test dataset...")
    # Load the exact same test dataset used in evaluate.py
    ds = MATWISensorScalogramDataset(
        SCALOGRAM_DIR, FEATURES_PATH, split="test", augment=False
    )
    
    # We can use a DataLoader to batch process it efficiently
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    all_inputs = []
    all_targets = []
    
    print(f"Extracting {len(ds)} samples...")
    for scalograms, targets in loader:
        all_inputs.append(scalograms.numpy())
        all_targets.append(targets.numpy())
        
    X_test = np.concatenate(all_inputs, axis=0)
    Y_test = np.concatenate(all_targets, axis=0)
    
    print(f"Inputs shape: {X_test.shape}")
    print(f"Targets shape: {Y_test.shape}")
    
    print(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path, 
        input=X_test
    )
    print("Done!")

if __name__ == "__main__":
    main()
