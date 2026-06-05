import torch
import onnx
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sensor.multiscale.model import MultiScaleSensorCNN

CKPT = ROOT / "sensor" / "multiscale" / "checkpoints" / "phase4_multiscale_sgdm_best_25.pt"
OUT  = Path(__file__).parent / "sensor_cnn.onnx"

model = MultiScaleSensorCNN()
model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
model.eval()

dummy = torch.zeros(1, 5, 64, 64)
# Use the legacy TorchScript-based exporter (dynamo=False).
# The newer dynamo exporter produces opset-18 graphs with constructs
# that X-CUBE-AI's ONNX shape-inference preprocessing cannot handle
# ("list index out of range" before quantisation even starts).
# The legacy exporter produces a simpler opset-13 graph.
torch.onnx.export(
    model,
    dummy,
    str(OUT),
    opset_version=13,
    input_names=["scalogram"],
    output_names=["wear_um"],
    dynamo=False,
)

# Merge external weight file back into the .onnx so it is self-contained.
# The dynamo exporter splits large models into .onnx + .onnx.data by default;
# X-CUBE-AI (and most other tools) expect a single file.
m = onnx.load(str(OUT), load_external_data=True)
onnx.save(m, str(OUT), save_as_external_data=False)

# Clean up the now-redundant .data sidecar if it still exists
data_file = Path(str(OUT) + ".data")
if data_file.exists():
    data_file.unlink()

size_kb = OUT.stat().st_size // 1024
print(f"Exported → {OUT}  ({size_kb} KB, self-contained)")