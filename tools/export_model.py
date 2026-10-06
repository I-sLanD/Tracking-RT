"""Export the official YOLO26n as a fixed-shape, NMS-free ONNX reference."""
from pathlib import Path
import hashlib
import json
import sys
import onnx
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    weights = ROOT / "models/yolo26n.pt"
    expected = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Reference weights do not match the official asset digest")
    model = YOLO(weights)
    labels = [model.names[i] for i in range(len(model.names))]
    (ROOT / "models/classes.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    arguments = dict(format="onnx", imgsz=640, batch=1, dynamic=False,
                     simplify=False, opset=17, nms=False, device="cpu")
    exported = Path(model.export(**arguments))
    graph = onnx.load(exported)
    onnx.checker.check_model(graph)
    dimensions = lambda value: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
    input_shape = dimensions(graph.graph.input[0])
    output_shape = dimensions(graph.graph.output[0])
    if input_shape != [1, 3, 640, 640] or output_shape != [1, 300, 6]:
        raise RuntimeError(f"Unexpected export layout: {input_shape} -> {output_shape}")
    report = {
        "weights": str(weights), "weights_sha256": expected,
        "onnx": str(exported), "onnx_sha256": hashlib.sha256(exported.read_bytes()).hexdigest(),
        "torch_version": torch.__version__, "ultralytics_version": ultralytics.__version__,
        "onnx_version": onnx.__version__, "arguments": arguments,
        "input_shape": input_shape, "output_shape": output_shape,
        "output_layout": "x1,y1,x2,y2,confidence,class_id; input-image coordinates; no external NMS",
    }
    (ROOT / "models/export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
