"""Export the author's MSMT17-trained OSNet x0.25, not ImageNet-only weights."""
from pathlib import Path
import importlib.util
import hashlib
import json
import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]

def load_model():
    source = ROOT / "third_party/torchreid/osnet.py"
    spec = importlib.util.spec_from_file_location("official_osnet", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weights = torch.load(ROOT / "models/osnet_x0_25_msmt17.pth", map_location="cpu", weights_only=True)
    state = weights.get("state_dict", weights)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model = module.osnet_x0_25(num_classes=state["classifier.weight"].shape[0], pretrained=False)
    model.load_state_dict(state, strict=True)
    return model.eval()

def main():
    torch.set_num_threads(4)
    model = load_model()
    target = ROOT / "models/osnet_x0_25_msmt17.onnx"
    torch.manual_seed(7)
    example = torch.randn(1, 3, 256, 128)
    torch.onnx.export(model, example, target, input_names=["images"], output_names=["embeddings"], opset_version=17,
                      do_constant_folding=True, dynamic_axes=None)
    exported = onnx.load(target)
    onnx.checker.check_model(exported)
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        reference = model(example).numpy()
    actual = session.run(None, {"images": example.numpy()})[0]
    np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=2e-5)
    metadata = {
        "architecture": "osnet_x0_25", "training_dataset": "MSMT17 (combineall=True)",
        "model_zoo": "https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO",
        "input_shape": [1, 3, 256, 128], "output_shape": [1, 512], "dtype": "float32", "opset": 17,
        "preprocessing": "clipped integer xyxy crop; RGB bilinear resize 128x256; /255; ImageNet mean/std; NCHW",
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "postprocessing": "L2 normalize 512-dimensional embedding in C++",
        "batch": "static 1; sequential per-person baseline", "pytorch_onnx_max_abs": float(np.max(np.abs(actual-reference))),
        "weights_sha256": hashlib.sha256((ROOT / "models/osnet_x0_25_msmt17.pth").read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (ROOT / "models/reid-export.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
