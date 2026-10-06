"""Check native preprocessing, ONNX execution and decoded boxes independently."""
from collections import Counter
from pathlib import Path
import json
import platform
import sys
import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

ROOT = Path(__file__).resolve().parents[1]

def iou(a, b):
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    intersection = np.prod(np.maximum(hi - lo, 0))
    area_a = np.prod(np.maximum(a[2:] - a[:2], 0))
    area_b = np.prod(np.maximum(b[2:] - b[:2], 0))
    return float(intersection / max(area_a + area_b - intersection, 1e-9))

def match_detections(reference, actual, min_iou=0.98, max_score_difference=0.01):
    if len(reference) != len(actual):
        raise AssertionError(f"Detection count mismatch: {len(reference)} != {len(actual)}")
    remaining = set(range(len(actual)))
    overlaps, score_differences, box_differences = [], [], []
    for item in sorted(reference, key=lambda x: -x['confidence']):
        candidates = [j for j in remaining if actual[j]['class_id'] == item['class_id']]
        if not candidates:
            raise AssertionError(f"Missing class {item['class_id']}")
        j = max(candidates, key=lambda k: iou(np.asarray(item['box']), np.asarray(actual[k]['box'])))
        other = actual[j]
        overlap = iou(np.asarray(item['box']), np.asarray(other['box']))
        score_delta = abs(item['confidence'] - other['confidence'])
        if overlap < min_iou or score_delta > max_score_difference:
            raise AssertionError(f"Detection differs: IoU={overlap}, score_delta={score_delta}")
        overlaps.append(overlap)
        score_differences.append(score_delta)
        box_differences.append(float(np.max(np.abs(np.asarray(item['box']) - np.asarray(other['box'])))))
        remaining.remove(j)
    if not overlaps:
        raise AssertionError("Reference image produced no detections")
    return {"count": len(actual), "min_iou": min(overlaps),
            "max_score_difference": max(score_differences), "max_box_difference_pixels": max(box_differences)}

def inverse_boxes(boxes, result):
    boxes = np.asarray(boxes, dtype=np.float64).copy()
    transform = result['letterbox']
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - transform['left']) / transform['scale']
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - transform['top']) / transform['scale']
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, result['image_size'][0])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, result['image_size'][1])
    return boxes

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(4)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(ROOT/'models/yolo26n.onnx'), sess_options=session_options,
                                   providers=['CPUExecutionProvider'])
    reference_model = YOLO(ROOT/'models/yolo26n.pt')
    report = {"status": "passed", "platform": platform.platform(), "python": sys.version,
              "torch_version": torch.__version__, "python_ort_version": ort.__version__, "images": {},
              "scope": "Two official still images; numeric/output consistency and in-memory repeated-image timing. No tracking, webcam, TensorRT or Jetson validation."}
    for image_name, cpu_directory, gpu_directory in [('bus','cpu','dml'),('zidane','cpu-zidane','dml-zidane')]:
        cpu_root, gpu_root = ROOT/'output'/cpu_directory, ROOT/'output'/gpu_directory
        cpu = json.loads((cpu_root/'result.json').read_text(encoding='utf-8'))
        gpu = json.loads((gpu_root/'result.json').read_text(encoding='utf-8'))
        tensor = np.fromfile(cpu_root/'input.f32', dtype=np.float32).reshape(cpu['input_shape'])
        raw = np.fromfile(cpu_root/'output.f32', dtype=np.float32).reshape(cpu['output_shape'])
        gpu_tensor = np.fromfile(gpu_root/'input.f32', dtype=np.float32).reshape(gpu['input_shape'])
        if not np.array_equal(tensor, gpu_tensor):
            raise AssertionError('CPU/GPU preprocessing differs')
        python_raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        if not np.allclose(raw, python_raw, atol=1e-4, rtol=1e-4):
            raise AssertionError(f"C++ / Python ONNX output mismatch: {np.max(np.abs(raw-python_raw))}")
        bgr = cv2.imdecode(np.fromfile(ROOT/f'assets/{image_name}.jpg', dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise AssertionError('OpenCV image decoding failed')
        reference_image = LetterBox(new_shape=(640,640), auto=False)(image=bgr)
        reference_tensor = np.ascontiguousarray(reference_image[:,:,::-1].transpose(2,0,1)[None]).astype(np.float32)/255
        input_difference = np.abs(reference_tensor - tensor)
        # stb vs libjpeg and byte rounding can differ slightly. This also catches channel/order/layout errors.
        if float(np.mean(input_difference)) > 0.005 or float(np.quantile(input_difference,0.999)) > 0.03:
            raise AssertionError('Native preprocessing differs materially from Ultralytics/OpenCV')
        predictions = reference_model.predict(source=torch.from_numpy(tensor), nms=False,
                                              conf=cpu['threshold'], imgsz=640, device='cpu', verbose=False)[0]
        boxes = inverse_boxes(predictions.boxes.xyxy.cpu().numpy(), cpu)
        torch_detections = [{'class_id': int(cls), 'confidence': float(score), 'box': box.tolist()}
                            for box, score, cls in zip(boxes, predictions.boxes.conf.cpu().numpy(), predictions.boxes.cls.cpu().numpy())]
        pytorch_comparison = match_detections(torch_detections, cpu['detections'])
        gpu_comparison = match_detections(cpu['detections'], gpu['detections'])
        raw_rows = python_raw[0]
        selected = raw_rows[raw_rows[:,4] >= cpu['threshold']]
        decoded = inverse_boxes(selected[:,:4], cpu)
        decoded_detections = [{'class_id': int(row[5]), 'confidence': float(row[4]), 'box': box.tolist()}
                              for row,box in zip(selected,decoded) if box[2]>box[0] and box[3]>box[1]]
        postprocess_comparison = match_detections(decoded_detections, cpu['detections'], min_iou=0.99999, max_score_difference=1e-5)
        report['images'][image_name] = {
            "cpp_vs_python_ort_max_abs_difference": float(np.max(np.abs(raw-python_raw))),
            "preprocess_mean_abs_difference": float(np.mean(input_difference)),
            "preprocess_p999_abs_difference": float(np.quantile(input_difference,0.999)),
            "pytorch_comparison": pytorch_comparison, "cpu_gpu_comparison": gpu_comparison,
            "coordinate_decode_comparison": postprocess_comparison,
            "classes": dict(Counter(d['label'] for d in cpu['detections'])),
            "cpu_timings_ms": cpu['timings_ms'], "gpu_timings_ms": gpu['timings_ms'],
        }
    profiled = json.loads((ROOT/'output/dml-profile/result.json').read_text(encoding='utf-8'))
    events = json.loads(Path(profiled['profile_path']).read_text(encoding='utf-8'))
    providers = Counter(event.get('args',{}).get('provider') for event in events
                        if event.get('args',{}).get('provider'))
    if not providers.get('DmlExecutionProvider',0):
        raise AssertionError('GPU provider execution not proven by native profiling')
    report['profile_provider_events'] = dict(providers)
    destination = ROOT/'reports'
    destination.mkdir(exist_ok=True)
    (destination/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# C++ YOLO26 初步验证结果', '', '验证通过：两个官方示例图的 C++ CPU / GPU 检测，与 Python ONNX 和 PyTorch 参考结果进行交叉检查。', '',
             '当前设备：Windows 11 / Intel i7-12700KF / NVIDIA RTX 4080。GPU 后端为 ONNX Runtime DirectML。', '',
             '| 图片 | 后端 | 预处理均值 ms | 推理均值 ms | 链路 P95 ms | 检测数 |',
             '|---|---|---:|---:|---:|---:|']
    for image, values in report['images'].items():
        for name,key in [('CPU','cpu_timings_ms'),('GPU / DirectML','gpu_timings_ms')]:
            t=values[key]
            lines.append(f"| {image} | {name} | {t['preprocess']['mean']:.3f} | {t['inference']['mean']:.3f} | {t['pipeline']['p95']:.3f} | {values['pytorch_comparison']['count']} |")
    lines += ['', '计时范围为内存中单张图像的预处理、同步推理与结果解析，排除模型初始化、文件读取、图像解码、画框和保存；这是桌面初验，不代表视频处理帧率或 Jetson 性能。', '',
              '检查包含：RGB / NCHW / Letterbox 对照、C++ 与 Python ONNX 输出、PyTorch 检测结果、原图坐标还原、CPU 与 GPU 检测一致性，以及 GPU provider 实际执行记录。', '',
              '尚未覆盖：DeepSORT、ReID、实时视频、云台、TensorRT 和 Jetson 部署。']
    (destination/'validation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
