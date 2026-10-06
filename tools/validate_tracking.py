"""Independent upstream DeepSORT, PyTorch ReID and native-provider cross-checks."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
import cv2
import numpy as np
import onnxruntime as ort
import torch
from export_reid import load_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/deep_sort_reference"))
from deep_sort.detection import Detection
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.tracker import Tracker

class InstrumentedReference(Tracker):
    """Attach observation indices without modifying the upstream matching algorithm."""
    def update(self, detections):
        for track in self.tracks:
            track.reference_detection_index = -1
        super().update(detections)

    def _match(self, detections):
        result = super()._match(detections)
        for track_index, detection_index in result[0]:
            self.tracks[track_index].reference_detection_index = detection_index
        return result

    def _initiate_track(self, detection):
        super()._initiate_track(detection)
        self.tracks[-1].reference_detection_index = detection.reference_index

def reference(frames, config=None):
    config = config or {"max_cosine": .2, "max_iou": .7, "max_age": 30, "n_init": 3, "nn_budget": 100}
    tracker = InstrumentedReference(NearestNeighborDistanceMetric("cosine", config["max_cosine"], config["nn_budget"]),
                      max_iou_distance=config["max_iou"], max_age=config["max_age"], n_init=config["n_init"])
    result=[]
    for detections in frames:
        tracker.predict()
        tagged=[]
        for index,(box,score,feature) in enumerate(detections):
            detection=Detection(box,score,feature)
            detection.reference_index=index
            tagged.append(detection)
        tracker.update(tagged)
        result.append({t.track_id: {"mean": t.mean.copy(), "covariance": t.covariance.copy(),
                                   "box": t.to_tlwh().copy(), "hits": t.hits, "age": t.age,
                                   "time_since_update": t.time_since_update, "detection_index": t.reference_detection_index,
                                   "state": "confirmed" if t.is_confirmed() else "tentative"}
                       for t in tracker.tracks})
    return result

def synthetic():
    frames=[]
    for i in range(115):
        detections=[]
        if i < 65 or i >= 105:
            if i not in range(15,19) and i not in range(42,49):
                detections.append(([100+i*.8,50,40,100],.9,[1,0,0,0]))
            detections.append(([120-i*.8,50,40,100],.85,[0,1,0,0]))
        if i==58:
            detections.append(([500,50,40,100],.3,[0,0,1,0]))
        if i<7:
            # A shrinking small target, then a long gap, exercises unconstrained linear size prediction.
            detections.append(([300,90,5,10-i],.8,[0,0,0,1]))
        if i%2:
            detections.reverse()
        frames.append(detections)
    expected=reference(frames)
    # Ground truth for this controlled fixture only; it is not a MOT dataset score.
    for i in range(3,40):
        for box,_,feature in frames[i]:
            if not (feature[0] or feature[1]):
                continue
            tid=1 if feature[0] else 2
            assert expected[i][tid]["time_since_update"]==0
            assert np.linalg.norm(expected[i][tid]["box"][:2]-np.asarray(box[:2]))<2
    input_path=ROOT/"output/tracking-reference-input.txt"
    output_path=ROOT/"output/tracking-reference-output.csv"
    with input_path.open("w",encoding="ascii") as f:
        f.write(f"{len(frames)} 4\n")
        for detections in frames:
            f.write(f"{len(detections)}\n")
            for box,score,feature in detections:
                f.write(" ".join(map(str,[*box,score,*feature]))+"\n")
    subprocess.run([str(ROOT/"build/tracker_tests.exe"),"--replay",str(input_path.relative_to(ROOT)),str(output_path.relative_to(ROOT))],cwd=ROOT,check=True)
    native=[{} for _ in frames]
    for row in csv.DictReader(output_path.open(encoding="ascii")):
        native[int(row["frame"])-1][int(row["id"])]=row
    mean_error=cov_error=0
    compared=0
    for i,(wanted,actual) in enumerate(zip(expected,native)):
        assert set(wanted)==set(actual),(i,set(wanted),set(actual))
        for tid,target in wanted.items():
            got=actual[tid]
            assert int(got["state"])==(1 if target["state"]=="confirmed" else 0)
            for name in ("hits","age","time_since_update"):
                assert int(got[name])==target[name]
            mean=np.array([float(got[f"mean{j}"]) for j in range(8)])
            covariance=np.array([float(got[f"cov{j}"]) for j in range(64)]).reshape(8,8)
            np.testing.assert_allclose(mean,target["mean"],rtol=1e-10,atol=1e-9)
            np.testing.assert_allclose(covariance,target["covariance"],rtol=1e-10,atol=1e-9)
            mean_error=max(mean_error,float(np.max(np.abs(mean-target["mean"]))))
            cov_error=max(cov_error,float(np.max(np.abs(covariance-target["covariance"]))))
            compared+=1
    return {"frames":len(frames),"track_states_compared":compared,"ids_and_lifecycle_equal":True,
            "mean_max_abs":mean_error,"covariance_max_abs":cov_error,
            "scenarios":["crossing","reversed detection order","4/7-frame misses","tentative false detection","max_age expiry","re-entry","shrinking lost-track size prediction"]}

def read_run(provider):
    directory=ROOT/f"output/tracking-{provider}"
    records=[json.loads(line) for line in (directory/"tracks.jsonl").read_text(encoding="utf-8").splitlines()]
    features=np.fromfile(directory/"features.f32",dtype=np.float32)
    frames=[]
    offset=0
    for record in records:
        detections=[]
        for d in record["detections"]:
            assert d["feature_offset"]==offset
            feature=features[offset:offset+512]
            assert len(feature)==512 and np.isfinite(feature).all()
            np.testing.assert_allclose(np.linalg.norm(feature),1,atol=1e-6)
            detections.append((d["tlwh"],d["confidence"],feature))
            offset+=512
        frames.append(detections)
    assert offset==len(features)
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    expected=reference(frames,summary["config"])
    max_box_error=0
    id_mapping={}
    for i,(wanted,record) in enumerate(zip(expected,records)):
        # The upstream solver returns unmatched detections in a different order.
        # Bind each new ID once using its birth observation; never remap existing IDs.
        for got in record["tracks"]:
            if got["age"]==1:
                candidate=[tid for tid,t in wanted.items() if t["age"]==1 and t["detection_index"]==got["detection_index"]]
                assert len(candidate)==1 and got["id"] not in id_mapping,(provider,i,got)
                assert candidate[0] not in id_mapping.values()
                id_mapping[got["id"]]=candidate[0]
        actual={id_mapping[t["id"]]:t for t in record["tracks"]}
        assert set(wanted)==set(actual),(provider,i,set(wanted),set(actual))
        for tid,target in wanted.items():
            got=actual[tid]
            assert got["state"]==target["state"]
            for name in ("hits","age","time_since_update","detection_index"):
                assert got[name]==target[name]
            assert got["observed"]==(target["time_since_update"]==0)
            assert got["gallery_size"]<=summary["config"]["nn_budget"]
            np.testing.assert_allclose(got["tlwh"],target["box"],rtol=1e-8,atol=1e-6)
            max_box_error=max(max_box_error,float(np.max(np.abs(np.asarray(got["tlwh"])-target["box"]))))
    return records,frames,summary,{"frames":len(frames),"all_associations_lifecycles_equal_to_upstream":True,
                                 "id_mapping_policy":"bind at birth detection once; never remap an existing ID",
                                 "literal_id_numbers_equal":all(a==b for a,b in id_mapping.items()),
                                 "box_max_abs_px":max_box_error}

def reid_validation(provider,model):
    directory=ROOT/f"output/tracking-{provider}"
    data=np.fromfile(directory/"reid-input.f32",dtype=np.float32).reshape(1,3,256,128)
    actual=np.fromfile(directory/"reid-output.f32",dtype=np.float32).reshape(1,512)
    with torch.no_grad():
        expected=model(torch.from_numpy(data)).numpy()
    session=ort.InferenceSession(str(ROOT/"models/osnet_x0_25_msmt17.onnx"),providers=["CPUExecutionProvider"])
    onnx=session.run(None,{"images":data})[0]
    np.testing.assert_allclose(actual,onnx,rtol=3e-4,atol=2e-4)
    cosine=float(np.dot(actual.flatten(),expected.flatten())/(np.linalg.norm(actual)*np.linalg.norm(expected)))
    assert cosine>0.99999
    probe=json.loads((directory/"reid-probe.json").read_text())
    image=cv2.imdecode(np.fromfile(ROOT/"assets/pedestrian-frames"/probe["file"],dtype=np.uint8),cv2.IMREAD_COLOR)
    x1,y1,x2,y2=probe["crop"]
    rgb=cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2RGB)
    resized=cv2.resize(rgb,(128,256),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255
    transformed=(resized-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
    crop_error=float(np.max(np.abs(data[0].transpose(1,2,0)-transformed)))
    # C++ and OpenCV byte resize differ by at most one uint8 rounding step.
    assert crop_error<.018
    return {"pytorch_embedding_cosine":cosine,"python_ort_max_abs":float(np.max(np.abs(actual-onnx))),
            "opencv_crop_max_abs_normalized":crop_error}

def profile_check(directory):
    summary=json.loads((directory/"summary.json").read_text())
    providers={}
    for name in ("detector","reid"):
        events=json.loads(Path(summary[f"{name}_profile"]).read_text())
        counts={}
        for event in events:
            provider=event.get("args",{}).get("provider")
            if provider:
                counts[provider]=counts.get(provider,0)+1
        assert counts.get("DmlExecutionProvider",0)>0,(name,counts)
        providers[name]=counts
    return providers

def main():
    torch.set_num_threads(4)
    result={"status":"passed","synthetic":synthetic()}
    model=load_model()
    runs={}
    for provider in ("cpu","dml"):
        records,frames,summary,validation=read_run(provider)
        validation["reid"]=reid_validation(provider,model)
        runs[provider]=(records,frames,summary)
        result[provider]=validation
    cpu_records,cpu_frames,cpu_summary=runs["cpu"]
    dml_records,dml_frames,dml_summary=runs["dml"]
    assert len(cpu_records)==len(dml_records)
    box_error=score_error=0
    min_cosine=1
    for cf,gf,cr,gr in zip(cpu_frames,dml_frames,cpu_records,dml_records):
        assert len(cf)==len(gf)
        for c,g in zip(cf,gf):
            box_error=max(box_error,float(np.max(np.abs(np.array(c[0])-g[0]))))
            score_error=max(score_error,abs(c[1]-g[1]))
            min_cosine=min(min_cosine,float(np.dot(c[2],g[2])))
        assert [(t["id"],t["state"],t["detection_index"]) for t in cr["tracks"]]==[(t["id"],t["state"],t["detection_index"]) for t in gr["tracks"]]
    assert box_error<.01 and score_error<1e-4 and min_cosine>.99999
    result["cpu_dml"]={"box_max_abs_px":box_error,"score_max_abs":score_error,"feature_min_cosine":min_cosine,"all_track_assignments_equal":True}
    result["gpu_profiles"]=profile_check(ROOT/"output/tracking-dml-profile")
    result["timings"]={p:runs[p][2]["timings_ms"] for p in runs}
    result["scope"]="Desktop Windows CPU/DirectML initial baseline. Public sample has no ground truth: no IDF1/HOTA/MOTA or Jetson performance claim. Offline Python video IO, C++ detection/ReID/tracking. Unprofiled runs used for timing."
    result["provenance"]={
        "models":{name:hashlib.sha256((ROOT/"models"/name).read_bytes()).hexdigest()
                  for name in ("yolo26n.onnx","osnet_x0_25_msmt17.onnx")},
        "sample_sha256":hashlib.sha256((ROOT/"assets/pedestrians.avi").read_bytes()).hexdigest(),
        "packages":{name:metadata.version(name) for name in ("torch","onnxruntime","numpy","scipy","opencv-python")},
    }
    (ROOT/"reports/tracking-validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    lines=["# DeepSORT C++ 初步验证", "", "验证通过：原生 YOLO26 → OSNet x0.25 → DeepSORT。", "",
           f"- 公共视频：OpenCV vtest.avi，前 {cpu_summary['frames']} 帧，{cpu_summary['fps']:g} FPS，连续帧不跳帧。",
           f"- C++ 算法测试通过；{result['synthetic']['frames']} 帧受控序列与作者 Python 实现逐帧对照，ID、状态和生命周期一致。",
           f"- Kalman mean 最大误差 {result['synthetic']['mean_max_abs']:.3g}，covariance 最大误差 {result['synthetic']['covariance_max_abs']:.3g}。",
           "- 公共视频的 CPU 和 GPU 缓存分别与作者 DeepSORT 对照，轨迹关联、状态、生命周期一致。新建轨迹编号排序可能不同，验证仅在出生帧绑定编号对应关系，之后禁止重映射。",
           f"- CPU/GPU 检测框最大差 {box_error:.6g} px，ReID 特征最小余弦相似度 {min_cosine:.9f}，轨迹关联一致。",
           "- 两个模型的 ORT profile 均确认 DirectML 节点执行；可能包含少量 CPU 节点，事件数量不表示耗时比例。", "",
           "| 阶段（ms） | CPU 均值 | DirectML 均值 | DirectML P95 |", "|---|---:|---:|---:|"]
    for stage,label in [("detector_pre","检测预处理"),("detector_infer","YOLO26 推理"),("detector_post","检测解析"),("reid_pre","ReID 裁剪/归一化"),("reid_infer","ReID 推理"),("association","DeepSORT 关联"),("pipeline","感知流水线"),("png_decode","额外 PNG 解码")]:
        c=cpu_summary["timings_ms"][stage];g=dml_summary["timings_ms"][stage]
        lines.append(f"| {label} | {c['mean']:.3f} | {g['mean']:.3f} | {g['p95']:.3f} |")
    lines.extend(["",f"每帧平均行人数 {cpu_summary['detections']/cpu_summary['frames']:.2f}，峰值 {cpu_summary['peak_people']}；ReID 使用固定 batch=1，逐人推理。",
                  "计时排除模型初始化、暖机、PNG 解码、结果日志、视频解码和渲染；不是摄像头端到端 FPS。", "",
                  "## 验证边界", "", "受控交叉/漏检验证只证明实现行为。公开视频无真实 ID 标注，不能据此声称 IDF1、HOTA、MOTA 改善或复杂遮挡效果。",
                  "当前仅跟踪 COCO person；OSNet 是行人 ReID 模型。固定 dt=1，max_age 按帧计；尚未实现真实时间间隔预测、相机运动补偿、实时视频采集或 Jetson TensorRT。", "",
                  "## 后续深化", "", "1. 在带真实 ID 标注的固定序列上建立 IDF1 / HOTA / ID switch 基线，并检查错误片段。",
                  "2. 在固定检测缓存上对比余弦门限、特征入库质量和图库大小，保证检测器一致。",
                  "3. 对比逐人 ReID 与批量/按轨迹状态调度；统一记录每帧人数、P95 延迟与身份指标。",
                  "4. 迁移 Jetson 后测 TensorRT、GPU 预处理、解码/推理流水并行，以及包含 IO 的端到端延迟。"])
    (ROOT/"reports/tracking-validation.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Tracking validation PASS")
    print(json.dumps({k:v for k,v in result.items() if k not in ("timings",)},indent=2))

if __name__=="__main__":
    main()
