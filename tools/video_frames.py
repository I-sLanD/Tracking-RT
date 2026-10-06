"""Offline video IO only. Detection, ReID and tracking run in the native executable."""
from pathlib import Path
import argparse
from collections import defaultdict, deque
import json
import math
import cv2
import numpy as np

def prepare(video, directory, limit):
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "frames.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old["source"] == str(video.resolve()) and old["requested_limit"] == limit and all((directory/f).exists() for f in old["files"]):
            print(f"Reusing {len(old['files'])} frames; fps={old['fps']}")
            return
        raise RuntimeError("Choose a new frames directory for a different input/limit.")
    if list(directory.glob("*.png")):
        raise RuntimeError("Frames directory already contains images; choose a fresh directory.")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError("Video frame rate is unavailable.")
    files = []
    last_shape = None
    while len(files) < limit:
        success, frame = cap.read()
        if not success:
            break
        last_shape = frame.shape
        filename = f"{len(files)+1:06d}.png"
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("PNG encoding failed")
        (directory / filename).write_bytes(encoded.tobytes())
        files.append(filename)
    cap.release()
    if not files:
        raise RuntimeError("No video frames decoded.")
    meta = {"source": str(video.resolve()), "requested_limit": limit, "fps": fps,
            "width": last_shape[1], "height": last_shape[0], "files": files,
            "timestamp_policy": "CFR frame index / source fps; no frame skipping", "frame_count": len(files)}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(files)} frames at {fps} FPS")

def render(directory, output):
    meta = json.loads((directory / "frames.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (output / "tracks.jsonl").read_text(encoding="utf-8").splitlines()]
    video = output / "tracked.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (meta["width"], meta["height"]))
    if not writer.isOpened():
        raise RuntimeError("Cannot open output video writer.")
    trails = defaultdict(lambda: deque(maxlen=30))
    for record in records:
        frame = cv2.imdecode(np.fromfile(directory / record["file"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Frame decoding failed")
        for track in record["tracks"]:
            if track["state"] != "confirmed" or not track["observed"] or not track.get("valid_box", True):
                continue
            tid = track["id"]
            x,y,w,h = track["tlwh"]
            color = ((tid*73+70)%200+55,(tid*131+20)%200+55,(tid*47+100)%200+55)
            p1,p2 = (round(x),round(y)), (round(x+w),round(y+h))
            cv2.rectangle(frame,p1,p2,color,2)
            cv2.putText(frame,f"ID {tid} {track['confidence']:.2f}",(p1[0],max(20,p1[1]-7)),cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2)
            trails[tid].append((round(x+w/2),round(y+h)))
            if len(trails[tid])>1:
                cv2.polylines(frame,[np.asarray(trails[tid],dtype=np.int32)],False,color,2)
        # Bound rendering history too; stale/deleted trajectories are removed.
        active = {t["id"] for t in record["tracks"]}
        for tid in list(trails):
            if tid not in active:
                del trails[tid]
        cv2.putText(frame,f"YOLO26 + DeepSORT | frame {record['frame']} | {meta['fps']:.1f} source FPS",(15,27),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
        writer.write(frame)
        if record is records[-1]:
            ok, encoded = cv2.imencode(".png",frame)
            if not ok: raise RuntimeError("Preview encoding failed")
            (output / "preview.png").write_bytes(encoded.tobytes())
    writer.release()
    check = cv2.VideoCapture(str(video))
    if not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != len(records):
        raise RuntimeError("Rendered video validation failed")
    check.release()
    print(f"Rendered {len(records)} frames: {video}")

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["prepare","render"])
    parser.add_argument("--video",type=Path)
    parser.add_argument("--frames",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--limit",type=int,default=180)
    args=parser.parse_args()
    if args.mode=="prepare":
        if not args.video or args.limit<1: parser.error("--video and positive --limit required")
        prepare(args.video,args.frames,args.limit)
    else:
        if not args.output: parser.error("--output required")
        render(args.frames,args.output)
