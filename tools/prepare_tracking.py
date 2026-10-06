"""Fetch primary-source ReID weights, Eigen, and the upstream test reference."""
from pathlib import Path
import json
import zipfile
from download_dependencies import fetch, ROOT

OSNET_REPO = "https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/master/"
DEEPSORT_REPO = "https://raw.githubusercontent.com/nwojke/deep_sort/master/"
MODEL_URL = "https://drive.usercontent.google.com/download?id=1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e&export=download&confirm=t"

def main():
    records = []
    for item in [
        ("third_party/torchreid/osnet.py", OSNET_REPO + "torchreid/models/osnet.py", None),
        ("third_party/torchreid/LICENSE", OSNET_REPO + "LICENSE", None),
        ("models/osnet_x0_25_msmt17.pth", MODEL_URL, "cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18"),
        ("third_party/downloads/eigen-3.4.0.zip", "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip", None),
    ]:
        records.append(fetch(item))
    for filename in ["__init__.py", "kalman_filter.py", "linear_assignment.py", "nn_matching.py", "iou_matching.py", "track.py", "tracker.py", "detection.py"]:
        records.append(fetch(("third_party/deep_sort_reference/deep_sort/" + filename, DEEPSORT_REPO + "deep_sort/" + filename, None)))
    records.append(fetch(("third_party/deep_sort_reference/LICENSE", DEEPSORT_REPO + "LICENSE", None)))
    if not (ROOT / "third_party/eigen-3.4.0/Eigen/Core").exists():
        with zipfile.ZipFile(ROOT / "third_party/downloads/eigen-3.4.0.zip") as archive:
            archive.extractall(ROOT / "third_party")
    (ROOT / "models/tracking-sources.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    record = fetch(("assets/pedestrians.avi", "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi", "45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf"))
    (ROOT / "models/sample-video-source.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("Tracking dependencies and OpenCV sample video prepared.")

if __name__ == "__main__":
    main()
