"""Download official reference assets and project-local native dependencies."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = [
    ("models/yolo26n.pt", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt", "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"),
    ("assets/bus.jpg", "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg", None),
    ("assets/zidane.jpg", "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg", None),
    ("third_party/downloads/llvm-mingw.zip", "https://github.com/mstorsjo/llvm-mingw/releases/download/20260922/llvm-mingw-20260922-ucrt-x86_64.zip", None),
    ("third_party/downloads/onnxruntime-directml.nupkg", "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/1.24.4/microsoft.ml.onnxruntime.directml.1.24.4.nupkg", None),
    ("third_party/include/stb_image.h", "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h", None),
    ("third_party/include/stb_image_write.h", "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h", None),
]

def fetch(item):
    relative, url, expected = item
    dest = ROOT / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        temporary = dest.with_suffix(dest.suffix + ".part")
        with urlopen(url, timeout=120) as source, temporary.open("wb") as target:
            expected_size = source.headers.get("Content-Length")
            shutil.copyfileobj(source, target)
        if expected_size and temporary.stat().st_size != int(expected_size):
            raise RuntimeError(f"Incomplete download: {relative}")
        if relative.endswith((".zip", ".nupkg")) and not zipfile.is_zipfile(temporary):
            raise RuntimeError(f"Invalid archive: {relative}")
        temporary.replace(dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if expected and digest != expected:
        raise RuntimeError(f"SHA256 mismatch: {relative}")
    print(f"Downloaded {relative}: {dest.stat().st_size} bytes", flush=True)
    return {"path": relative, "url": url, "sha256": digest, "size": dest.stat().st_size}

def extract(relative, target, sentinel):
    if (ROOT / target / sentinel).exists():
        return
    with zipfile.ZipFile(ROOT / relative) as archive:
        archive.extractall(ROOT / target)

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, ASSETS))
    extract("third_party/downloads/llvm-mingw.zip", "third_party", "llvm-mingw-20260922-ucrt-x86_64/bin/x86_64-w64-mingw32-clang++.exe")
    extract("third_party/downloads/onnxruntime-directml.nupkg", "third_party/onnxruntime-directml", "build/native/include/onnxruntime_cxx_api.h")
    # Windows 11 on the current device supplies DirectML 1.15.5 in System32.
    # The native runtime loads it through the normal Windows DLL search rules.
    (ROOT / "models" / "sources.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("Native dependencies extracted.", flush=True)
