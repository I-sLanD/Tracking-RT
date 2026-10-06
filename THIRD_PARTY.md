# Third-party sources

- **DeepSORT algorithm**: Wojke, Bewley, Paulus, *Simple Online and Realtime Tracking with a Deep Association Metric* (2017), https://arxiv.org/abs/1703.07402. `src/deepsort.*` implements the algorithm in C++ (8D XYAH Kalman model, motion gating, nearest-neighbor cosine gallery, matching cascade, IoU fallback and track lifecycle). ReID is replaced by OSNet; this is not the original paper's appearance network.
- **DeepSORT Python reference**: https://github.com/nwojke/deep_sort, GPL-3.0. Downloaded to `third_party/deep_sort_reference` only for independent validation; its LICENSE is retained there. It is not linked into the C++ executable. `tools/validate_tracking.py` imports it to compare trajectories and numerical state.
- **Torchreid / OSNet**: https://github.com/KaiyangZhou/deep-person-reid, MIT. Original `osnet.py` and LICENSE are retained in `third_party/torchreid`. Author's MSMT17 `combineall=True` OSNet x0.25 checkpoint is linked from https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO. Architecture code is used for export and numerical reference. Checkpoint provenance and hashes are recorded in `models/tracking-sources.json` and `models/reid-export.json`.
- **Eigen 3.4.0**: https://gitlab.com/libeigen/eigen, MPL-2.0 and individual source-file notices; archive includes COPYING files. Used for fixed-size Kalman matrices and Cholesky solves.
- **ONNX Runtime / DirectML package**: https://github.com/microsoft/onnxruntime, MIT and package notices, retained in the NuGet package directory.
- **stb image IO**: https://github.com/nothings/stb, public domain / MIT alternatives described in header notices.
- **Ultralytics YOLO26 reference**: https://github.com/ultralytics/ultralytics and https://github.com/ultralytics/assets, original licenses apply. Model weights, export metadata and source records are retained in `models`.
- **OpenCV sample video**: https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi. Local file `assets/pedestrians.avi`; downloaded source and SHA-256 are recorded in `models/sample-video-source.json`. Used for functional video demonstration, without identity ground truth.
- **LLVM-MinGW**: https://github.com/mstorsjo/llvm-mingw. Toolchain package includes its source component notices.

Downloads and local dependencies are excluded from Git by `.gitignore`; preparation scripts preserve licenses and record hashes. Upstream source files and models retain their original ownership.
