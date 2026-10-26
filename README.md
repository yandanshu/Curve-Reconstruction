# Curve Reconstruction

Two-view curve and line reconstruction using CSD detection, TEED edges, RoMa matching, and NURBS fitting.

## Build

Requires CMake 3.16+, a C++17 compiler, OpenCV with contrib modules, Eigen3, PCL, Ceres, yaml-cpp, pybind11, and Python development headers.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

Use `-DPython3_EXECUTABLE=path/to/your/python` to select the Python environment used by the embedded interpreter. The inference wrappers use NumPy, OpenCV (`cv2`), Pillow (`PIL`), and PyTorch. Make these available in that Python environment, then install the bundled RoMa package, which declares its own dependencies:

```sh
python -m pip install -e 3rdparties/RoMa
```

## Input

Each case contains exactly two images and COLMAP text calibration records:

```text
image_pairs/
  case_name/
    cameras.txt
    images.txt
    image_1.jpg
    image_2.jpg
```

`cameras.txt` uses `PINHOLE` records: `CAMERA_ID MODEL WIDTH HEIGHT FX FY CX CY`. `images.txt` uses `IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`, followed by a points line for each image. Input poses are world-to-camera transforms. Image names must match the local files.

The default image scale is 0.2, matching the prepared input pairs. Supply images at that scale relative to the dimensions in `cameras.txt`; the program scales intrinsics and remaps images, without a separate image resize step.

## Run

Copy `configs/recon.yaml` to `configs/recon.local.yaml`. Set `dataset.root` to your input directory and optionally set `run.case` to a case directory name. An empty case selects all pairs. Paths in YAML are relative to the YAML file directory.

Weights and datasets are external and are not included in this repository. Set the weight locations relative to the directory from which you launch the program:

```sh
export TEED_CHECKPOINT=path/to/your/teed_model.pth
export ROMA_WEIGHTS=path/to/your/roma_outdoor.pth
export DINOV2_WEIGHTS=path/to/your/dinov2_vitl14_pretrain.pth
./build/main --config configs/recon.local.yaml
```

Use the TEED checkpoint matching the bundled `TED` architecture. If the RoMa weight variables are omitted, the upstream loader uses its download/cache mechanism. The program writes PLY files and `runtimes.txt` to `run.output_root`.

`include/config.hpp` holds compile-time detection and visualization settings. Reconstruction parameters and optional pose perturbations are in YAML; zero perturbations use the supplied poses.

## Evaluation

`evaluation/evaluation.ipynb` is an optional analysis tool for reconstruction PLY files, metric plots, and tables. Run the notebook from `evaluation/` and configure these variables relative to that directory:

```sh
export DATASET_ROOT=path/to/your/image_pairs
export MULTIVIEW_EVAL_BIN=path/to/your/multiview_evaluator
```

The external evaluator must accept `--reconstruction_ply_path`, `--ground_truth_mlp_path`, and `--tolerances`, and emit `Tolerances`, `Completenesses`, `Accuracies`, and `F1-scores` lines. Ground-truth scan alignment files belong at `<case>/scan_clean/scan_alignment.mlp`. The evaluator and scans are not bundled.

Third-party license notices are retained with their source. See `THIRD_PARTY_NOTICES.md`.
