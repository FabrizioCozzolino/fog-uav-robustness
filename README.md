# Fog-UAV-Robustness

Evaluating robustness of U-Net semantic segmentation on UAV aerial imagery (VDD)
under GAN-generated fog trained on Foggy Cityscapes.

> CASA project — follows the README and proposal provided by the tutor.

## Pipeline

```
           Foggy Cityscapes (paired: clean ↔ fog)
                          │
                          ▼
                [2] Train Pix2Pix (clean→fog)
                          │
                          ▼
VDD (clean) ──► [3] Generate VDD_foggy ◄── Pix2Pix
    │                     │
    │                     ▼
[1] Train U-Net     [4] Test U-Net on VDD_foggy (no retrain)
    │                     │
    └─────────────────────┴───► Measure performance drop
                                        │
                                        ▼
                         [5] Retrain U-Net on VDD_foggy (or mix)
                                        │
                                        ▼
                         [6] Final evaluation & report
```

## Project structure

```
fog-uav-robustness/
├── configs/                 # YAML config files
├── data/
│   ├── raw/                 # Original downloads (VDD, Foggy Cityscapes)
│   └── processed/           # Generated VDD_foggy, preprocessed splits
├── src/
│   ├── datasets/            # PyTorch Dataset classes
│   ├── models/              # U-Net, Pix2Pix
│   ├── training/            # Training loops
│   ├── inference/           # VDD_foggy generation
│   ├── evaluation/          # mIoU, F1, FID, LPIPS
│   └── utils/
├── notebooks/               # Exploratory + debugging notebooks
├── scripts/                 # Utility scripts (env check, download, etc.)
├── checkpoints/             # Saved model weights (gitignored)
├── outputs/                 # Predictions, figures, result tables
└── requirements.txt
```

## Setup (local, Intel Arc laptop — CPU-only dev)

1. Install Python 3.11 (https://www.python.org/downloads/) — during install, check
   "Add Python to PATH".
2. Open PowerShell in the project folder and create a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```
3. Install PyTorch CPU build (small download, no CUDA drivers needed):
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
4. Install the rest:
   ```powershell
   pip install -r requirements.txt
   ```
5. Verify:
   ```powershell
   python scripts/check_env.py
   ```

## Setup (Google Colab — for actual training)

Open a new Colab notebook, enable GPU runtime (Runtime → Change runtime type → T4),
then run in a cell:

```python
!git clone https://github.com/<YOUR_USER>/fog-uav-robustness.git
%cd fog-uav-robustness
!pip install -r requirements-colab.txt
!python scripts/check_env.py
```

## Datasets

- **VDD (Varied Drone Dataset)** — 400 high-res UAV images, 7 classes.
  https://github.com/RussRobin/VDD (or HuggingFace mirror)
- **Foggy Cityscapes** — 500 clean + 500 medium fog + 500 high fog (paired).
  https://www.kaggle.com/datasets/yessicatuteja/foggy-cityscapes-image-dataset

## Citing / references

See `docs/references.bib` (TBD).
