# CourtMaster: Rule Every Matchup

A computer-vision based analytics projects that will give live insights from a 1 on 1 match.

## Features

- Player detection in video frames using a YOLO26 model (`yolo26n.pt`).
- Configurable dataset and class labels via `data.yaml`.
- Simple detection script (`detection-only.py`) as a starting point for building full analytics.

This repository is currently an experimental sandbox while core detection and data collection are being built

## Repository structure

```text
CourtMaster-Rule-every-matchup/
├─ data.yaml         # YOLO dataset + class configuration
├─ detection-only.py # Initial player detection script
└─ yolo26n.pt        # YOLOv6 model weights (nano variant)
```

- `data.yaml`  
  - Defines the dataset paths and class names for YOLO.  
  - Update this if you change your data folders or add new classes.

- `detection-only.py`  
  - Minimal script to run YOLO-based player detection on images/videos.  
  - This is where you will plug in tracking, event detection, and analytics later.

- `yolo26n.pt`  
  - Pretrained YOLO26 model weights used by `detection-only.py`.  
  - Replace or fine-tune these weights as you improve your model.
