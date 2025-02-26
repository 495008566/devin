# Z128: Weakly Supervised Rotation Object Detection in Remote Sensing Scenes

This repository contains the implementation of a weakly supervised rotation object detection system for remote sensing imagery. The project focuses on detecting oriented objects in aerial and satellite images using limited supervision.

## Features

- Weakly supervised learning approach using point-level annotations
- Support for rotation object detection in remote sensing imagery
- Integration with standard remote sensing datasets (DOTA, HRSC2016, UCAS-AOD)
- Based on OBBDetection framework with modifications for weak supervision

## Installation

```bash
# Clone the repository
git clone https://github.com/username/Z128.git
cd Z128

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Dataset Preparation

### DOTA Dataset

1. Download the DOTA dataset from the [official website](https://captain-whu.github.io/DOTA/dataset.html)
2. Place the dataset in the `data/dota` directory
3. Run the preprocessing script:

```bash
python tools/prepare_dota.py --data-root data/dota --output-dir data/dota/processed
```

## Training

```bash
# Train with point supervision
python tools/train.py configs/weakly_supervised/oriented_rcnn_r50_fpn_ws_1x_dota.py
```

## Evaluation

```bash
# Evaluate the model
python tools/test.py configs/weakly_supervised/oriented_rcnn_r50_fpn_ws_1x_dota.py work_dirs/oriented_rcnn_r50_fpn_ws_1x_dota/latest.pth --eval mAP
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgements

This project is based on the following open-source projects:
- [OBBDetection](https://github.com/jbwang1997/OBBDetection)
- [MMRotate](https://github.com/open-mmlab/mmrotate)
- [DOTA Dataset](https://github.com/CAPTAIN-WHU/DOTA)
