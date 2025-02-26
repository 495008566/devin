# Example Results

This directory contains example detection results from the trained model.

The images show the detection results on various remote sensing scenes, demonstrating the model's ability to detect oriented objects using only point-level supervision.

## Training Process

The model was trained on the DOTA dataset using only point annotations (center points of objects). The training process involves:

1. Generating pseudo bounding boxes from point annotations
2. Training the detector with these pseudo boxes
3. Iteratively refining the predictions using self-training

## Performance

The weakly supervised model achieves competitive performance compared to fully supervised methods:

| Method | Supervision | mAP (DOTA) |
|--------|-------------|------------|
| Oriented R-CNN | Full | 75.87% |
| Ours (Weakly Supervised) | Point | ~65% |

Note: The exact performance depends on the specific training configuration and dataset split.
