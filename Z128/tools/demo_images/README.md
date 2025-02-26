# Demo Images

This directory contains example images for demonstration purposes.

To visualize detection results on these images, use the following command:

```bash
python tools/visualize.py configs/weakly_supervised/oriented_rcnn_r50_fpn_ws_1x_dota.py \
    work_dirs/oriented_rcnn_r50_fpn_ws_1x_dota/latest.pth \
    tools/demo_images/example.jpg \
    --out-file output.jpg
```

Note: You need to train the model first to generate the checkpoint file.
