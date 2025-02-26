#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from ..builder import PIPELINES

@PIPELINES.register_module()
class LoadPointAnnotations(object):
    """Load point annotations for weakly supervised detection."""
    
    def __init__(self, with_bbox=True, with_label=True, with_point=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_point = with_point
        
    def _load_points(self, results):
        """Load point annotations."""
        ann_info = results['ann_info']
        results['gt_points'] = ann_info['gt_points'].copy()
        
        # Generate pseudo bounding boxes from points
        # Use fixed size or adaptive size based on image dimensions
        h, w = results['img_info']['height'], results['img_info']['width']
        default_size = min(h, w) * 0.05  # 5% of the smaller dimension
        
        gt_points = results['gt_points']
        num_points = len(gt_points)
        
        # Generate pseudo bounding boxes (x, y, w, h, angle)
        gt_bboxes = np.zeros((num_points, 5), dtype=np.float32)
        for i, point in enumerate(gt_points):
            gt_bboxes[i, 0:2] = point  # center x, y
            gt_bboxes[i, 2:4] = default_size, default_size  # width, height
            gt_bboxes[i, 4] = 0.0  # angle
            
        results['gt_bboxes'] = gt_bboxes
        return results
        
    def _load_labels(self, results):
        """Load class labels."""
        results['gt_labels'] = results['ann_info']['gt_labels'].copy()
        return results
        
    def __call__(self, results):
        """Call function."""
        if self.with_point:
            results = self._load_points(results)
        if self.with_label:
            results = self._load_labels(results)
        return results
