#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp
import json
import numpy as np
from mmdet.datasets.custom import CustomDataset
from mmdet.core import eval_map, eval_recalls

class DOTADataset(CustomDataset):
    """DOTA dataset for detection."""
    
    CLASSES = ('plane', 'ship', 'storage-tank', 'baseball-diamond', 
              'tennis-court', 'basketball-court', 'ground-track-field', 
              'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 
              'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool')
    
    def __init__(self, *args, **kwargs):
        super(DOTADataset, self).__init__(*args, **kwargs)
        # Create a mapping from class name to index
        self.class_to_idx = {cls: i for i, cls in enumerate(self.CLASSES)}
        
    def load_annotations(self, ann_file):
        """Load annotation from point annotation file."""
        with open(ann_file) as f:
            data = json.load(f)
            
        data_infos = []
        for img_id, img_info in data.items():
            filename = img_info['file_name']
            
            # Get image path
            img_path = osp.join(self.img_prefix, filename)
            
            # Get image size
            height, width = self._get_image_size(img_path)
            
            data_info = dict(
                filename=filename,
                width=width,
                height=height
            )
            
            # Add point annotations
            gt_points = np.array(img_info['points'], dtype=np.float32)
            gt_labels = [self.class_to_idx[cls] for cls in img_info['labels']]
            
            data_info['ann'] = dict(
                gt_points=gt_points,
                gt_labels=np.array(gt_labels, dtype=np.int64)
            )
            
            data_infos.append(data_info)
            
        return data_infos
    
    def _get_image_size(self, img_path):
        """Get image size (height, width) from image path."""
        try:
            import cv2
            img = cv2.imread(img_path)
            if img is not None:
                return img.shape[:2]
            else:
                return (1024, 1024)  # Default size if image cannot be read
        except:
            return (1024, 1024)  # Default size if image cannot be read
        
    def get_ann_info(self, idx):
        """Get annotation by index."""
        return self.data_infos[idx]['ann']
        
    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds
        
    def prepare_train_img(self, idx):
        """Get training data and annotations."""
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def evaluate(self,
                results,
                metric='mAP',
                logger=None,
                proposal_nums=(100, 300, 1000),
                iou_thr=0.5,
                scale_ranges=None):
        """Evaluate the dataset.
        
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            gt_bboxes = [ann['gt_bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes,
                results,
                proposal_nums,
                iou_thr,
                logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
        
        return eval_results
