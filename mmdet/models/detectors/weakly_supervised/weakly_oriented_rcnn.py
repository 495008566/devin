#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from ..base import BaseDetector
from ...builder import DETECTORS, build_backbone, build_head, build_neck

@DETECTORS.register_module()
class WeaklyOrientedRCNN(BaseDetector):
    """Weakly supervised Oriented R-CNN for rotation detection."""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(WeaklyOrientedRCNN, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.rpn_head = build_head(rpn_head)
        self.roi_head = build_head(roi_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights."""
        super(WeaklyOrientedRCNN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.rpn_head.init_weights()
        self.roi_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Dummy forward function."""
        x = self.extract_feat(img)
        rpn_outs = self.rpn_head(x)
        proposals = self.rpn_head.get_bboxes(*rpn_outs, img_metas=None, cfg=self.test_cfg.rpn)
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        return roi_outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_points,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """Forward function for training."""
        x = self.extract_feat(img)
        
        # RPN forward and loss
        rpn_outs = self.rpn_head(x)
        rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas)
        rpn_losses = self.rpn_head.loss(
            *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        
        # Generate proposals
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        
        # ROI forward and loss
        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs)
        
        losses = {}
        losses.update(rpn_losses)
        losses.update(roi_losses)
        
        return losses
        
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        
        x = self.extract_feat(img)
        
        # Get proposals if not provided
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
            
        # ROI forward
        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
            
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentation."""
        raise NotImplementedError
        
    def show_result(self, img, result, score_thr=0.3, wait_time=0, show=True, out_file=None):
        """Show detection results."""
        # For demonstration, just use a simple visualization
        import cv2
        import numpy as np
        
        img = img.copy()
        
        # Draw bounding boxes
        for bbox, label, score in zip(result[0], result[1], result[2]):
            if score < score_thr:
                continue
                
            # Convert to int coordinates
            bbox = bbox.astype(np.int32)
            
            # Draw rotated rectangle
            rect = ((bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox[4])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Get color based on label
            color = (0, 255, 0)  # Default green
            
            # Draw the box
            cv2.drawContours(img, [box], 0, color, 2)
            
            # Put label and score
            cv2.putText(img, f'{self.CLASSES[label]}: {score:.2f}',
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
        # Show or save the image
        if show:
            cv2.imshow('result', img)
            cv2.waitKey(wait_time)
        if out_file is not None:
            cv2.imwrite(out_file, img)
            
        return img
