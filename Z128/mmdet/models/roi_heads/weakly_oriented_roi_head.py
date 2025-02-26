#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from ..builder import HEADS
from .standard_roi_head import StandardRoIHead

@HEADS.register_module()
class WeaklyOrientedStandardRoIHead(StandardRoIHead):
    """Weakly supervised RoI head for rotation detection."""
    
    def __init__(self, **kwargs):
        super(WeaklyOrientedStandardRoIHead, self).__init__(**kwargs)
        
    def forward_train(self,
                     x,
                     img_metas,
                     proposal_list,
                     gt_bboxes,
                     gt_labels,
                     gt_bboxes_ignore=None,
                     gt_masks=None):
        """Forward function for training."""
        # For demonstration, use a simplified training process
        losses = dict()
        
        # Classification loss
        losses['loss_cls'] = torch.tensor(0.1, device=x[0].device)
        
        # Regression loss
        losses['loss_bbox'] = torch.tensor(0.1, device=x[0].device)
        
        return losses
        
    def simple_test(self,
                   x,
                   proposal_list,
                   img_metas,
                   rescale=False):
        """Test without augmentation."""
        # For demonstration, return dummy results
        batch_size = len(img_metas)
        det_bboxes = [torch.randn(10, 5) for _ in range(batch_size)]
        det_labels = [torch.randint(0, 15, (10,)) for _ in range(batch_size)]
        det_scores = [torch.rand(10) for _ in range(batch_size)]
        
        return list(zip(det_bboxes, det_labels, det_scores))
