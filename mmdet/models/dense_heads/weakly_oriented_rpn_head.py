#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .rpn_head import RPNHead

@HEADS.register_module()
class WeaklyOrientedRPNHead(RPNHead):
    """RPN head for weakly supervised rotation detection."""
    
    def __init__(self, **kwargs):
        super(WeaklyOrientedRPNHead, self).__init__(**kwargs)
        
    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred
        
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head."""
        # For demonstration, use a simplified loss calculation
        losses = dict()
        
        # Classification loss
        labels = torch.zeros_like(cls_scores[0])
        losses['loss_rpn_cls'] = F.binary_cross_entropy_with_logits(
            cls_scores[0], labels)
            
        # Regression loss
        losses['loss_rpn_bbox'] = torch.tensor(0.0, device=cls_scores[0].device)
        
        return losses
        
    def get_bboxes(self,
                  cls_scores,
                  bbox_preds,
                  img_metas,
                  cfg=None,
                  rescale=False):
        """Transform network output for a batch into bbox predictions."""
        # For demonstration, return dummy proposals
        batch_size = len(img_metas)
        proposals = [torch.randn(1000, 5) for _ in range(batch_size)]
        return proposals
