#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

class RPNHead(nn.Module):
    """RPN head."""
    
    def __init__(self,
                in_channels,
                feat_channels=256,
                anchor_generator=None,
                bbox_coder=None,
                loss_cls=None,
                loss_bbox=None):
        super(RPNHead, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.anchor_generator = anchor_generator
        self.bbox_coder = bbox_coder
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        
        self.rpn_conv = nn.Conv2d(in_channels, feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(feat_channels, 1, 1)
        self.rpn_reg = nn.Conv2d(feat_channels, 4, 1)
        
    def init_weights(self):
        """Initialize the weights."""
        pass
        
    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        pass
        
    def forward(self, x):
        """Forward function."""
        pass
        
    def get_bboxes(self, *args, **kwargs):
        """Get bboxes."""
        pass
        
    def loss(self, *args, **kwargs):
        """Compute loss."""
        pass
