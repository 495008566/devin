#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

class BBoxHead(nn.Module):
    """Bbox head."""
    
    def __init__(self,
                with_avg_pool=False,
                with_cls=True,
                with_reg=True,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=None,
                reg_class_agnostic=False,
                loss_cls=None,
                loss_bbox=None):
        super(BBoxHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels
        self.roi_feat_size = roi_feat_size
        self.num_classes = num_classes
        self.bbox_coder = bbox_coder
        self.reg_class_agnostic = reg_class_agnostic
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        
        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            in_channels *= roi_feat_size * roi_feat_size
            
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes + 1)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
            
    def init_weights(self):
        """Initialize the weights."""
        pass
        
    def forward(self, x):
        """Forward function."""
        pass
