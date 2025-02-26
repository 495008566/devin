#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from ..builder import HEADS
from .bbox_head import BBoxHead

@HEADS.register_module()
class WeaklyOrientedBBoxHead(BBoxHead):
    """Bbox head for weakly supervised rotation detection."""
    
    def __init__(self, **kwargs):
        super(WeaklyOrientedBBoxHead, self).__init__(**kwargs)
        
    def forward(self, x):
        """Forward function."""
        # Shared fully-connected layers
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_cls(x)
        
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        
        return cls_score, bbox_pred
