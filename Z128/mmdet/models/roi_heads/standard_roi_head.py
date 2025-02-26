#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

class StandardRoIHead(nn.Module):
    """Standard RoI head."""
    
    def __init__(self,
                bbox_roi_extractor=None,
                bbox_head=None):
        super(StandardRoIHead, self).__init__()
        self.bbox_roi_extractor = bbox_roi_extractor
        self.bbox_head = bbox_head
        
    def init_weights(self):
        """Initialize the weights."""
        pass
        
    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        return None
