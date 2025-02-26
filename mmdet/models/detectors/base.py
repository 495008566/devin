#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

class BaseDetector(nn.Module):
    """Base class for detectors."""
    
    def __init__(self):
        super(BaseDetector, self).__init__()
        
    def init_weights(self, pretrained=None):
        """Initialize the weights."""
        pass
        
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Forward function."""
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)
            
    def forward_train(self, img, img_metas, **kwargs):
        """Forward function for training."""
        pass
        
    def forward_test(self, img, img_metas, **kwargs):
        """Forward function for testing."""
        pass
        
    def simple_test(self, img, img_metas, **kwargs):
        """Test without augmentation."""
        pass
        
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test with augmentation."""
        pass
        
    @property
    def with_neck(self):
        """Check if has neck."""
        return hasattr(self, 'neck') and self.neck is not None
        
    @property
    def with_bbox(self):
        """Check if has bbox head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None
        
    @property
    def with_mask(self):
        """Check if has mask head."""
        return hasattr(self, 'mask_head') and self.mask_head is not None
