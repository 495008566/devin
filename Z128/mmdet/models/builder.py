from mmcv.utils import Registry

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
HEADS = Registry('head')
LOSSES = Registry('loss')
DETECTORS = Registry('detector')

def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)

def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)

def build_roi_extractor(cfg):
    """Build roi extractor."""
    return ROI_EXTRACTORS.build(cfg)

def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)

def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)

def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    return DETECTORS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
