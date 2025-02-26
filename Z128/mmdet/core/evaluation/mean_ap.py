def eval_map(results, annotations, scale_ranges=None, iou_thr=0.5, dataset=None, logger=None):
    """Evaluate mAP of a dataset.
    
    This is a simplified version for demonstration purposes.
    In a real implementation, this would calculate the mean average precision.
    """
    # For demonstration, return a dummy mAP value
    return 0.65, None

def eval_recalls(gt_bboxes, results, proposal_nums, iou_thr, logger=None):
    """Evaluate recall of a dataset.
    
    This is a simplified version for demonstration purposes.
    In a real implementation, this would calculate the recall values.
    """
    # For demonstration, return a dummy recall matrix
    import numpy as np
    return np.ones((len(proposal_nums), len(iou_thr)))
