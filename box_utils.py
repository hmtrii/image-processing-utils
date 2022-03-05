import numpy as np


def overlap_area(box1, box2):
    """ Calculate IoU of 2 input boxes

    Args:
        box1, box2 (numpy.array): [xmin, ymin, xmax, ymax]
    Return:
        Overlap area of box1 and box2 
    """
    
    x_min_union = max(box1[0], box2[0])
    y_min_union = max(box1[1], box2[1])
    x_max_union = min(box1[2], box2[2])
    y_max_union = min(box1[3], box2[3])
    
    inter_area = max(0, x_max_union - x_min_union) * \
                    max(0, y_max_union - y_min_union)
    return inter_area

def IoU(box1, box2):
    """ Calculate IoU of 2 input boxes

    Args:
            box1, box2 (numpy.array): [xmin, ymin, xmax, ymax]
    Return:
        IoU of box1 and box2
    """

    inter_area = overlap_area(box1, box2)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = float(inter_area / (box1_area + box2_area - inter_area))

    return iou
