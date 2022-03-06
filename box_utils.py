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

def nms(boxes, score=[], overlap_thresh=0.3):
    """Apply non-maximum suppression to boxes

    Args:
        boxes ()
    """

    if len(boxes) == 0:
        return []
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes).astype(float)
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
        
    pick_id = []
    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]
    area = (xmax - xmin) * (ymax - ymin)
    if score:
        idxs = np.argsort(score)
    else:
        assert len(score) == len(boxes), print(f'Assert length of boxes and score are the same, \
            len(boxes) = {len(boxes)} and len(score) = {len(score)}')
        idxs = np.argsort(ymax)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick_id.append(i)

        xx1 = np.maximum(xmin[i], xmin[idxs[:last]])
        yy1 = np.maximum(ymin[i], ymin[idxs[:last]])
        xx2 = np.minimum(xmax[i], xmax[idxs[:last]])
        yy2 = np.minimum(ymax[i], ymax[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], \
            np.where(overlap > overlap_thresh)[0])))
    return boxes[pick_id].astype('int')
