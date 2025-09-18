import numpy as np

def nms_boxes(preds, conf_thresh=0.25, iou_thresh=0.45):
    """
    preds: (N, 6) -> [x1, y1, x2, y2, conf, class_id]
    Returns: boxes, scores, class_ids after NMS
    """
    if preds.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])

    boxes = preds[:, :4]
    scores = preds[:, 4]
    class_ids = preds[:, 5].astype(int)

    # Filter by confidence
    mask = scores >= conf_thresh
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(scores) == 0:
        return np.array([]), np.array([]), np.array([])

    # Perform NMS
    keep = []
    idxs = scores.argsort()[::-1]

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        if len(idxs) == 1:
            break

        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thresh]

    return boxes[keep], scores[keep], class_ids[keep]

def compute_iou(box, boxes):
    """Compute IoU between 1 box and many boxes"""
    if len(boxes) == 0:
        return np.array([])
    
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union_area = box_area + boxes_area - inter_area

    return inter_area / (union_area + 1e-6)
