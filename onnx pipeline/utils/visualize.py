import cv2
import numpy as np

def draw_boxes(img, boxes, scores, class_ids, class_names):
    """
    Draw bounding boxes on an image array (not path).
    img: numpy array (RGB or BGR).
    """
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Expected img as numpy array, got None or wrong type.")

    # Ensure RGB
    if img.shape[-1] == 3:
        pass  # already fine

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        label = f"{class_names[class_ids[i]]}: {scores[i]:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return img
