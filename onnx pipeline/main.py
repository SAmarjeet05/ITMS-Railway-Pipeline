# -------------------- main.py --------------------
import onnxruntime as ort
import cv2
import sys
from utils.preprocess import preprocess_image
from utils.postprocess import nms_boxes
from utils.visualize import draw_boxes
import numpy as np
import os
import json
import time
import random
from datetime import datetime

# ---------------- SETTINGS ----------------
IMAGE_FOLDER = "../test/images"
JSON_OUTPUT = "detection_results.json"
CONF_THRESH = 0.6
IOU_THRESH = 0.5
CLASS_NAMES = ["defective", "non-defective"]
FRAME_DELAY = 1  # seconds between images

# ---------------- LOAD MODEL ----------------
MODEL_TYPE = sys.argv[1] if len(sys.argv) > 1 else 'onnx'
USER_CONF_THRESH = float(sys.argv[2]) if len(sys.argv) > 2 else CONF_THRESH
USER_IOU_THRESH = float(sys.argv[3]) if len(sys.argv) > 3 else IOU_THRESH

if MODEL_TYPE == 'onnx':
    ort_session = ort.InferenceSession("models/best.onnx")
elif MODEL_TYPE == 'yolov8':
    from ultralytics import YOLO
    yolo_model = YOLO("../best.pt")
else:
    print(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
    sys.exit(1)

results = []
processed_images = set()

print("Starting dynamic time-based ONNX detection...")


# ---------------- FUNCTIONS ----------------
def xywh_to_xyxy(boxes):
    """Convert [xc, yc, w, h] → [x1, y1, x2, y2]."""
    x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def process_preds(outputs, conf_thresh, iou_thresh):
    """
    Convert raw ONNX outputs to boxes, scores, class_ids.
    Assumes model already outputs xc,yc,w,h,conf,cls.
    """
    preds = outputs[0][0].T  # shape (N,6) [xc, yc, w, h, conf, cls]

    boxes = xywh_to_xyxy(preds[:, :4])
    scores = preds[:, 4]
    class_ids = preds[:, 5].astype(int)

    # Filter by confidence
    mask = scores > conf_thresh
    boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Run NMS
    packed = np.hstack((boxes, scores[:, None], class_ids[:, None]))
    boxes, scores, class_ids = nms_boxes(
        packed, conf_thresh=conf_thresh, iou_thresh=iou_thresh
    )

    return boxes, scores, class_ids


# ---------------- MAIN LOOP ----------------
while True:
    images = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
              if f.lower().endswith(('.jpg', '.png')) and f not in processed_images]
    if not images:
        print("No more images to process. Exiting...")
        break

    image_path = images[0]
    processed_images.add(os.path.basename(image_path))

    # ---------------- Preprocess ----------------
    input_tensor, orig_shape = preprocess_image(image_path, return_shape=True)
    print(f"[DEBUG] Processing {os.path.basename(image_path)} | Original shape: {orig_shape[:2]} | Input tensor shape: {input_tensor.shape}")

    # ---------------- Inference ----------------
    t0 = time.time()
    outputs = ort_session.run(None, {"images": input_tensor})
    fps = 1 / (time.time() - t0)
    print(f"[DEBUG] Inference done | FPS: {round(fps,2)}")

    # ---------------- Postprocess ----------------
    boxes, scores, class_ids = process_preds(outputs, USER_CONF_THRESH, USER_IOU_THRESH)
    print(f"[DEBUG] Boxes detected: {len(boxes)}")

    # ---------------- Draw boxes ----------------

    # ---------------- Store results ----------------
    gps = {"lat": round(random.uniform(28.6, 28.7), 6),
           "lon": round(random.uniform(77.2, 77.3), 6)}
    status = "defective" if len(boxes) > 0 else "non-defective"
    detections = [{"label": CLASS_NAMES[int(c)], "confidence": float(s),
                   "box": {"x1": float(b[0]), "y1": float(b[1]),
                           "x2": float(b[2]), "y2": float(b[3])}}
                  for b, s, c in zip(boxes, scores, class_ids)]
    result = {
        "image_name": os.path.basename(image_path),
        "timestamp": datetime.now().isoformat(),
        "gps": gps,
        "status": status,
        "detections": detections,
        "fps": round(fps, 2)
    }
    results.append(result)
    print(f"[INFO] Processed {os.path.basename(image_path)} | Status: {status} | FPS: {round(fps,2)}")

    time.sleep(FRAME_DELAY)

# ---------------- SAVE RESULTS ----------------
with open(JSON_OUTPUT, "w") as f:
    json.dump(results, f, indent=4)
print(f"All results saved to {JSON_OUTPUT}")
