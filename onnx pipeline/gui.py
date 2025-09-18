# gui_app.py
import sys
import os
import cv2
import json
import time
import random
import numpy as np
from datetime import datetime
import onnxruntime as ort

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QTextEdit
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage

from utils.preprocess import preprocess_image
from utils.postprocess import nms_boxes
from utils.visualize import draw_boxes


# ---------------- SETTINGS ----------------
IMAGE_FOLDER = "../test/images"
JSON_OUTPUT = "detection_results.json"
CONF_THRESH = 0.6
IOU_THRESH = 0.5
CLASS_NAMES = ["defective", "non-defective"]

# ---------------- LOAD MODEL ----------------
ort_session = ort.InferenceSession("models/best.onnx")


# ---------------- HELPERS ----------------
def xywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def process_preds(outputs, conf_thresh, iou_thresh):
    preds = outputs[0][0].T  # shape (N,6)
    boxes = xywh_to_xyxy(preds[:, :4])
    scores = preds[:, 4]
    class_ids = preds[:, 5].astype(int)

    mask = scores > conf_thresh
    boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    packed = np.hstack((boxes, scores[:, None], class_ids[:, None]))
    boxes, scores, class_ids = nms_boxes(packed, conf_thresh, iou_thresh)
    return boxes, scores, class_ids


# ---------------- MAIN GUI ----------------
class DetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Railway Track Detection GUI")
        self.resize(1000, 700)

        self.layout = QVBoxLayout(self)

        # Countdown label
        self.counter_label = QLabel("Press Start", self)
        self.counter_label.setAlignment(Qt.AlignCenter)
        self.counter_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.layout.addWidget(self.counter_label)

        # Image label
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # JSON text
        self.json_text = QTextEdit(self)
        self.json_text.setReadOnly(True)
        self.layout.addWidget(self.json_text)

        # Start button
        self.start_btn = QPushButton("Start Detection", self)
        self.start_btn.clicked.connect(self.start_cycle)
        self.layout.addWidget(self.start_btn)

        # Vars
        self.counter = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_countdown)

        self.images = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
                       if f.lower().endswith(('.jpg', '.png'))]
        self.image_index = 0
        self.results = []

    def start_cycle(self):
        if self.image_index >= len(self.images):
            self.counter_label.setText("✅ All images processed")
            return
        self.counter = 1
        self.counter_label.setText("1")
        self.timer.start(250)  # 0.5 sec per step

    def update_countdown(self):
        if self.counter < 25:
            self.counter += 1
            self.counter_label.setText(str(self.counter))
        else:
            self.timer.stop()
            self.run_detection()

    def run_detection(self):
        if self.image_index >= len(self.images):
            self.counter_label.setText("✅ All images processed")
            return

        image_path = self.images[self.image_index]
        self.image_index += 1

        # Preprocess
        input_tensor, orig_shape = preprocess_image(image_path, return_shape=True)

        # Inference
        t0 = time.time()
        outputs = ort_session.run(None, {"images": input_tensor})
        fps = 1 / (time.time() - t0)

        # Postprocess
        boxes, scores, class_ids = process_preds(outputs, CONF_THRESH, IOU_THRESH)

        # Draw boxes
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = draw_boxes(img, boxes, scores, class_ids, CLASS_NAMES)

        # Convert image for QLabel
        h, w, ch = img.shape
        qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(600, 400, Qt.KeepAspectRatio))

        # JSON results
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
        self.results.append(result)

        # Show JSON
        self.json_text.setPlainText(json.dumps(result, indent=4))

        # Save JSON
        with open(JSON_OUTPUT, "w") as f:
            json.dump(self.results, f, indent=4)

        # Start next countdown automatically
        QTimer.singleShot(1000, self.start_cycle)  # wait 1s then start next cycle


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DetectionApp()
    win.show()
    sys.exit(app.exec_())
