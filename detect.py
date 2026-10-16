"""Rotation-ensemble person detection for overhead camera footage."""
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def unrotate_box(box, turns, width, height):
    x1, y1, x2, y2 = box
    points = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
    x, y = points.T
    if turns == 1:
        points = np.column_stack([width - y, x])
    elif turns == 2:
        points = np.column_stack([width - x, height - y])
    elif turns == 3:
        points = np.column_stack([y, height - x])
    lo, hi = points.min(axis=0), points.max(axis=0)
    return np.clip([lo[0], lo[1], hi[0], hi[1]], 0, [width, height, width, height]).tolist()


class PersonDetector:
    def __init__(self, model, device='auto', imgsz=640, rotations=(0, 90, 180, 270), conf=0.12):
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device, self.imgsz, self.rotations, self.conf = device, imgsz, rotations, conf
        Path(model).parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(model))

    def __call__(self, frame):
        height, width = frame.shape[:2]
        candidates = []
        for angle in self.rotations:
            turns = angle // 90
            rotated = np.ascontiguousarray(np.rot90(frame, turns))
            result = self.model.predict(rotated, classes=[0], conf=self.conf, imgsz=self.imgsz,
                                        device=self.device, verbose=False)[0]
            for row in result.boxes.data.cpu().numpy():
                box = unrotate_box(row[:4], turns, width, height)
                candidates.append(box + [float(row[4]), 0.0])
        if not candidates:
            return np.empty((0, 6), dtype=np.float32)
        arr = np.asarray(candidates, dtype=np.float32)
        xywh = arr[:, :4].copy()
        xywh[:, 2:] -= xywh[:, :2]
        keep = cv2.dnn.NMSBoxes(xywh.tolist(), arr[:, 4].tolist(), self.conf, 0.45)
        return arr[np.asarray(keep).reshape(-1)]


class MotionProposals:
    """Optional proposals for a fixed camera; these are not semantic person detections."""
    def __init__(self, fps):
        self.background = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=36, detectShadows=True)
        self.warmup = max(10, round(fps))
        self.frames = 0

    def supplement(self, frame, detections):
        self.frames += 1
        mask = self.background.apply(frame)
        mask = (mask == 255).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        if self.frames < self.warmup:
            return detections
        height, width = frame.shape[:2]
        extra = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < width * height * 0.0015:
                continue
            if min(w, h) < min(width, height) * 0.045 or max(w, h) > max(width, height) * 0.4:
                continue
            if not 0.2 < w / h < 5 or w * h > width * height * 0.12:
                continue
            box = np.array([x, y, x+w, y+h], dtype=float)
            if len(detections):
                intersection = np.maximum(0, np.minimum(detections[:, 2:4], box[2:]) - np.maximum(detections[:, :2], box[:2])).prod(axis=1)
                area = (detections[:, 2:4] - detections[:, :2]).prod(axis=1)
                if np.any(intersection / max(1, w*h) > 0.6):
                    continue
            extra.append([max(0,x-4), max(0,y-4), min(width,x+w+4), min(height,y+h+4), 0.31, 0])
        return np.concatenate([detections, np.asarray(extra, dtype=np.float32).reshape(-1, 6)])
