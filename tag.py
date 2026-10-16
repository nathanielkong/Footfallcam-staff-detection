"""Rotation/scale template matching; scores are correlations, not probabilities."""
from pathlib import Path
import cv2
import numpy as np


class TagDetector:
    def __init__(self, template_dir, widths=(12, 16, 20, 24), angle_step=30):
        self.bank = []
        for path in sorted(Path(template_dir).glob('tag-*.png')):
            template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise ValueError(f'Unreadable template: {path}')
            for width in widths:
                height = max(6, round(width * template.shape[0] / template.shape[1]))
                small = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
                for angle in range(0, 360, angle_step):
                    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
                    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
                    nw, nh = int(np.ceil(height * sin + width * cos)), int(np.ceil(height * cos + width * sin))
                    matrix[:, 2] += [(nw - width) / 2, (nh - height) / 2]
                    rotated = cv2.warpAffine(small, matrix, (nw, nh), borderMode=cv2.BORDER_REPLICATE)
                    feature = rotated.astype(np.float32)
                    feature -= cv2.GaussianBlur(feature, (0, 0), 1.5)
                    self.bank.append((rotated, feature, path.stem))
        if not self.bank:
            raise ValueError(f'No tag-*.png templates in {template_dir}')

    def __call__(self, frame, box):
        x1, y1, x2, y2 = np.round(box).astype(int)
        height, width = frame.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 - x1 < 12 or y2 - y1 < 12:
            return 0.0, None
        #exclude box margins where desks and neighbouring people often enter the crop
        dx, dy = round((x2-x1)*0.1), round((y2-y1)*0.1)
        x1, x2, y1, y2 = x1+dx, x2-dx, y1+dy, y2-dy
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        features = gray.astype(np.float32)
        features -= cv2.GaussianBlur(features, (0, 0), 1.5)
        best, best_box = 0.0, None
        for template, detail, name in self.bank:
            th, tw = template.shape
            if th > gray.shape[0] or tw > gray.shape[1]:
                continue
            response = 0.75 * cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED) + 0.25 * cv2.matchTemplate(features, detail, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(response)
            if score > best:
                patch = gray[y:y+th, x:x+tw]
                if float(patch.std()) < 12:
                    continue
                best = float(score)
                best_box = [x1 + x, y1 + y, x1 + x + tw, y1 + y + th]
        return best, best_box
