import cv2
import numpy as np
from collections import deque
import time

def to_binary(crop_bgr):
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return None
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 19, 2)
    if (bw == 255).mean() > 0.6:
        bw = 255 - bw
    kernel = np.ones((3,3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    return bw

def resize_keep_ar(img, size=64):
    if img is None:
        return None
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None
    scale = size / max(h, w)
    nh, nw = max(1, int(h*scale)), max(1, int(w*scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = small
    return canvas

class TemplateBank:
    def __init__(self, max_per_label=6, size=64, max_total=200):
        self.bank = {}  # label -> deque[{"img": np.ndarray, "ts": float}]
        self.max_per_label = int(max_per_label)
        self.size = int(size)
        self.max_total = int(max_total)

    def _total_count(self):
        return sum(len(dq) for dq in self.bank.values())

    def add(self, label: str, crop_bgr, ts: float | None = None):
        bw = to_binary(crop_bgr)
        bw = resize_keep_ar(bw, self.size)
        if bw is None or label is None:
            return
        dq = self.bank.get(label)
        if dq is None:
            dq = deque(maxlen=self.max_per_label)
            self.bank[label] = dq
        dq.append({"img": bw, "ts": float(ts or time.time())})
        # limit total
        while self._total_count() > self.max_total:
            max_label = None
            max_len = -1
            for k, v in self.bank.items():
                if len(v) > max_len:
                    max_len = len(v); max_label = k
            if max_label is None:
                break
            if self.bank[max_label]:
                self.bank[max_label].popleft()
            if len(self.bank[max_label]) == 0:
                del self.bank[max_label]

    def score(self, label: str, crop_bgr) -> float:
        if label not in self.bank or len(self.bank[label]) == 0:
            return 0.0
        bw = to_binary(crop_bgr)
        bw = resize_keep_ar(bw, self.size)
        if bw is None:
            return 0.0
        best = 0.0
        for item in self.bank[label]:
            tmpl = item["img"]
            res = cv2.matchTemplate(bw, tmpl, cv2.TM_CCOEFF_NORMED)
            val = float(res.max()) if res.size > 0 else 0.0
            if val > best:
                best = val
        return best

    def best_match(self, crop_bgr):
        if not self.bank:
            return None, 0.0
        bw = to_binary(crop_bgr)
        bw = resize_keep_ar(bw, self.size)
        if bw is None:
            return None, 0.0
        best_label, best_score = None, 0.0
        for label, dq in self.bank.items():
            for item in dq:
                tmpl = item["img"]
                res = cv2.matchTemplate(bw, tmpl, cv2.TM_CCOEFF_NORMED)
                val = float(res.max()) if res.size > 0 else 0.0
                if val > best_score:
                    best_score = val
                    best_label = label
        return best_label, best_score

    def prune(self, ttl_sec=8.0, now: float | None = None, min_keep=1):
        now = float(now or time.time())
        ttl = float(ttl_sec)
        to_del = []
        for label, dq in self.bank.items():
            while len(dq) > min_keep and dq and (now - dq[0]["ts"]) > ttl:
                dq.popleft()
            if len(dq) == 0:
                to_del.append(label)
        for k in to_del:
            del self.bank[k]

    def drop_label(self, label: str):
        if label in self.bank:
            del self.bank[label]

    def clear(self):
        self.bank.clear()