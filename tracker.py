from dataclasses import dataclass
from typing import List, Tuple
import time

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0: return 0.0
    areaA = max(1, (boxA[2]-boxA[0])*(boxA[3]-boxA[1]))
    areaB = max(1, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
    return inter / (areaA + areaB - inter + 1e-6)

@dataclass
class Track:
    id: int
    box: Tuple[int,int,int,int]
    last_seen: float
    first_seen: float
    hits: int = 1
    misses: int = 0
    det_ema: float = 0.0
    pressed_at: float | None = None
    press_count: int = 0
    label: str | None = None
    prob: float | None = None

class IoUTracker:
    """
    Tracker IoU đơn giản có đếm hits/misses để ổn định đối tượng qua nhiều khung hình.
    - iou_thres: ngưỡng bắt cặp
    - ttl: xoá track nếu không thấy lại trong TTL giây
    - alpha: hệ số EMA cho confidence
    """
    def __init__(self, iou_thres=0.5, ttl=1.2, alpha=0.6):
        self.iou_thres = iou_thres
        self.ttl = ttl
        self.alpha = alpha
        self._tracks: List[Track] = []
        self._next_id = 1

    def update(self, det_boxes: List[Tuple[int,int,int,int]], det_confs: List[float]) -> List[Track]:
        now = time.time()
        updated: List[Track] = []
        used_track_idx = set()
        used_det_idx = set()

        # Tham lam: với mỗi detection, tìm track có IoU tốt nhất
        for di, box in enumerate(det_boxes):
            match = None; best = 0.0; idx = -1
            for ti, t in enumerate(self._tracks):
                if ti in used_track_idx:
                    continue
                score = iou(t.box, box)
                if score > best and score >= self.iou_thres:
                    best = score; match = t; idx = ti
            if match:
                match.box = box
                match.last_seen = now
                match.hits += 1
                match.misses = 0
                # EMA confidence
                conf = float(det_confs[di]) if det_confs is not None and di < len(det_confs) else 0.0
                match.det_ema = self.alpha*conf + (1.0-self.alpha)*match.det_ema
                updated.append(match)
                used_track_idx.add(idx)
                used_det_idx.add(di)
            else:
                conf = float(det_confs[di]) if det_confs is not None and di < len(det_confs) else 0.0
                tnew = Track(self._next_id, box, now, now, hits=1, misses=0, det_ema=conf)
                self._next_id += 1
                updated.append(tnew)
                used_det_idx.add(di)

        # Tracks không được match -> tăng misses, có thể giữ lại nếu chưa quá TTL
        for ti, t in enumerate(self._tracks):
            if ti in used_track_idx:
                continue
            if (now - t.last_seen) <= self.ttl:
                t.misses += 1
                updated.append(t)

        # Dọn track quá hạn
        self._tracks = [t for t in updated if (now - t.last_seen) <= self.ttl]
        return self._tracks

    def mark_pressed(self, track_id: int):
        for t in self._tracks:
            if t.id == track_id:
                t.pressed_at = time.time()
                t.press_count += 1
                break