import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def _prep_glyph(img_bgr: np.ndarray, out_size: int = 48) -> np.ndarray:
    """
    Chuẩn hoá glyph:
    - BGR -> Gray, blur nhẹ, adaptive threshold -> nhị phân
    - Tự động đảo màu nếu nền trắng chiếm ưu thế
    - Cắt tight bounding box, pad thành vuông, resize về out_size
    Trả về ảnh uint8 (0..255), nền đen chữ trắng.
    """
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 19, 2)
    # Đảo nền: mong muốn chữ trắng trên nền đen
    if (bw == 255).mean() > 0.6:
        bw = 255 - bw

    # Cắt bbox foreground
    ys, xs = np.where(bw > 0)
    if len(xs) == 0 or len(ys) == 0:
        # không có foreground -> trả luôn khung đen
        return np.zeros((out_size, out_size), dtype=np.uint8)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop = bw[y1:y2+1, x1:x2+1]

    # Pad vuông
    h, w = crop.shape[:2]
    m = max(h, w)
    pad_top = (m - h) // 2
    pad_bottom = m - h - pad_top
    pad_left = (m - w) // 2
    pad_right = m - w - pad_left
    sq = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                            borderType=cv2.BORDER_CONSTANT, value=0)
    # Resize
    sq = cv2.resize(sq, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return sq

class GlyphFallback:
    def __init__(self,
                 labels: List[str],
                 store_dir: Optional[str] = None,
                 max_templates_per_label: int = 3,
                 out_size: int = 48):
        self.labels = labels
        self.max_templates_per_label = max_templates_per_label
        self.out_size = out_size
        self.templates: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.store_dir = Path(store_dir) if store_dir else None
        if self.store_dir:
            self.store_dir.mkdir(parents=True, exist_ok=True)

    def add_template(self, label: str, img_bgr: np.ndarray):
        if label not in self.labels or img_bgr is None or getattr(img_bgr, "size", 0) == 0:
            return
        glyph = _prep_glyph(img_bgr, self.out_size)
        # giới hạn số template mỗi nhãn
        lst = self.templates[label]
        lst.append(glyph)
        if len(lst) > self.max_templates_per_label:
            # đơn giản: bỏ template cũ nhất
            lst.pop(0)
        if self.store_dir:
            # lưu tham khảo (không ảnh hưởng runtime)
            idx = len(lst)
            out = self.store_dir / f"{label}_{idx}_{self.out_size}.png"
            try:
                cv2.imwrite(str(out), glyph)
            except Exception:
                pass

    def score(self, img_bgr: np.ndarray) -> Tuple[Optional[str], float, float]:
        """
        Trả về (best_label, best_score, second_best_score)
        - score dùng TM_CCOEFF_NORMED trong [-1..1], càng cao càng tốt.
        """
        if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
            return None, 0.0, 0.0
        glyph = _prep_glyph(img_bgr, self.out_size)
        # match với từng nhãn (tối đa N template/nhãn -> lấy score tốt nhất)
        best_label = None
        best_score = -1.0
        second = -1.0
        for lb, temps in self.templates.items():
            if not temps:
                continue
            # với kích thước bằng nhau, matchTemplate trả về 1 giá trị
            s_max = -1.0
            for t in temps:
                try:
                    res = cv2.matchTemplate(glyph, t, cv2.TM_CCOEFF_NORMED)
                    s = float(res[0,0])
                    if s > s_max:
                        s_max = s
                except Exception:
                    continue
            if s_max > best_score:
                second = best_score
                best_score = s_max
                best_label = lb
            elif s_max > second:
                second = s_max
        if best_score < 0:
            return None, 0.0, 0.0
        return best_label, best_score, max(second, 0.0)