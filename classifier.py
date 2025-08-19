import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18
import numpy as np
from pathlib import Path

class KeyClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x): return self.backbone(x)

def _try_load_class_names_from_ckpt(ckpt):
    # Hỗ trợ nhiều key phổ biến
    for k in ["class_names", "classes", "names", "idx_to_class", "labels"]:
        if k in ckpt:
            val = ckpt[k]
            if isinstance(val, dict):
                # idx_to_class kiểu {"0":"a","1":"b",...}
                try:
                    items = sorted(((int(i), v) for i, v in val.items()), key=lambda x: x[0])
                    return [v for _, v in items]
                except Exception:
                    continue
            elif isinstance(val, (list, tuple)):
                return list(val)
    return None

def _try_load_class_names_from_file(weights_path: str):
    p = Path(weights_path).parent / "cls_labels.txt"
    if p.exists():
        try:
            names = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
            return names if names else None
        except Exception:
            return None
    return None

class Predictor:
    """
    Classifier 1 ký tự.
    - Tự dò class_names từ checkpoint hoặc file cls_labels.txt để đảm bảo đúng thứ tự train.
    - TTA (ensemble) + mask allowed_idx nếu có.
    """
    def __init__(self, weights_path: str, class_names_cfg: list[str] | None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        try:
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(weights_path, map_location=self.device)

        # Xác định class_names theo thứ tự ưu tiên
        class_names = _try_load_class_names_from_ckpt(ckpt)
        if class_names is None:
            class_names = _try_load_class_names_from_file(weights_path)
        if class_names is None:
            # Fallback: dùng config nhưng cảnh báo
            class_names = list(class_names_cfg or [])
            print("[WARN] ckpt không có class_names; dùng config keys. Hãy đảm bảo thứ tự khớp khi train!")

        self.class_names = class_names

        # Build model theo đúng số lớp
        self.model = KeyClassifier(num_classes=len(self.class_names))
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        # strict=True để phát hiện lệch kiến trúc
        missing, unexpected = [], []
        try:
            info = self.model.load_state_dict(state, strict=True)
            if hasattr(info, "missing_keys"): missing = info.missing_keys
            if hasattr(info, "unexpected_keys"): unexpected = info.unexpected_keys
        except Exception as e:
            # Thử strict=False nếu ckpt bọc khác
            res = self.model.load_state_dict(state, strict=False)
            if hasattr(res, "missing_keys"): missing = res.missing_keys
            if hasattr(res, "unexpected_keys"): unexpected = res.unexpected_keys
        self.model.eval().to(self.device)
        print(f"[CLS] loaded weights from {weights_path} | missing={len(missing)} unexpected={len(unexpected)} strict=True")
        print(f"[CLS] classes={len(self.class_names)} -> {self.class_names}")

        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize((96, 96)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def _to_tensor(self, crop_bgr):
        img = crop_bgr[:, :, ::-1]  # BGR->RGB
        return self.tf(img).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict_proba(self, crop_bgr) -> np.ndarray:
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return np.zeros((len(self.class_names),), dtype=np.float32)
        x = self._to_tensor(crop_bgr)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)
        return probs

    @torch.inference_mode()
    def predict_ensemble(self, crops_bgr: list, allowed_idx: set[int] | None = None) -> tuple[str, float, float]:
        valid = [c for c in crops_bgr if c is not None and getattr(c, "size", 0) != 0]
        if not valid:
            return "?", 0.0, 0.0
        probs = [self.predict_proba(c) for c in valid]
        mean_prob = np.mean(probs, axis=0)

        if allowed_idx:
            mask = np.full_like(mean_prob, -1.0, dtype=np.float32)
            for i in allowed_idx:
                if 0 <= i < mean_prob.shape[0]:
                    mask[i] = mean_prob[i]
            mean_prob = mask

        top1 = int(np.argmax(mean_prob))
        if mean_prob[top1] < 0:
            return "?", 0.0, 0.0
        part = mean_prob.copy()
        part[top1] = -1.0
        top2 = int(np.argmax(part))
        return self.class_names[top1], float(mean_prob[top1]), float(max(0.0, part[top2]))