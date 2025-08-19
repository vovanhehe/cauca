"""
WGCCapture cho Windows Graphics Capture (dxcam 0.0.5) + fallback MSS.
- Ưu tiên async (start/get_latest_frame); nếu lỗi, rơi về grab blocking.
- Vá lỗi dxcam 0.0.5: __del__/stop/release có thể truy cập is_capturing không tồn tại -> monkey-patch class DXCamera.
- Đảm bảo frame luôn C-contiguous để OpenCV không lỗi.
"""
from __future__ import annotations
import atexit
import numpy as np

class WGCCapture:
    def __init__(self, output_idx: int = 0, prefer_async: bool = True, target_fps: int = 120, max_buffer_len: int = 4):
        self.backend = None
        self.cam = None
        self.prefer_async = bool(prefer_async)
        self.target_fps = int(target_fps)
        self.max_buffer_len = int(max_buffer_len)
        self.region = None
        self.started = False
        self.output_idx = int(output_idx)

        # Thử dxcam 0.0.5
        try:
            import dxcam as _dx
            self.backend = "dxcam"

            # Monkey-patch cho DXCamera ở cả 2 kiểu đóng gói
            candidates = []
            DXC1 = getattr(_dx, "DXCamera", None)
            if DXC1 is not None:
                candidates.append(DXC1)
            try:
                from dxcam.dxcam import DXCamera as DXC2
            except Exception:
                DXC2 = None
            if DXC2 is not None and DXC2 not in candidates:
                candidates.append(DXC2)

            for DXC in candidates:
                try:
                    if not hasattr(DXC, "is_capturing"):
                        setattr(DXC, "is_capturing", False)
                    orig_stop = getattr(DXC, "stop", None)
                    if callable(orig_stop):
                        def _safe_stop(self_):
                            try:
                                return orig_stop(self_)
                            except Exception:
                                return None
                        setattr(DXC, "stop", _safe_stop)
                    orig_release = getattr(DXC, "release", None)
                    if callable(orig_release):
                        def _safe_release(self_):
                            try:
                                return orig_release(self_)
                            except Exception:
                                return None
                        setattr(DXC, "release", _safe_release)
                    def _safe_del(self_):
                        try:
                            if hasattr(self_, "stop"):
                                self_.stop()
                        except Exception:
                            pass
                    setattr(DXC, "__del__", _safe_del)
                except Exception:
                    pass

            # Tạo camera (BGRA)
            self.cam = _dx.create(output_idx=self.output_idx, output_color="BGRA")
            try:
                if not hasattr(self.cam, "is_capturing"):
                    setattr(self.cam, "is_capturing", False)
            except Exception:
                pass

            self._dx = _dx

        except Exception:
            # Fallback MSS
            self.backend = "mss"
            import mss
            self.sct = mss.mss()

        atexit.register(self._safe_atexit_stop)

    def _safe_atexit_stop(self):
        try:
            self.stop()
        except Exception:
            pass

    def start(self, region: tuple[int,int,int,int], target_fps: int | None = None, max_buffer_len: int | None = None):
        self.region = region
        if self.backend == "dxcam" and self.cam is not None and self.prefer_async:
            tfps = int(target_fps or self.target_fps)
            mbl = int(max_buffer_len or self.max_buffer_len)
            try:
                self.cam.start(target_fps=tfps, region=self.region, max_buffer_len=mbl)
                self.started = True
            except TypeError:
                try:
                    self.cam.start(target_fps=tfps)
                    self.started = True
                except Exception:
                    self.prefer_async = False
            except Exception:
                self.prefer_async = False

    def _ensure_contig(self, arr):
        if arr is None:
            return None
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr

    def get_latest(self):
        if self.backend == "dxcam" and self.cam is not None and self.started:
            try:
                frame = self.cam.get_latest_frame()
            except Exception:
                frame = None
            if frame is None:
                return None
            if self.region is not None:
                x1,y1,x2,y2 = self.region
                frame = frame[y1:y2, x1:x2]
            return self._ensure_contig(frame)
        return None

    def grab(self, region: tuple[int,int,int,int] | None = None):
        region = region or self.region
        if self.backend == "dxcam" and self.cam is not None:
            try:
                frame = self.cam.grab(region=region)
                return self._ensure_contig(frame)
            except Exception:
                return None
        # MSS fallback
        x1,y1,x2,y2 = region if region is not None else (0,0,0,0)
        w = max(1, x2-x1); h = max(1, y2-y1)
        mon = {"left": x1, "top": y1, "width": w, "height": h}
        try:
            img = self.sct.grab(mon)  # BGRA
        except Exception:
            return None
        frame = np.array(img)
        return self._ensure_contig(frame)

    def stop(self):
        if self.backend == "dxcam" and self.cam is not None:
            try:
                self.cam.stop()
            except Exception:
                pass
        self.started = False