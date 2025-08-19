import os, time, cv2, yaml, torch, threading, json
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
from ultralytics import YOLO
import pydirectinput as pdi
from pynput import keyboard
import tkinter as tk
import queue       
import threading
import hashlib
from tkinter import messagebox, simpledialog

from classifier import Predictor
from template_matcher import TemplateBank
from wgc_direct import WGCCapture

# ============ Utils ============
def resource_path(rel: str) -> str:
    base = Path(__file__).resolve().parents[1]
    return str(base / rel)

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def expand_box(x1,y1,x2,y2, W,H, margin_ratio):
    w = x2-x1; h = y2-y1
    mx = int(w*margin_ratio); my = int(h*margin_ratio)
    return max(0,x1-mx), max(0,y1-my), min(W-1,x2+mx), min(H-1,y2+my)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6
    return inter / ua

def center(b):
    return ((b[0]+b[2])*.5, (b[1]+b[3])*.5)

def dist2(p, q):
    return (p[0]-q[0])**2 + (p[1]-q[1])**2

def scene_signature(frame_bgr):
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (32, 18), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0

# ============ Human-in-the-loop Learning System ============
class AdaptiveLearner:
    def __init__(self, data_dir="learning_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.bypass_learning = True
        
        # File paths
        self.learned_file = self.data_dir / "learned_characters.json"
        self.corrections_file = self.data_dir / "corrections.json"
        self.confirmations_file = self.data_dir / "confirmations.json"
        self.thresholds_file = self.data_dir / "adaptive_thresholds.json"
        
        # Load persistent data
        self.learned_characters = self._load_learned_characters()
        self.corrections = self._load_corrections()
        self.confirmations = self._load_confirmations()
        self.adaptive_thresholds = self._load_thresholds()
        
        # Configuration
        self.confidence_threshold = 0.9  # Auto-learn threshold
        self.enable_double_check = False  # Simple mode

        # SIMPLE NON-BLOCKING SYSTEM
        self.pending_confirmations = queue.Queue()
        self.confirmation_results = {}
        self.popup_active = False
        self.ask_again_requests = []
        
        # Background popup thread
        self.popup_thread = threading.Thread(target=self._popup_worker, daemon=True)
        self.popup_thread.start()
        
        print(f"[LEARN] 🚀 Non-blocking popup system initialized")
        print(f"[LEARN] 🚀 Initialized with {len(self.learned_characters)} learned characters")
        print(f"[LEARN] 🎮 Game keyset: {self.get_valid_game_characters()}")
        print(f"[LEARN] 🚫 Excluded: {self.get_excluded_characters()}")
        
    def _load_learned_characters(self):
        """Load permanently learned characters"""
        if self.learned_file.exists():
            try:
                with open(self.learned_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"[LEARN] 📚 Loaded {len(data)} permanently learned characters")
                    return data
            except Exception as e:
                print(f"[LEARN] Error loading learned characters: {e}")
        return {}
    
    def _save_learned_characters(self):
        """Save permanently learned characters"""
        try:
            with open(self.learned_file, 'w', encoding='utf-8') as f:
                json.dump(self.learned_characters, f, indent=2, ensure_ascii=False)
            print(f"[LEARN] 💾 Saved {len(self.learned_characters)} learned characters")
        except Exception as e:
            print(f"[LEARN] Save error: {e}")
    
    def _load_corrections(self):
        """Load correction history"""
        if self.corrections_file.exists():
            try:
                with open(self.corrections_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[LOAD] Error loading corrections: {e}")
        return {"corrections": [], "stats": {}}
    
    def _load_confirmations(self):
        """Load confirmation history"""
        if self.confirmations_file.exists():
            try:
                with open(self.confirmations_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[LOAD] Error loading confirmations: {e}")
        return {"confirmations": [], "stats": {}}
    
    def _load_thresholds(self):
        """Load adaptive thresholds"""
        if self.thresholds_file.exists():
            try:
                with open(self.thresholds_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[LOAD] Error loading thresholds: {e}")
        return {}
    
    def _save_corrections(self):
        """Save corrections"""
        try:
            with open(self.corrections_file, 'w', encoding='utf-8') as f:
                json.dump(self.corrections, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[SAVE] Error saving corrections: {e}")
    
    def _save_confirmations(self):
        """Save confirmations"""
        try:
            with open(self.confirmations_file, 'w', encoding='utf-8') as f:
                json.dump(self.confirmations, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[SAVE] Error saving confirmations: {e}")
    
    def _save_thresholds(self):
        """Save adaptive thresholds"""
        try:
            with open(self.thresholds_file, 'w', encoding='utf-8') as f:
                json.dump(self.adaptive_thresholds, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[SAVE] Error saving thresholds: {e}")
    
    def get_valid_game_characters(self):
        """Get list of valid game characters (no i,o,x,0)"""
        return "123456789abcdefghjklmnpqrstuvwyz"
    
    def get_excluded_characters(self):
        """Get list of excluded characters"""
        return ["i", "o", "x", "0"]
    
    def _is_valid_game_character(self, char):
        """Check if character is valid in game (no i,o,x,0)"""
        if not char or len(char) != 1:
            return False
        
        char = char.lower()
        return char in self.get_valid_game_characters()
    
    def _get_stable_content_hash(self, crop_img, predicted_label, box=None):
        """Create stable hash for character recognition across movement/lighting"""
        try:
            # Try robust preprocessing first
            robust_hash = self._get_robust_content_hash(crop_img, predicted_label)
            
            # Validate hash quality
            if robust_hash and not robust_hash.startswith(('failed_', 'error_', 'empty_')):
                return robust_hash
            
            # Fallback to simple method
            return self._get_simple_content_hash(crop_img, predicted_label)
            
        except Exception as e:
            print(f"[HASH] All methods failed: {e}")
            return f"fallback_{predicted_label}_{int(time.time() * 1000) % 1000000}"

    def _get_simple_content_hash(self, crop_img, predicted_label):
        """Simple hash method (original implementation)"""
        try:
            resized = cv2.resize(crop_img, (32, 32))
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
            
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
            features = self._extract_visual_features(thresh)
            feature_str = f"{predicted_label}_{features}"
            content_hash = hashlib.md5(feature_str.encode()).hexdigest()[:16]
            return content_hash
        except Exception as e:
            print(f"[SIMPLE-HASH] Error: {e}")
            return f"{predicted_label}_{int(time.time() * 1000) % 1000000}"

    def _get_robust_content_hash(self, crop_img, predicted_label):
        """Robust hash for moving bubbles with lighting/quality issues"""
        try:
            if crop_img is None or crop_img.size == 0:
                return f"empty_{predicted_label}_{int(time.time() * 1000) % 1000000}"
            
            # Multi-approach preprocessing
            processed_images = self._preprocess_multiple_approaches(crop_img)
            
            # Extract features from each approach
            all_features = []
            for img in processed_images:
                features = self._extract_robust_visual_features(img, predicted_label)
                if features:
                    all_features.append(features)
            
            if not all_features:
                return f"failed_{predicted_label}_{int(time.time() * 1000) % 1000000}"
            
            # Combine features with voting
            consensus_features = self._combine_features_with_voting(all_features)
            
            # Create stable hash
            feature_str = f"{predicted_label}_{consensus_features}"
            content_hash = hashlib.md5(feature_str.encode()).hexdigest()[:16]
            
            return content_hash
            
        except Exception as e:
            print(f"[ROBUST-HASH] Error: {e}")
            return f"error_{predicted_label}_{int(time.time() * 1000) % 1000000}"

    def _preprocess_multiple_approaches(self, crop_img):
        """Multiple preprocessing to handle lighting/blur issues"""
        approaches = []
        
        try:
            # Approach 1: Standard normalization
            resized = cv2.resize(crop_img, (32, 32))
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
            approaches.append(gray)
            
            # Approach 2: Histogram equalization (fix lighting)
            equalized = cv2.equalizeHist(gray)
            approaches.append(equalized)
            
            # Approach 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
            clahe_img = clahe.apply(gray)
            approaches.append(clahe_img)
            
            return approaches
            
        except Exception as e:
            print(f"[PREPROCESS] Error: {e}")
            return [crop_img]

    def _extract_robust_visual_features(self, img, predicted_label):
        """Extract robust features that work with poor quality images"""
        try:
            h, w = img.shape
            if h == 0 or w == 0:
                return None
            
            # Multiple thresholding approaches
            thresh_results = []
            
            # Method 1: Adaptive threshold (multiple block sizes)
            for block_size in [7, 11, 15]:
                try:
                    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, block_size, 2)
                    if (thresh == 255).mean() < 0.3:
                        thresh = 255 - thresh
                    thresh_results.append(thresh)
                except:
                    continue
            
            # Method 2: Otsu threshold
            try:
                _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if (otsu == 255).mean() < 0.3:
                    otsu = 255 - otsu
                thresh_results.append(otsu)
            except:
                pass
            
            if not thresh_results:
                return None
            
            # Extract features from all thresholded images
            all_region_features = []
            
            for thresh in thresh_results:
                region_features = self._extract_region_densities(thresh)
                if region_features:
                    all_region_features.append(region_features)
            
            if not all_region_features:
                return None
            
            # Average features across all approaches (consensus)
            consensus = {}
            feature_keys = all_region_features[0].keys()
            
            for key in feature_keys:
                values = [f[key] for f in all_region_features if key in f]
                if values:
                    # Use median instead of mean (more robust to outliers)
                    consensus[key] = float(np.median(values))
            
            # Round to reduce sensitivity
            rounded_features = {k: round(v, 2) for k, v in consensus.items()}
            
            # Create feature string
            feature_str = "_".join([f"{k}:{v:.2f}" for k, v in sorted(rounded_features.items())])
            
            return feature_str
            
        except Exception as e:
            print(f"[ROBUST-FEATURES] Error: {e}")
            return None

    def _extract_region_densities(self, thresh_img):
        """Extract region densities from thresholded image"""
        try:
            h, w = thresh_img.shape
            if h == 0 or w == 0:
                return None
            
            # Define regions (more robust regions)
            regions = {
                "top": thresh_img[:h//3, :],
                "middle": thresh_img[h//3:2*h//3, :],
                "bottom": thresh_img[2*h//3:, :],
                "left": thresh_img[:, :w//3],
                "right": thresh_img[:, 2*w//3:],
                "center": thresh_img[h//4:3*h//4, w//4:3*w//4] if h > 4 and w > 4 else thresh_img
            }
            
            densities = {}
            for name, region in regions.items():
                if region.size > 0:
                    density = np.sum(region == 255) / max(region.size, 1)
                    densities[name] = density
                else:
                    densities[name] = 0.0
            
            # Add aspect ratio and total density
            densities["aspect_ratio"] = w / max(h, 1)
            densities["total_density"] = np.sum(thresh_img == 255) / max(thresh_img.size, 1)
            
            return densities
            
        except Exception as e:
            print(f"[REGION-DENSITY] Error: {e}")
            return None

    def _combine_features_with_voting(self, all_features):
        """Combine features using voting/consensus"""
        try:
            if len(all_features) == 1:
                return all_features[0]
            
            # Simple approach: use most common feature string
            feature_counts = {}
            for features in all_features:
                feature_counts[features] = feature_counts.get(features, 0) + 1
            
            # Return most common
            most_common = max(feature_counts.items(), key=lambda x: x[1])
            return most_common[0]
            
        except Exception as e:
            print(f"[COMBINE-FEATURES] Error: {e}")
            return all_features[0] if all_features else "default"
    
    def _extract_visual_features(self, thresh_img):
        """Extract key visual features for stable recognition"""
        try:
            h, w = thresh_img.shape
            
            if h == 0 or w == 0:
                return "empty_image"
            
            # Divide image into regions
            top = thresh_img[:h//3, :]
            middle = thresh_img[h//3:2*h//3, :]
            bottom = thresh_img[2*h//3:, :]
            left = thresh_img[:, :w//3]
            right = thresh_img[:, 2*w//3:]
            center = thresh_img[h//4:3*h//4, w//4:3*w//4]
            
            # Calculate white pixel density for each region
            features = {
                "top": np.sum(top == 255) / max(top.size, 1),
                "middle": np.sum(middle == 255) / max(middle.size, 1),
                "bottom": np.sum(bottom == 255) / max(bottom.size, 1),
                "left": np.sum(left == 255) / max(left.size, 1),
                "right": np.sum(right == 255) / max(right.size, 1),
                "center": np.sum(center == 255) / max(center.size, 1),
                "total": np.sum(thresh_img == 255) / max(thresh_img.size, 1)
            }
            
            # Round to 3 decimal places for stability
            feature_str = "_".join([f"{k}:{v:.3f}" for k, v in sorted(features.items())])
            return feature_str
            
        except Exception as e:
            print(f"[FEATURES] Error extracting visual features: {e}")
            return "default_features"
    
    def should_confirm(self, predicted_label, confidence, margin, features, obj_id=None, box=None, crop=None):
        """Determine if confirmation is needed based on learning state and confidence"""
        
        if crop is None:
            print(f"[LEARN] No crop image for ID{obj_id}, skipping learning")
            return True
        
        # Check if predicted character is valid in game
        if not self._is_valid_game_character(predicted_label):
            print(f"[LEARN] ⚠️ ID{obj_id}: Invalid game character '{predicted_label}' - forcing confirmation")
            return True
        
        # CREATE hash and debug it - THÊM DEBUG
        content_hash = self._get_stable_content_hash(crop, predicted_label)
        print(f"[DEBUG] ID{obj_id}: '{predicted_label}' (conf:{confidence:.3f})")
        print(f"[DEBUG] Hash: {content_hash}")
        print(f"[DEBUG] In learned: {content_hash in self.learned_characters}")
        
        # Check if permanently learned
        if content_hash in self.learned_characters:
            learned_label = self.learned_characters[content_hash]
            print(f"[DEBUG] Found learned: '{learned_label}'")
            
            if learned_label == predicted_label:
                print(f"[LEARN] ✅ ID{obj_id}: Matches learned '{learned_label}' - SKIP CONFIRMATION")
                return False  # Skip confirmation
            else:
                print(f"[LEARN] ⚠️ ID{obj_id}: AI says '{predicted_label}' but learned '{learned_label}' - confirming")
                return True
        
        # NEW character - check confidence
        if confidence >= self.confidence_threshold:
            print(f"[DEBUG] High confidence - auto-learning")
            self.learned_characters[content_hash] = predicted_label
            self._save_learned_characters()
            print(f"[LEARN] 🚀 ID{obj_id}: Auto-learned high confidence '{predicted_label}' ({confidence:.2f})")
            return False  # Skip confirmation
        else:
            print(f"[DEBUG] Low confidence - need confirmation")
            print(f"[LEARN] 🆕 ID{obj_id}: New character, low confidence ({confidence:.2f}) - needs confirmation")
            return True
    
    def sequential_confirm_bubbles(self, bubble_candidates):
        """Process bubbles with permanent learning system"""
        results = {}
        
        if not bubble_candidates:
            return results
        
        print(f"\n[LEARN] 🎯 Processing {len(bubble_candidates)} bubble candidates:")
        for obj_id, label, _, confidence, _, _ in bubble_candidates:
            print(f"  - ID{obj_id}: '{label}' (confidence: {confidence:.2f})")
        
        for obj_id, label, crop, confidence, margin, box in bubble_candidates:
            content_hash = self._get_stable_content_hash(crop, label)
            
            # Check if already learned
            if content_hash in self.learned_characters:
                learned_label = self.learned_characters[content_hash]
                print(f"[LEARN] 🧠 ID{obj_id}: Using learned '{learned_label}' (hash: {content_hash[:8]}...)")
                results[obj_id] = learned_label
                continue
            
            # Check if auto-learned by high confidence
            if confidence >= self.confidence_threshold:
                print(f"[LEARN] 🚀 ID{obj_id}: Auto-learning '{label}' (high confidence: {confidence:.2f})")
                self.learned_characters[content_hash] = label
                self._save_learned_characters()
                results[obj_id] = label
                continue
                
            # Need user confirmation for low confidence
            print(f"[LEARN] 🆕 ID{obj_id}: Learning new character '{label}' (low confidence: {confidence:.2f})")
            
            confirmed_label = self.request_confirmation_simple(
                obj_id, label, crop, confidence, box
            )
            
            results[obj_id] = confirmed_label
            
            # PERMANENTLY save learned character
            self.learned_characters[content_hash] = confirmed_label
            self._save_learned_characters()
            
            # Record learning data
            if confirmed_label == label:
                self._record_positive_confirmation(label, crop, confidence, f"id{obj_id}")
            else:
                self._record_correction(label, confirmed_label, crop, f"id{obj_id}")
            
            print(f"[LEARN] 💾 ID{obj_id}: Permanently learned hash {content_hash[:8]}... = '{confirmed_label}'")
            
            time.sleep(0.2)  # Brief pause between confirmations
            
        return results
    
    def request_confirmation_simple(self, obj_id, predicted_label, crop_img, confidence, box):
        """Simple confirmation popup without double-check"""
        try:
            print(f"[POPUP] 📋 Showing confirmation for ID{obj_id}: '{predicted_label}'")
            
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            root.lift()
            root.focus_force()
            
            # Main confirmation dialog
            result = messagebox.askyesnocancel(
                "🎯 Character Learning",
                f"🎮 Bubble ID{obj_id} (look for green overlay 'id{obj_id}' on screen)\n\n"
                f"🤖 AI predicted: '{predicted_label}' (confidence: {confidence:.2f})\n\n"
                f"❓ Is this prediction correct?\n\n"
                f"✅ YES = Correct (will remember forever)\n"
                f"❌ NO = Wrong (I'll ask what's correct)\n"
                f"⏭️ CANCEL = Skip this time",
                icon='question'
            )
            
            if result is True:  # YES - Correct
                print(f"[POPUP] ✅ ID{obj_id}: User confirmed '{predicted_label}' - will remember forever")
                root.destroy()
                return predicted_label
                
            elif result is False:  # NO - Wrong, ask for correction
                # Game-specific valid characters (no i, o, x, 0)
                valid_chars = self.get_valid_game_characters()
                
                correction_prompt = (
                    f"🔧 What's the correct character for bubble ID{obj_id}?\n\n"
                    f"🤖 AI predicted '{predicted_label}' but it's wrong\n"
                    f"📝 Valid characters: 1-9, a-z (NO i,o,x,0)\n"
                    f"🎮 Game keys: {valid_chars}\n\n"
                    f"Type the correct character:"
                )
                
                correct_char = simpledialog.askstring(
                    "🔧 Correct Character",
                    correction_prompt,
                    parent=root
                )
                
                root.destroy()
                
                if correct_char and len(correct_char) == 1 and self._is_valid_game_character(correct_char):
                    correct_char = correct_char.lower()
                    print(f"[POPUP] ❌ ID{obj_id}: User corrected '{predicted_label}' → '{correct_char}'")
                    return correct_char
                else:
                    print(f"[POPUP] ⚠️ ID{obj_id}: Invalid correction '{correct_char}', not in game keyset")
                    print(f"[POPUP] Valid keys: {valid_chars}")
                    return predicted_label
                    
            else:  # CANCEL - Skip
                print(f"[POPUP] ⏭️ ID{obj_id}: User skipped confirmation")
                root.destroy()
                return predicted_label
            
        except Exception as e:
            print(f"[POPUP] ❌ Error with confirmation for ID{obj_id}: {e}")
            try:
                root.destroy()
            except:
                pass
            return predicted_label
    
    def _record_positive_confirmation(self, label, crop_img, confidence, bubble_id=None):
        """Record positive confirmation (YES response)"""
        try:
            timestamp = time.time()
            confirmation_id = f"confirm_{timestamp:.3f}"
            
            # Save crop image
            crop_path = self.data_dir / f"{confirmation_id}_{label}_YES.png"
            cv2.imwrite(str(crop_path), crop_img)
            
            confirmation_data = {
                "id": confirmation_id,
                "timestamp": timestamp,
                "bubble_id": bubble_id,
                "confirmed_label": label,
                "confidence": confidence,
                "crop_file": str(crop_path.name),
                "status": "confirmed_positive",
                "valid_character": self._is_valid_game_character(label)
            }
            
            self.confirmations["confirmations"].append(confirmation_data)
            
            # Update stats
            if "stats" not in self.confirmations:
                self.confirmations["stats"] = {}
            
            label_key = f"confirmed_{label}"
            self.confirmations["stats"][label_key] = self.confirmations["stats"].get(label_key, 0) + 1
            
            self._save_confirmations()
            print(f"[RECORD] 💾 Recorded positive confirmation: {label} (confidence: {confidence:.2f})")
            
        except Exception as e:
            print(f"[RECORD] Error recording confirmation: {e}")
    
    def _record_correction(self, wrong_label, correct_label, crop_img, bubble_id=None):
        """Record correction with feature learning"""
        try:
            timestamp = time.time()
            correction_id = f"correction_{timestamp:.3f}"
            
            # Save crop image
            crop_path = self.data_dir / f"{correction_id}_{wrong_label}_to_{correct_label}.png"
            cv2.imwrite(str(crop_path), crop_img)
            
            # Extract features for learning
            features = self._extract_features(crop_img)
            
            correction_data = {
                "id": correction_id,
                "timestamp": timestamp,
                "bubble_id": bubble_id,
                "wrong_prediction": wrong_label,
                "correct_label": correct_label,
                "crop_file": str(crop_path.name),
                "features": features,
                "status": "correction",
                "valid_character": self._is_valid_game_character(correct_label)
            }
            
            self.corrections["corrections"].append(correction_data)
            
            # Update stats
            if "stats" not in self.corrections:
                self.corrections["stats"] = {}
            
            pair_key = f"{wrong_label}→{correct_label}"
            self.corrections["stats"][pair_key] = self.corrections["stats"].get(pair_key, 0) + 1
            
            # Update adaptive thresholds with extracted features
            self._update_adaptive_thresholds(wrong_label, correct_label, features)
            
            self._save_corrections()
            self._save_thresholds()
            
            print(f"[RECORD] 💾 Recorded correction: {wrong_label} → {correct_label}")
            
        except Exception as e:
            print(f"[RECORD] Error recording correction: {e}")
    
    def _update_adaptive_thresholds(self, wrong_label, correct_label, features=None):
        """Update adaptive thresholds for ML improvement"""
        try:
            pair_key = f"{wrong_label}_vs_{correct_label}"
            
            if pair_key not in self.adaptive_thresholds:
                self.adaptive_thresholds[pair_key] = {
                    "corrections_count": 0,
                    "learned_rules": {}
                }
            
            entry = self.adaptive_thresholds[pair_key]
            entry["corrections_count"] += 1
            
            # Update learned rules with features if available
            if features:
                for feat_name in ["ar", "stems", "top_density", "bottom_density", "center_density"]:
                    if feat_name in features:
                        rule_name = f"{feat_name}_threshold"
                        feat_val = features[feat_name]
                        
                        if rule_name not in entry["learned_rules"]:
                            entry["learned_rules"][rule_name] = feat_val
                        else:
                            # Exponential moving average update
                            entry["learned_rules"][rule_name] = (
                                entry["learned_rules"][rule_name] * 0.7 + feat_val * 0.3
                            )
            
            print(f"[THRESHOLD] Updated adaptive rules for {pair_key} (corrections: {entry['corrections_count']})")
            
        except Exception as e:
            print(f"[THRESHOLD] Error updating thresholds: {e}")
    
    def get_adaptive_override(self, predicted_label, features):
        """Apply learned rules to override predictions based on corrections"""
        if not features:
            return predicted_label
        
        # Check if prediction is valid game character
        if not self._is_valid_game_character(predicted_label):
            print(f"[ADAPTIVE] ⚠️ Invalid game character '{predicted_label}' - no override")
            return predicted_label
            
        # Common confusions for game keyset (no i,o,x,0)
        common_confusions = [
            ("4", "q"), ("q", "4"), ("5", "s"), ("5", "w"), ("5", "f"),
            ("j", "t"), ("t", "j"), ("p", "h"), ("h", "p"),
            ("v", "h"), ("h", "v"), ("k", "r"), ("r", "k"),
            ("1", "r"), ("r", "1"), ("1", "l"), ("l", "1"),
            ("6", "g"), ("g", "6"), ("2", "z"), ("z", "2"),
            ("3", "e"), ("e", "3"), ("8", "b"), ("b", "8"),
            ("9", "g"), ("g", "9"), ("7", "t"), ("t", "7"),
            ("d", "p"), ("p", "d"), ("n", "m"), ("m", "n"),
            ("u", "v"), ("v", "u"), ("c", "e"), ("e", "c"),
            ("a", "e"), ("e", "a"), ("s", "5"), ("w", "v"),
            ("f", "t"), ("t", "f"), ("y", "v"), ("v", "y")
        ]
        
        # Check if we have learned rules for this prediction
        for label_a, label_b in common_confusions:
            if predicted_label == label_a and self._is_valid_game_character(label_b):
                pair_key = f"{label_a}_vs_{label_b}"
                if pair_key in self.adaptive_thresholds:
                    rules = self.adaptive_thresholds[pair_key].get("learned_rules", {})
                    if self._should_override_to(features, rules, label_b):
                        print(f"[ADAPTIVE] 🔄 Override: {label_a} → {label_b} based on learned rules")
                        return label_b
        
        # No override needed
        return predicted_label
    
    def _should_override_to(self, features, learned_rules, target_label):
        """Check if features match learned rules for override"""
        if not learned_rules:
            return False
            
        score = 0
        total = 0
        
        # Check feature thresholds
        for rule_name, threshold in learned_rules.items():
            if rule_name.endswith("_threshold"):
                feat_name = rule_name.replace("_threshold", "")
                if feat_name in features:
                    feat_val = features[feat_name]
                    diff = abs(feat_val - threshold)
                    if diff < 0.2:  # 20% tolerance
                        score += 1
                    total += 1
        
        # Return True if 60% of rules match
        return total > 0 and (score / total) >= 0.6
    
    def _extract_features(self, crop_img):
        """Extract features for adaptive override (compatibility with existing code)"""
        try:
            if len(crop_img.shape) == 3:
                gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop_img
            
            h, w = gray.shape
            if h == 0 or w == 0:
                return {}
            
            features = {
                "ar": w / h,
                "stems": 0,  # Placeholder - could enhance with actual stem detection
                "top_density": np.mean(gray[:h//3]) / 255.0,
                "bottom_density": np.mean(gray[2*h//3:]) / 255.0,
                "center_density": np.mean(gray[h//4:3*h//4, w//4:3*w//4]) / 255.0 if h > 4 and w > 4 else 0.0,
                "small_top_dot": False  # Placeholder - could enhance with dot detection
            }
            
            return features
        except Exception as e:
            print(f"[FEATURES] Error extracting features for override: {e}")
            return {}
    
    def get_learned_stats(self):
        """Get comprehensive learning statistics"""
        learned_chars = list(set(self.learned_characters.values()))
        total_confirmations = len(self.confirmations.get("confirmations", []))
        total_corrections = len(self.corrections.get("corrections", []))
        
        # Filter to valid game characters only
        valid_learned_chars = [c for c in learned_chars if self._is_valid_game_character(c)]
        invalid_learned_chars = [c for c in learned_chars if not self._is_valid_game_character(c)]
        
        # Calculate coverage
        total_possible = len(self.get_valid_game_characters())
        coverage_percent = (len(valid_learned_chars) / total_possible) * 100 if total_possible > 0 else 0
        
        return {
            "total_learned_patterns": len(self.learned_characters),
            "unique_characters": sorted(learned_chars),
            "valid_game_characters": sorted(valid_learned_chars),
            "invalid_characters": sorted(invalid_learned_chars),
            "character_count": len(learned_chars),
            "valid_character_count": len(valid_learned_chars),
            "coverage_percent": coverage_percent,
            "total_confirmations": total_confirmations,
            "total_corrections": total_corrections,
            "correction_stats": self.corrections.get("stats", {}),
            "confirmation_stats": self.confirmations.get("stats", {}),
            "confidence_threshold": self.confidence_threshold,
            "game_keyset": self.get_valid_game_characters(),
            "excluded_chars": self.get_excluded_characters()
        }
    
    def reset_learning(self):
        """Reset all learned characters (for testing/retraining)"""
        patterns_count = len(self.learned_characters)
        confirmations_count = len(self.confirmations.get("confirmations", []))
        corrections_count = len(self.corrections.get("corrections", []))
        
        # Clear all learning data
        self.learned_characters.clear()
        self.confirmations = {"confirmations": [], "stats": {}}
        self.corrections = {"corrections": [], "stats": {}}
        self.adaptive_thresholds = {}
        
        # Save cleared state
        self._save_learned_characters()
        self._save_confirmations()
        self._save_corrections()
        self._save_thresholds()
        
        print(f"[RESET] 🗑️ Reset complete:")
        print(f"  - Cleared {patterns_count} learned patterns")
        print(f"  - Cleared {confirmations_count} confirmations")
        print(f"  - Cleared {corrections_count} corrections")
        print(f"[RESET] ⚡ Ready for fresh learning!")
        
        return {
            "patterns_cleared": patterns_count,
            "confirmations_cleared": confirmations_count,
            "corrections_cleared": corrections_count
        }
    
    def update_confidence_threshold(self, new_threshold):
        """Update confidence threshold for auto-learning"""
        old_threshold = self.confidence_threshold
        self.confidence_threshold = max(0.0, min(1.0, new_threshold))  # Clamp to [0,1]
        
        print(f"[CONFIG] 🎛️ Confidence threshold: {old_threshold:.2f} → {self.confidence_threshold:.2f}")
        
        if self.confidence_threshold > old_threshold:
            print(f"[CONFIG] ⬆️ Higher threshold = More popups, higher quality learning")
        else:
            print(f"[CONFIG] ⬇️ Lower threshold = Fewer popups, faster auto-learning")
    
    def export_learning_data(self, export_path=None):
        """Export all learning data for backup or sharing"""
        if export_path is None:
            export_path = self.data_dir / f"learning_export_{int(time.time())}.json"
        
        try:
            export_data = {
                "export_timestamp": time.time(),
                "export_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "learned_characters": self.learned_characters,
                "corrections": self.corrections,
                "confirmations": self.confirmations,
                "adaptive_thresholds": self.adaptive_thresholds,
                "config": {
                    "confidence_threshold": self.confidence_threshold,
                    "game_keyset": self.get_valid_game_characters(),
                    "excluded_chars": self.get_excluded_characters()
                },
                "stats": self.get_learned_stats()
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"[EXPORT] 📦 Learning data exported to: {export_path}")
            return str(export_path)
            
        except Exception as e:
            print(f"[EXPORT] Error exporting learning data: {e}")
            return None
    
    def import_learning_data(self, import_path):
        """Import learning data from backup file"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Backup current data
            backup_path = self.data_dir / f"backup_before_import_{int(time.time())}.json"
            self.export_learning_data(backup_path)
            
            # Import data
            self.learned_characters = import_data.get("learned_characters", {})
            self.corrections = import_data.get("corrections", {"corrections": [], "stats": {}})
            self.confirmations = import_data.get("confirmations", {"confirmations": [], "stats": {}})
            self.adaptive_thresholds = import_data.get("adaptive_thresholds", {})
            
            # Update config if available
            config = import_data.get("config", {})
            if "confidence_threshold" in config:
                self.confidence_threshold = config["confidence_threshold"]
            
            # Save imported data
            self._save_learned_characters()
            self._save_corrections()
            self._save_confirmations()
            self._save_thresholds()
            
            stats = self.get_learned_stats()
            print(f"[IMPORT] 📥 Learning data imported successfully!")
            print(f"[IMPORT] 📊 Imported {stats['total_learned_patterns']} patterns, {stats['valid_character_count']} valid chars")
            
            return True
            
        except Exception as e:
            print(f"[IMPORT] Error importing learning data: {e}")
            return False
    def _popup_worker(self):
        """Background thread for popups - no FPS lag"""
        while True:
            try:
                if not self.pending_confirmations.empty():
                    data = self.pending_confirmations.get(timeout=0.1)
                    self._handle_popup_async(data)
                else:
                    time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[POPUP] Error: {e}")
                time.sleep(0.5)
    
    def _handle_popup_async(self, data):
        """Handle popup in background - won't affect game FPS"""
        try:
            obj_id = data['obj_id']
            predicted_label = data['predicted_label']
            confidence = data['confidence']
            crop_img = data['crop_img']
            content_hash = data['content_hash']
            
            self.popup_active = True
            print(f"[POPUP] 📋 Showing popup for ID{obj_id}: '{predicted_label}' (background thread)")
            
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            root.lift()
            root.focus_force()
            
            # Simple dialog với REFRESH button
            dialog = tk.Toplevel(root)
            dialog.title("Character Learning")
            dialog.attributes('-topmost', True)
            dialog.grab_set()
            
            # Center dialog
            dialog.geometry("400x200")
            x = (dialog.winfo_screenwidth() // 2) - 200
            y = (dialog.winfo_screenheight() // 2) - 100
            dialog.geometry(f"400x200+{x}+{y}")
            
            # Message
            message = (
                f"Bubble ID{obj_id}\n"
                f"AI predicted: '{predicted_label}' (confidence: {confidence:.2f})\n\n"
                f"Is this correct?"
            )
            
            tk.Label(dialog, text=message, font=('Arial', 10)).pack(pady=20)
            
            result = {"value": None}
            
            def on_yes():
                result["value"] = "YES"
                dialog.destroy()
            
            def on_no():
                result["value"] = "NO"
                dialog.destroy()  # ← FIX: DESTROY DIALOG!
            
            def on_refresh():
                result["value"] = "REFRESH"
                dialog.destroy()
            
            def on_skip():
                result["value"] = "SKIP"
                dialog.destroy()
            
            # Buttons frame
            btn_frame = tk.Frame(dialog)
            btn_frame.pack(pady=10)
            
            tk.Button(btn_frame, text="YES", command=on_yes, width=8).pack(side='left', padx=5)
            tk.Button(btn_frame, text="NO", command=on_no, width=8).pack(side='left', padx=5)
            tk.Button(btn_frame, text="REFRESH", command=on_refresh, width=8).pack(side='left', padx=5)
            tk.Button(btn_frame, text="SKIP", command=on_skip, width=8).pack(side='left', padx=5)
            
            dialog.wait_window()
            
            # Process result
            if result["value"] == "YES":  # Correct
                self.learned_characters[content_hash] = predicted_label
                self._save_learned_characters()
                self._record_positive_confirmation(predicted_label, crop_img, confidence, f"id{obj_id}")
                self.confirmation_results[content_hash] = predicted_label
                print(f"[POPUP] ✅ ID{obj_id}: Confirmed '{predicted_label}'")
                
            elif result["value"] == "NO":  # Wrong
                print(f"[POPUP] ❌ ID{obj_id}: Wrong prediction, asking for correction...")
                
                # Ask for correction using simpledialog
                valid_chars = self.get_valid_game_characters()
                correction = simpledialog.askstring(
                    "Correct Character",
                    f"Bubble ID{obj_id} - AI said '{predicted_label}' but it's wrong\n\n"
                    f"What's the correct character?\n"
                    f"Valid: {valid_chars}\n\n"
                    f"Type correct character:",
                    parent=root
                )
                
                if correction and len(correction) == 1 and self._is_valid_game_character(correction):
                    correction = correction.lower()
                    self.learned_characters[content_hash] = correction
                    self._save_learned_characters()
                    self._record_correction(predicted_label, correction, crop_img, f"id{obj_id}")
                    self.confirmation_results[content_hash] = correction  # ← FIX
                    print(f"[POPUP] ❌ ID{obj_id}: Corrected to '{correction}'")
                else:
                    print(f"[POPUP] ⚠️ ID{obj_id}: Invalid correction '{correction}', keeping original")
                    self.confirmation_results[content_hash] = predicted_label  # ← FIX
                    
            elif result["value"] == "REFRESH":  # Ask again
                print(f"[POPUP] 🔄 ID{obj_id}: REFRESH - Will ask again in 5 seconds")
                retry_time = time.time() + 5.0
                self.ask_again_requests.append({
                    'time': retry_time,
                    'data': data,
                    'content_hash': content_hash  # ← THÊM DÒNG NÀY
                })
                self.confirmation_results[content_hash] = predicted_label
                
            else:  # SKIP
                print(f"[POPUP] ⏭️ ID{obj_id}: SKIP")
                self.confirmation_results[content_hash] = predicted_label
            
            root.destroy()
            
        except Exception as e:
            print(f"[POPUP] Error: {e}")
            self.confirmation_results[content_hash] = predicted_label
        finally:
            self.popup_active = False
    
    def should_confirm_non_blocking(self, predicted_label, confidence, margin, features, obj_id=None, box=None, crop=None):
        """Non-blocking confirmation - no FPS impact"""
        if self.bypass_learning:
            return predicted_label
        if crop is None:
            return predicted_label
        
        if not self._is_valid_game_character(predicted_label):
            return predicted_label
        
        content_hash = self._get_stable_content_hash(crop, predicted_label)
        
        # Check if already learned
        if content_hash in self.learned_characters:
            learned_label = self.learned_characters[content_hash]
            if learned_label == predicted_label:
                return learned_label
        
        # Check if we have cached result
        if content_hash in self.confirmation_results:
            result = self.confirmation_results[content_hash]
            print(f"[LEARN] 📥 Using cached result for hash {content_hash[:8]}: '{result}'")
            return result
        
        # Auto-learn high confidence
        if confidence >= self.confidence_threshold:
            self.learned_characters[content_hash] = predicted_label
            self._save_learned_characters()
            print(f"[LEARN] 🚀 Auto-learned '{predicted_label}' ({confidence:.2f})")
            return predicted_label
        
        # Queue for background popup (no FPS lag)
        if not self.popup_active:
            try:
                self.pending_confirmations.put_nowait({
                    'obj_id': obj_id,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'crop_img': crop.copy() if crop is not None else None,
                    'content_hash': content_hash
                })
                print(f"[LEARN] 📤 Queued popup for ID{obj_id} (no FPS impact)")
            except queue.Full:
                pass
        
        return predicted_label
    
    def check_retry_requests(self):
        """Check if any popups need to be re-shown"""
        current_time = time.time()
        ready_to_retry = []
        still_waiting = []
        
        for request in self.ask_again_requests:
            if current_time >= request['time']:
                # Check if this content_hash is still pending (not answered yet)
                content_hash = request.get('content_hash')
                if content_hash and content_hash not in self.confirmation_results:
                    ready_to_retry.append(request)
                    print(f"[RETRY] 🔄 Hash {content_hash[:8]} ready for retry")
                else:
                    print(f"[RETRY] ⏭️ Hash {content_hash[:8]} already answered, skipping retry")
            else:
                still_waiting.append(request)
        
        self.ask_again_requests = still_waiting
        
        # Re-queue ready items
        for request in ready_to_retry:
            if not self.popup_active:
                try:
                    self.pending_confirmations.put_nowait(request['data'])
                    print(f"[RETRY] 🔄 Re-queued popup for hash {request['content_hash'][:8]}")
                except queue.Full:
                    # If queue full, try again later
                    request['time'] = current_time + 2.0
                    self.ask_again_requests.append(request)
    def _ask_correction_simple(self, parent, obj_id, predicted_label):
        """Simple correction dialog"""
        valid_chars = self.get_valid_game_characters()
        
        return simpledialog.askstring(
            "🔧 Correct Character",
            f"🔧 Correct character for ID{obj_id}?\n"
            f"🤖 AI said '{predicted_label}' but it's wrong\n"
            f"🎮 Valid: {valid_chars}\n\n"
            f"Type correct character:",
            parent=parent
        )                
# ============ Glyph helpers ============
def _glyph_bw(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return None
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if (bw == 255).mean() < 0.3:
        bw = 255 - bw
    return bw

def _glyph_features(img_bgr):
    bw = _glyph_bw(img_bgr)
    if bw is None:
        return None
    H, W = bw.shape[:2]
    if H == 0 or W == 0:
        return None
    on = (bw > 0).astype(np.uint8)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(on, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    order = np.argsort(areas)[::-1]
    small_top_dot = False
    if num >= 3:
        main_idx = order[1] if order[0] == 0 else order[0]
        main_area = max(1.0, areas[main_idx])
        for idx in order:
            if idx in (0, main_idx): continue
            a = areas[idx]
            x,y,w,h = stats[idx,0], stats[idx,1], stats[idx,2], stats[idx,3]
            cy = centroids[idx][1] / max(1.0, H)
            arat = a / main_area
            fill = float(a) / max(1, w*h)
            ar_box = float(w) / max(1, h)
            dot_ok = (0.5 <= ar_box <= 1.8) and (fill >= 0.35)
            if 0.01 <= arat <= 0.18 and cy <= 0.40 and dot_ok:
                small_top_dot = True
                break

    top_band = max(1, int(0.25*H))
    bot_band = max(1, int(0.18*H))
    top_density = float(on[: top_band, :].mean())
    bottom_density = float(on[H-bot_band:, :].mean())
    center_cols = slice(int(0.35*W), int(0.65*W))
    center_density = float(on[:, center_cols].mean()) if (int(0.65*W) > int(0.35*W)) else float(on.mean())

    col_sum = on.sum(axis=0).astype(np.float32) / max(1, H)
    cthr = max(0.35, 0.5*float(col_sum.max()))
    stems, on_flag = 0, False
    for v in col_sum:
        if (not on_flag) and v >= cthr:
            on_flag = True; stems += 1
        elif on_flag and v < cthr*0.6:
            on_flag = False

    ys, xs = np.where(on > 0)
    if xs.size == 0 or ys.size == 0:
        ar = 1.0
    else:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        ar = float((x2 - x1 + 1)) / max(1.0, float(y2 - y1 + 1))

    return {
        "H": H, "W": W,
        "small_top_dot": small_top_dot,
        "bottom_density": bottom_density,
        "top_density": top_density,
        "center_density": center_density,
        "stems": stems,
        "ar": ar,
    }

def count_glyph_holes(img_bgr):
    try:
        bw = _glyph_bw(img_bgr)
        if bw is None: return 0
        cnts, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None: return 0
        holes = 0
        for i in range(len(cnts)):
            if hier[0][i][3] != -1:
                holes += 1
        return holes
    except Exception:
        return 0

def refine_glyph_roi(crop_bgr):
    try:
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr
        g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3,3), 0)
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 19, 2)
        if (bw == 255).mean() > 0.6: bw = 255 - bw
        kernel = np.ones((3,3), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return crop_bgr
        cnt = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h < 0.05*bw.size:
            return crop_bgr
        pad = 2
        x1 = max(0, x-pad); y1 = max(0, y-pad)
        x2 = min(crop_bgr.shape[1], x+w+pad); y2 = min(crop_bgr.shape[0], y+h+pad)
        tight = crop_bgr[y1:y2, x1:x2]
        return tight if tight.size > 0 else crop_bgr
    except Exception:
        return crop_bgr

# ============ Isolator ============
class GlyphIsolator:
    def __init__(self, cfg):
        cg = cfg.get("ui_color_gate", {})
        lg = cfg.get("letter_gate", {})
        self.h_low = int(cg.get("h_low", 80))
        self.h_high = int(cg.get("h_high", 110))
        self.s_low = int(cg.get("s_low", 50))
        self.v_low = int(cg.get("v_low", 30))
        self.white_s_max = int(lg.get("white_s_max", 60))
        self.white_v_min = int(lg.get("white_v_min", 185))
        self.erode_px = int(lg.get("erode_px", 1))
        self.min_ratio = float(lg.get("min_ratio", 0.004))

    def __call__(self, crop_bgr):
        try:
            if crop_bgr is None or crop_bgr.size == 0:
                return None
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(hsv)
            mask_bubble = (h >= self.h_low) & (h <= self.h_high) & (s >= self.s_low) & (v >= self.v_low)
            mb = (mask_bubble.astype(np.uint8) * 255)
            if self.erode_px > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.erode_px*2+1, self.erode_px*2+1))
                mb = cv2.erode(mb, k, iterations=1)
            mask_white = (s <= self.white_s_max) & (v >= self.white_v_min)
            mw = (mask_white.astype(np.uint8) * 255)
            letter = cv2.bitwise_and(mw, mb)

            cnts, _ = cv2.findContours(letter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None
            H, W = letter.shape[:2]
            areas = [cv2.contourArea(c) for c in cnts]
            main_i = int(np.argmax(areas))
            x,y,w,h = cv2.boundingRect(cnts[main_i])
            cx_main = x + w*0.5
            cy_main = y + h*0.5
            main_area = max(1.0, areas[main_i])

            keep = [main_i]
            for i,c in enumerate(cnts):
                if i == main_i: continue
                a = areas[i]
                if a <= 0.22*main_area and a >= 0.01*main_area:
                    rx,ry,rw,rh = cv2.boundingRect(c)
                    ar_box = float(max(rw,1))/float(max(rh,1))
                    fill = float(a)/max(1, rw*rh)
                    M = cv2.moments(c)
                    cx = (M["m10"]/M["m00"]) if M["m00"] else (rx+rw/2)
                    cy = (M["m01"]/M["m00"]) if M["m00"] else (ry+rh/2)
                    above = cy <= (y - 0.05*H) or cy <= (cy_main - 0.35*h)
                    aligned = abs(cx - cx_main) <= 0.5*w
                    dot_ok = (0.5 <= ar_box <= 1.8) and (fill >= 0.35)
                    if above and aligned and dot_ok:
                        keep.append(i)

            mask = np.zeros_like(letter)
            cv2.drawContours(mask, [cnts[i] for i in keep], -1, 255, -1)
            kept_area = float((mask>0).sum())
            if kept_area < self.min_ratio * (H*W):
                return None
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        except Exception:
            return None

# ============ Shape override ============
def shape_override(label, glyph_bgr, cfg, adaptive_learner=None):
    if glyph_bgr is None or glyph_bgr.size == 0:
        return label
    f = _glyph_features(glyph_bgr)
    if f is None:
        return label

    # ← THÊM DEBUG INFO CHO WIDE CHARACTERS
    if label in ['m', 'w', 'u', 'v', 'h']:
        print(f"[SHAPE-ANALYSIS] '{label}' features: ar={f['ar']:.2f} stems={f['stems']} cdens={f['center_density']:.2f} top={f['top_density']:.2f} bot={f['bottom_density']:.2f}")

    # Áp dụng adaptive learning trước
    if adaptive_learner:
        adaptive_result = adaptive_learner.get_adaptive_override(label, f)
        if adaptive_result != label:
            return adaptive_result

    top, bottom = f["top_density"], f["bottom_density"]
    cdens, stems, ar = f["center_density"], f["stems"], f["ar"]
    ascender = (top >= 0.070)
    descender = (bottom >= 0.10)
    has_top_dot = f["small_top_dot"]

    try:
        holes = count_glyph_holes(glyph_bgr)
        if label in ("q", "d"):
            if holes >= 1 and not descender:
                label = "d"
            elif holes == 0 and (descender or has_top_dot):
                label = "q"
    except Exception:
        holes = 0

    # ← CRITICAL FIX: IMPROVED m/w/u/h DETECTION
    if label in ("u", "m", "h", "w"):
        # Calculate features for wide characters
        very_wide = ar >= 1.4
        wide = ar >= 1.2
        narrow = ar <= 1.0
        many_stems = stems >= 3
        few_stems = stems <= 2
        high_density = cdens >= 0.45
        
        # Define character patterns more precisely
        w_like = very_wide and many_stems and high_density  # 'w' is very wide with many stems
        m_like = wide and many_stems and (cdens >= 0.40) and (not ascender) and stems >= 1 # 'm' is wide with stems, no ascender
        h_like = ascender and few_stems and (ar <= 1.10) and ((bottom - top) <= 0.08)  # 'h' has ascender, narrow
        u_like = (not ascender) and few_stems and ((bottom - top) >= 0.06) and (ar >= 0.85) and (ar <= 1.25)  # 'u' shape
        
        print(f"[SHAPE-DETECT] '{label}': w_like={w_like}, m_like={m_like}, h_like={h_like}, u_like={u_like}")
        print(f"[SHAPE-DETECT] Features: very_wide={very_wide}, wide={wide}, many_stems={many_stems}, ascender={ascender}")
        
        # ← CRITICAL FIX: PRESERVE ORIGINAL 'u' UNLESS VERY CERTAIN
        if label == "u":
            # Only override 'u' if VERY strong h signal
            if h_like and ascender and (top >= 0.85) and narrow and stems == 1:
                print(f"[SHAPE-FIX] 'u' → 'h' (VERY strong h signal)")
                return "h"
            else:
                print(f"[SHAPE-FIX] 'u' → 'u' (PRESERVED - weak override signal)")
                return "u"  # ← KEEP ORIGINAL 'u'
        
        if label == "m":
            # Only override 'm' in very specific cases
            if w_like:
                print(f"[SHAPE-FIX] 'm' → 'w' (very wide + many stems)")
                return "w"
            else:
                print(f"[SHAPE-FIX] 'm' → 'm' (PRESERVED - avoiding wrong override)")
                return "m"  # ← KEEP ORIGINAL 'm'

        # Apply detection in priority order for other characters
        if w_like:
            print(f"[SHAPE-FIX] '{label}' → 'w' (very wide + many stems)")
            return "w"
        elif label != "m" and m_like and not h_like:  # ← Don't override FROM m
            print(f"[SHAPE-FIX] '{label}' → 'm' (wide + stems, no ascender)")
            return "m"
        elif label != "u" and h_like and not m_like and not u_like:  # ← Don't override FROM u
            print(f"[SHAPE-FIX] '{label}' → 'h' (ascender + narrow)")
            return "h"
        elif label != "u" and u_like and not h_like and not m_like:  # ← Don't override FROM u
            print(f"[SHAPE-FIX] '{label}' → 'u' (u-shape)")
            return "u"
        else:
            # Conservative: keep original if ambiguous
            print(f"[SHAPE-FIX] '{label}' → '{label}' (ambiguous, keeping original)")
            return label

    # ← KEEP EXISTING v/h LOGIC BUT MORE CONSERVATIVE
    if label in ("v", "h"):
        v_like = (not ascender) and (bottom >= 0.07) and (ar >= 1.15) and (stems <= 2) and (cdens <= 0.52)
        h_like2 = ascender and (not descender) and (stems >= 2) and (ar <= 1.15)
        
        if v_like and not h_like2: 
            label = "v"
            print(f"[SHAPE] v/h → 'v'")
        elif h_like2 and not v_like: 
            label = "h"
            print(f"[SHAPE] v/h → 'h'")

    if label in ("p", "h"):
        p_like = descender and ((bottom - top) >= 0.06) and (stems >= 1) and (cdens >= 0.42)
        h_like3 = ascender and (not descender) and (stems >= 2)
        if p_like and not h_like3: label = "p"
        elif h_like3 and not p_like: label = "h"

    if label in ("f", "j", "t"):
        if has_top_dot:
            label = "j"
            print(f"[SHAPE-FIX] Detected 'j' with top dot")
        else:
            # ← MORE CONSERVATIVE j/t detection
            j_like = descender and (stems <= 2) and (cdens <= 0.58) and (not ascender)
            if j_like and ar <= 0.85:  # ← ADD AR CHECK
                label = "j"
                print(f"[SHAPE-FIX] 'j' pattern detected")

    if label in ("n", "h", "m"):
        n_like = (not ascender) and (ar >= 0.8) and (ar <= 1.1) and (stems >= 2)
        if n_like and label != "h":
            print(f"[SHAPE-FIX] 'n' pattern preserved")
            return "n"

    if label in ("t", "7"):
        seven_like = (ar >= 1.18) and (cdens <= 0.45) and ((top - bottom) >= 0.15)
        if seven_like:
            label = "7"
        else:
            if (cdens >= 0.54 and ar <= 1.10) or (stems >= 2 and ar <= 1.16):
                label = "t"

    if label in ("k", "r"):
        k_like = (stems >= 2) or (cdens >= 0.50 and ar >= 0.85)
        r_like = (stems <= 2) and (ar <= 1.05) and (top >= bottom - 0.02) and (cdens <= 0.48)
        if k_like and not r_like: label = "k"
        elif r_like and not k_like: label = "r"

    if label in ("1", "r"):
        one_like = (stems == 1) and (ar <= 0.80) and (cdens <= 0.42) and ((bottom - top) <= 0.08)
        r_like2 = (stems <= 2) and (ar <= 1.05) and (cdens >= 0.40)
        if one_like and not r_like2:
            label = "1"
        elif r_like2 and not one_like:
            label = "r"

    if label in ("1", "r", "l"):
        # Enhanced 1/r/l detection - MORE CONSERVATIVE
        stems_strong = stems >= 2
        very_narrow = ar <= 0.65  # Very narrow = likely "1"
        tall_thin = ar <= 0.80 and (bottom - top) <= 0.08  # Classic "1" shape
        wide_enough = ar >= 0.90  # Wide enough for "r"
        
        # ← CRITICAL FIX: PRESERVE ORIGINAL 'l' CLASSIFICATION
        if label == "l":
            # Only override 'l' in very specific cases
            if very_narrow and stems == 1 and (bottom - top) <= 0.05:
                label = "1"
                print(f"[SHAPE] 'l' → '1' (very narrow + single stem)")
            else:
                print(f"[SHAPE] 'l' → 'l' (PRESERVED - avoiding wrong override)")
                return "l"  # ← KEEP ORIGINAL 'l'
        
        # Apply detection for other characters
        elif label in ("1", "r"):
            if very_narrow or tall_thin:
                label = "1"
                print(f"[SHAPE] Strong '1' signal: ar={ar:.2f} stems={stems} narrow={very_narrow}")
            elif wide_enough and stems_strong:
                label = "r" 
                print(f"[SHAPE] Strong 'r' signal: ar={ar:.2f} stems={stems}")

    if label in ("4", "q"):
        four_like = (holes == 0) and (not descender) and ((stems >= 2) or (ar >= 1.00)) and (cdens >= 0.35) and (top >= 0.03)
        q_like = (holes >= 1) or (descender and cdens >= 0.45)
        if four_like and not q_like:
            label = "4"
        elif q_like and not four_like:
            label = "q"

    if label in ("f", "j", "t"):
        if has_top_dot:
            label = "j"
            print(f"[SHAPE-FIX] Detected 'j' with top dot")
        else:
            # ← ENHANCED f/j/t detection
            f_like = descender and ascender and (cdens >= 0.50) and (stems >= 2) and (ar <= 1.10)
            j_like = descender and (stems <= 2) and (cdens <= 0.58) and (not ascender) and (ar <= 0.85)
            t_like = (cdens >= 0.54 and ar <= 1.10) or (stems >= 2 and ar <= 1.16)
            
            if f_like and not j_like and not t_like:
                label = "f"
                print(f"[SHAPE-FIX] Strong 'f' pattern detected")
            elif j_like and not f_like and not t_like:
                label = "j"
                print(f"[SHAPE-FIX] Strong 'j' pattern detected")
            elif t_like and not f_like and not j_like:
                label = "t"
                print(f"[SHAPE-FIX] Strong 't' pattern detected")

    if label in ("4", "t"):
        # Enhanced 4/t detection based on aspect ratio and stems
        if ar >= 1.0 and stems >= 2 and cdens >= 0.40:
            label = "4"  # Wide with multiple stems = likely "4"
        elif ar <= 0.90 and cdens >= 0.54:
            label = "t"  # Narrower with high center density = likely "t"
        
        print(f"[SHAPE] 4/t detection: ar={ar:.2f} stems={stems} cdens={cdens:.2f} -> '{label}'")

    # ← ENHANCED 5/w DETECTION
    if label == "5":
        w_like = (ar >= 1.35) and (stems >= 3 or cdens >= 0.48)  # ← STRICTER for 'w'
        f_like2 = descender and ascender and (cdens >= 0.50) and (stems >= 2) and (ar <= 1.10)
        if w_like: 
            print(f"[SHAPE] '5' → 'w' (wide pattern: ar={ar:.2f}, stems={stems})")
            label = "w"
        elif f_like2: 
            label = "f"

    # ← FINAL w/u/v CHECK FROM CONFIG
    go = cfg.get("glyph_override", {})
    if go.get("enable", True):
        w_min_ar = float(go.get("w_min_ar", 1.30))
        u_max_ar = float(go.get("u_max_ar", 1.12))
        
        # More aggressive 'w' detection
        if label in ("u", "v", "m") and ar >= w_min_ar:
            print(f"[SHAPE-CONFIG] '{label}' → 'w' (AR {ar:.2f} >= {w_min_ar})")
            label = "w"
        elif label == "w" and ar <= u_max_ar:
            print(f"[SHAPE-CONFIG] 'w' → 'u' (AR {ar:.2f} <= {u_max_ar})")
            label = "u"

    return label

def seed_veto(label, glyph_bgr):
    f = _glyph_features(glyph_bgr)
    if f is None: return False
    top, bottom = f["top_density"], f["bottom_density"]
    cdens, stems, ar = f["center_density"], f["stems"], f["ar"]
    ascender = (top >= 0.070)
    descender = (bottom >= 0.10)
    holes = count_glyph_holes(glyph_bgr)

    if label == "t" and (f["small_top_dot"] or descender):
        return True
    if label == "r":
        one_like = (stems == 1) and (ar <= 0.80) and (cdens <= 0.42)
        k_like = (stems >= 2) or (cdens >= 0.50 and ar >= 0.85)
        if one_like or k_like:
            return True
    if label == "5":
        w_like = (ar >= 1.28) and (stems >= 3 or cdens >= 0.48)
        f_like = descender and ascender and (cdens >= 0.50) and (stems >= 2) and (ar <= 1.10)
        if w_like or f_like:
            return True
    if label == "q" and holes == 0 and not descender:
        return True
    if label == "h" and (bottom - top) >= 0.06:
        return True
    return False

# ============ Main ============
class AutoQTE:
    def __init__(self, cfg):
        self.cfg = cfg
        torch.backends.cudnn.benchmark = True

        # Initialize adaptive learner
        self.adaptive_learner = AdaptiveLearner()

        # ROI
        r = cfg["roi"]
        self.region = (int(r["x"]), int(r["y"]), int(r["x"])+int(r["w"]), int(r["y"])+int(r["h"]))

        # Capture
        cap = cfg.get("capture", {})
        self.cam = WGCCapture(
            output_idx=int(cap.get("output_idx", 0)),
            prefer_async=bool(cap.get("prefer_async", True)),
            target_fps=int(cap.get("target_fps", 120)),
            max_buffer_len=int(cap.get("max_buffer_len", 4))
        )
        self.cam.start(self.region)

        # Detector
        det_path = resource_path("weights/yolo_best.pt")
        if not os.path.exists(det_path):
            raise FileNotFoundError(f"Missing detector weights: {det_path}")
        self.det = YOLO(det_path)
        try: self.det.fuse()
        except Exception: pass
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.det.to(self.device)

        # Classifier
        self.keys_all = cfg["keys"]
        cls_path = resource_path("weights/cls_best.pt")
        if not os.path.exists(cls_path):
            raise FileNotFoundError(f"Missing classifier weights: {cls_path}")
        self.cls = Predictor(cls_path, self.keys_all, device=self.device)

        # Allowed subset
        ak = cfg.get("allowed_keys", [])
        self.allowed_idx = set()
        if ak:
            name_to_idx = {k:i for i,k in enumerate(self.cls.class_names)}
            self.allowed_idx = {name_to_idx[k] for k in ak if k in name_to_idx}

        # Template bank
        tfcfg = cfg.get("template", {})
        bankcfg = cfg.get("bank", {})
        self.tbank = TemplateBank(
            max_per_label=int(tfcfg.get("max_per_label", 6)),
            size=int(tfcfg.get("size", 64)),
            max_total=int(bankcfg.get("max_total_entries", 200)),
        )
        self.tm_seed_det = float(tfcfg.get("seed_min_det", 0.90))

        # Bank maintenance params
        self.bank_ttl_sec = float(bankcfg.get("ttl_sec", 8.0))
        self.scene_thr = float(bankcfg.get("scene_change_thr", 0.12))
        self.conflict_win = float(bankcfg.get("conflict_window_sec", 3.0))
        self.conflict_max = int(bankcfg.get("conflict_max", 3))
        self._conflicts = defaultdict(deque)
        self._last_maint_ts = 0.0
        self._last_scene_sig = None

        # Thresholds
        dec = cfg.get("decision", {})
        self.det_gate = float(dec.get("det_gate", 0.70))
        self.det_gate_tm = float(dec.get("det_gate_tm", self.det_gate))
        self.hits_req = int(dec.get("min_hits", 1))
        self.run_req = int(dec.get("min_run", 1))
        self.fused_thr = float(dec.get("fused_thr", 0.25))
        self.margin_thr = float(dec.get("margin_thr", 0.15))
        self.pmargin_thr = float(dec.get("pmargin_thr", 0.15))
        self.alpha_cls = float(cfg.get("fusion", {}).get("alpha_cls", 0.75))
        self.template_veto = float(dec.get("template_veto", 0.80))
        self.template_force = float(dec.get("template_force", 0.92))

        # Human-in-the-loop settings
        hitl = cfg.get("human_in_the_loop", {})
        self.hitl_enable = bool(hitl.get("enable", True))
        self.hitl_confidence_threshold = float(hitl.get("confidence_threshold", 0.65))
        self.hitl_margin_threshold = float(hitl.get("margin_threshold", 0.15))

        # Color/bubble gates
        cg = cfg.get("ui_color_gate", {})
        self.cg_enable = bool(cg.get("enable", True))
        self.cg_h_low = int(cg.get("h_low", 80))
        self.cg_h_high = int(cg.get("h_high", 110))
        self.cg_s_low = int(cg.get("s_low", 50))
        self.cg_v_low = int(cg.get("v_low", 30))
        self.cg_min_ratio = float(cg.get("min_ratio", 0.10))
        self.cg_dilate_px = int(cg.get("dilate_px", 2))

        self.exclude_zones = [(int(z["x"]), int(z["y"]), int(z["w"]), int(z["h"])) for z in cfg.get("exclude_zones", [])]
        self.exclude_iou_thr = float(cfg.get("exclude_iou_thr", 0.15))

        lg = cfg.get("letter_gate", {})
        self.lg_white_s_max = int(lg.get("white_s_max", 60))
        self.lg_white_v_min = int(lg.get("white_v_min", 185))
        self.lg_erode_px = int(lg.get("erode_px", 1))
        self.lg_min_ratio = float(lg.get("min_ratio", 0.004))

        go = cfg.get("glyph_override", {})
        self.go_enable = bool(go.get("enable", True))
        self.go_w_min_ar = float(go.get("w_min_ar", 1.30))
        self.go_u_max_ar = float(go.get("u_max_ar", 1.12))

        bg = cfg.get("bubble_gate", {})
        self.bg_enable = bool(bg.get("enable", True))
        self.bg_min_circularity = float(bg.get("min_circularity", 0.55))
        self.bg_max_ar = float(bg.get("max_aspect_ratio", 1.9))
        self.bg_min_teal_ratio = float(bg.get("min_teal_ratio", self.cg_min_ratio))

        # Overlay
        ov = cfg.get("overlay", {})
        self.green_mode = str(ov.get("green_mode", "det")).lower()

        # Safety/press
        sfty = cfg.get("safety", {})
        try:
            pdi.FAILSAFE = bool(sfty.get("pydirectinput_failsafe", False))
            pdi.PAUSE = 0
        except Exception:
            pass

        pr = cfg.get("press", {})
        self.per_key_delay = pr.get("per_key_delay_ms", 20)/1000.0
        self.cool_down = pr.get("cool_down_ms", 500)/1000.0
        self.global_rate_limit = pr.get("global_rate_limit_ms", 50)/1000.0
        self.last_press_ts = 0.0
        self.recent_label_press = {}
        self.max_presses_per_frame = int(pr.get("max_presses_per_frame", 3))
        self.region_ttl = pr.get("region_ttl_ms", 350)/1000.0
        self.region_iou_thr = float(pr.get("region_iou_thr", 0.35))
        self.region_center_thr = float(pr.get("region_center_dist_px", 40))
        self.locked = deque()

        # Filter
        fcfg = cfg.get("filter", {})
        self.min_area_ratio = fcfg.get("min_area_ratio", 0.0005)
        self.max_area_ratio = fcfg.get("max_area_ratio", 0.12)
        self.min_ar = fcfg.get("min_ar", 0.6)
        self.max_ar = fcfg.get("max_ar", 1.6)

        # Crop/tta
        cc = cfg.get("classification", {})
        self.crop_margin = float(cc.get("crop_margin", 0.30))
        self.tta = cc.get("tta", ["glyph","adaptive","none"])
        self.glyph_isolator = GlyphIsolator(cfg)

        # Debug
        dbg = cfg.get("debug", {})
        self.show = bool(dbg.get("show_window", True))
        self.print_logs = bool(dbg.get("print_logs", True))
        self.overlay_fps = int(dbg.get("overlay_fps", 30))
        self.last_overlay_ts = 0.0
        self.save_dir = dbg.get("save_crops_dir", "")
        self.draw_exclude = bool(dbg.get("draw_exclude_zones", True))

        self.active = False
        self.stop = False
        self.fps = 0.0
        self.accuracy_tracker = {
            "total_predictions": 0,
            "recent_predictions": deque(maxlen=100),
            "char_accuracy": defaultdict(list),
            "press_history": deque(maxlen=50),
            "last_stats_print": 0.0
        }
        self.shape_override_enabled = True  # ← SHAPE OVERRIDE FLAG
        self.performance_monitor = {
            "fps_history": deque(maxlen=60),
            "detection_count": defaultdict(int),
            "confidence_trends": defaultdict(list)
        }

        print("[TRACKING] 📊 Performance monitoring initialized")
    
    def get_char_difficulty(self, char):
        """Get difficulty level for character"""
        easy_chars = ["1", "2", "3", "5", "6", "7", "8", "9", "a", "e", "l", "s", "t"]
        hard_chars = ["4", "q", "d", "p", "h", "u", "v", "w", "b", "g", "n", "m"]
        confusing_pairs = ["4q", "db", "pq", "uv", "hw", "nm"]
        
        if char in easy_chars:
            return "easy"
        elif char in hard_chars:
            return "hard"
        else:
            return "medium"

    def get_adjusted_thresholds(self, predicted_char, confidence):
        """Get character-specific thresholds with confusion handling"""
        difficulty = self.get_char_difficulty(predicted_char)
        base_det = self.det_gate
        base_fused = self.fused_thr
        base_margin = self.margin_thr
        
        # ← THÊM CONFUSING PAIRS LOGIC
        confusing_pairs = ["1r", "1l", "rl", "kh", "uv", "pq", "db", "4t", "t4"]  # ← THÊM 4t
        is_confusing = any(predicted_char in pair for pair in confusing_pairs)
        
        if difficulty == "easy" and not is_confusing:
            return {
                "det_gate": max(0.35, base_det - 0.35),     # ← GIẢM MẠNH HƠN
                "fused_thr": max(0.10, base_fused - 0.15),  # ← GIẢM MẠNH HƠN
                "margin_thr": max(0.03, base_margin - 0.12), # ← GIẢM MẠNH HƠN
                "confidence_boost": 0.1
            }
        elif difficulty == "hard" or is_confusing:
            # ← CRITICAL: GIẢM MẠNH CHO HARD CHARS
            return {
                "det_gate": max(0.50, base_det - 0.20),     # ← GIẢM MẠNH HƠN
                "fused_thr": max(0.15, base_fused - 0.10),  # ← GIẢM MẠNH HƠN  
                "margin_thr": max(0.08, base_margin - 0.07), # ← GIẢM MẠNH HƠN
                "confidence_boost": -0.05
            }
        else:  # medium
            return {
                "det_gate": max(0.45, base_det - 0.25),     # ← GIẢM MẠNH HƠN
                "fused_thr": max(0.12, base_fused - 0.13),  # ← GIẢM MẠNH HƠN
                "margin_thr": max(0.05, base_margin - 0.10), # ← GIẢM MẠNH HƠN
                "confidence_boost": 0.0
            }

    def should_press_with_smart_thresholds(self, enriched_item):
        """Enhanced decision logic with character-specific thresholds"""
        e = enriched_item
        st = e["st"]
        predicted_char = e["label_cls"]

        # ← THÊM MOTION ANALYSIS FIRST
        # Analyze motion first
        is_fast_moving, speed, movement_type = self._is_moving_fast(e["obj_id"], e["box"], st)
        
        if is_fast_moving:
            # MUCH HIGHER REQUIREMENTS for fast moving bubbles
            motion_requirements = self._get_motion_requirements(speed, movement_type, predicted_char)
            
            conf_ok = st["ema_prob"] >= motion_requirements["min_confidence"]
            det_ok = st["det_ema"] >= motion_requirements["min_detection"]
            hits_ok = st["hits"] >= motion_requirements["min_hits"]
            run_ok = st["run"] >= motion_requirements["min_run"]
            stability_ok = st["run"] >= motion_requirements["min_stability"]
            
            if not (conf_ok and det_ok and hits_ok and run_ok and stability_ok):
                if self.print_logs:
                    reasons = []
                    if not conf_ok: reasons.append(f"conf({st['ema_prob']:.2f}<{motion_requirements['min_confidence']:.2f})")
                    if not det_ok: reasons.append(f"det({st['det_ema']:.2f}<{motion_requirements['min_detection']:.2f})")
                    if not hits_ok: reasons.append(f"hits({st['hits']}<{motion_requirements['min_hits']})")
                    if not run_ok: reasons.append(f"run({st['run']}<{motion_requirements['min_run']})")
                    if not stability_ok: reasons.append(f"stability({st['run']}<{motion_requirements['min_stability']})")
                    print(f"[MOTION] 🚫 REJECTED fast moving '{predicted_char}' ({speed:.1f}px/s): {' '.join(reasons)}")
                return False
            
            # Apply region lock check
            region_ok = not self._region_locked(e["box"])
            if not region_ok:
                if self.print_logs:
                    print(f"[MOTION] 🔒 Region locked for fast moving '{predicted_char}'")
                return False
            
            if self.print_logs:
                print(f"[MOTION] ✅ APPROVED fast moving '{predicted_char}' ({speed:.1f}px/s, {movement_type})")
            
            return True

        # ← ANALYZE BUBBLE STATE (REPLACE PARTIAL CHECK)
        bubble_state, visibility = self._analyze_bubble_state(e["box"], self._current_frame.shape)
        requirements = self._get_state_requirements(bubble_state, visibility)
        
        # Skip if bubble should wait until full
        if requirements["skip_until_full"]:
            if self.print_logs:
                print(f"[BUBBLE-STATE] 🕐 SKIPPING '{predicted_char}' - {bubble_state} bubble, waiting for full state")
            return False
        
        # Apply state-specific requirements
        conf_ok = st["ema_prob"] >= requirements["min_confidence"]
        det_ok = st["det_ema"] >= requirements["min_detection"]
        hits_ok = st["hits"] >= requirements["min_hits"]
        run_ok = st["run"] >= requirements["min_run"]
        
        if not (conf_ok and det_ok and hits_ok and run_ok):
            if self.print_logs:
                reasons = []
                if not conf_ok: reasons.append(f"conf({st['ema_prob']:.2f}<{requirements['min_confidence']:.2f})")
                if not det_ok: reasons.append(f"det({st['det_ema']:.2f}<{requirements['min_detection']:.2f})")
                if not hits_ok: reasons.append(f"hits({st['hits']}<{requirements['min_hits']})")
                if not run_ok: reasons.append(f"run({st['run']}<{requirements['min_run']})")
                print(f"[BUBBLE-STATE] 🚫 REJECTED '{predicted_char}' ({bubble_state}): {' '.join(reasons)}")
            return False
        
        # Apply region lock check
        region_ok = not self._region_locked(e["box"])
        if not region_ok:
            if self.print_logs:
                print(f"[BUBBLE-STATE] 🔒 Region locked for '{predicted_char}'")
            return False
        
        if self.print_logs:
            print(f"[BUBBLE-STATE] ✅ APPROVED '{predicted_char}' ({bubble_state}, vis:{visibility:.2f})")
        
        return True

    # ---- Bank maintenance ----
    def _maintain_bank(self, frame_bgr):
        now = time.time()
        if (now - self._last_maint_ts) < 1.0:
            return
        self._last_maint_ts = now

        self.tbank.prune(ttl_sec=self.bank_ttl_sec, now=now, min_keep=1)

        sig = scene_signature(frame_bgr)
        if self._last_scene_sig is not None:
            diff = float(np.mean(np.abs(sig - self._last_scene_sig)))
            if diff >= self.scene_thr:
                self.tbank.clear()
                self._conflicts.clear()
                if self.print_logs:
                    print(f"[BANK] scene-change diff={diff:.3f} -> cleared")
        self._last_scene_sig = sig

        cutoff = now - self.conflict_win
        for lbl, dq in list(self._conflicts.items()):
            while dq and dq[0] < cutoff:
                dq.popleft()
            if not dq:
                del self._conflicts[lbl]

    def _record_conflict(self, label: str):
        now = time.time()
        dq = self._conflicts[label]
        dq.append(now)
        cutoff = now - self.conflict_win
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= self.conflict_max:
            self.tbank.drop_label(label)
            self._conflicts[label].clear()
            if self.print_logs:
                print(f"[BANK] dropped label '{label}' due to conflicts >= {self.conflict_max} in {self.conflict_win}s")

    # ---- Filters / gates ---- 
    def _filter_boxes(self, boxes, W, H):
        out = []
        area_img = float(W*H)
        for (x1,y1,x2,y2) in boxes:
            w = x2-x1; h = y2-y1
            if w<=0 or h<=0: continue
            area = w*h
            ar = w / max(1e-6, h)
            ratio = area / area_img
            if not (self.min_area_ratio <= ratio <= self.max_area_ratio): continue
            if not (self.min_ar <= ar <= self.max_ar): continue
            out.append((x1,y1,x2,y2))
        return out

    def _box_in_exclude(self, box):
        x1,y1,x2,y2 = box
        for (ex,ey,ew,eh) in self.exclude_zones:
            X1, Y1, X2, Y2 = ex, ey, ex+ew, ey+eh
            cx, cy = center(box)
            if (X1 <= cx <= X2) and (Y1 <= cy <= Y2):
                return True
            ix1, iy1 = max(x1, X1), max(y1, Y1)
            ix2, iy2 = min(x2, X2), min(y2, Y2)
            iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
            inter = iw*ih
            if inter > 0:
                ua = (x2-x1)*(y2-y1) + (X2-X1)*(Y2-Y1) - inter + 1e-6
                if (inter/ua) >= self.exclude_iou_thr:
                    return True
        return False

    def _ui_color_pass(self, frame, box):
        if not self.cg_enable:
            return True
        x1,y1,x2,y2 = box
        x1 = max(0, x1 - self.cg_dilate_px); y1 = max(0, y1 - self.cg_dilate_px)
        x2 = min(frame.shape[1]-1, x2 + self.cg_dilate_px); y2 = min(frame.shape[0]-1, y2 + self.cg_dilate_px)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        mask = (h >= self.cg_h_low) & (h <= self.cg_h_high) & (s >= self.cg_s_low) & (v >= self.cg_v_low)
        ratio = float(mask.mean()) if mask.size else 0.0
        return ratio >= self.cg_min_ratio

    def _bubble_shape_pass(self, frame, box):
        if not self.bg_enable:
            return True
        x1,y1,x2,y2 = box
        crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        mask = ((h >= self.cg_h_low) & (h <= self.cg_h_high) & (s >= self.cg_s_low) & (v >= self.cg_v_low)).astype(np.uint8) * 255
        if mask.size == 0:
            return False
        teal_ratio = float((mask > 0).mean())
        if teal_ratio < self.bg_min_teal_ratio:
            return False
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return False
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perim = max(1e-6, cv2.arcLength(cnt, True))
        circularity = float(4.0 * np.pi * area / (perim * perim))
        x,y,w,h = cv2.boundingRect(cnt)
        ar = float(max(w,1)) / float(max(h,1))
        if circularity < self.bg_min_circularity:
            return False
        if ar > self.bg_max_ar or (1.0/ar) > self.bg_max_ar:
            return False
        return True

    def _tta_crops(self, frame, box):
        H, W = frame.shape[:2]
        
        # ← THÊM MOTION STATE CHECK
        # Store current obj_id for motion check
        if hasattr(self, '_current_obj_id'):
            obj_id = self._current_obj_id
            state = getattr(self, 'state', {}).get(obj_id, {})
            is_fast_moving, speed, movement_type = self._is_moving_fast(obj_id, box, state)
            
            if is_fast_moving:
                # Special processing for fast moving bubbles
                crop_margin = max(0.35, self.crop_margin + 0.05)  # Slightly larger crop
                if self.print_logs:
                    print(f"[MOTION-CROP] Using enhanced crops for fast moving bubble ({speed:.1f}px/s)")
            else:
                crop_margin = self.crop_margin
        else:
            crop_margin = self.crop_margin

        # ← THÊM PARTIAL BUBBLE CHECK
        is_partial, visibility = self._is_partial_bubble(box, frame.shape)
        
        if is_partial:
            # Use SMALLER margin for partial bubbles to avoid including background
            crop_margin = max(0.10, self.crop_margin - 0.20)  # ← GIẢM margin
            if self.print_logs:
                print(f"[PARTIAL-CROP] Using reduced margin {crop_margin:.2f} for partial bubble")
        else:
            crop_margin = self.crop_margin
        
        x1,y1,x2,y2 = expand_box(*box, W,H, crop_margin)
        crop = frame[y1:y2, x1:x2]
        
        # ← THÊM ENHANCED PREPROCESSING
        # Method 1: Original tight crop
        tight = refine_glyph_roi(crop)
        
        # Method 2: Enhanced contrast
        enhanced = self._enhance_crop_contrast(crop)
        
        # Method 3: Larger margin crop (more context)
        x1_big, y1_big, x2_big, y2_big = expand_box(*box, W, H, self.crop_margin + 0.1)
        crop_big = frame[y1_big:y2_big, x1_big:x2_big]
        tight_big = refine_glyph_roi(crop_big)
        
        outs = []
        for mode in self.tta:
            if mode == "adaptive":
                g = cv2.cvtColor(tight, cv2.COLOR_BGR2GRAY)
                g = cv2.GaussianBlur(g, (3,3), 0)
                bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 19, 2)
                if (bw == 255).mean() > 0.6: bw = 255 - bw
                outs.append(cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR))
            elif mode == "glyph":
                glyph = self.glyph_isolator(tight)
                outs.append(glyph if glyph is not None else tight)
            else:
                outs.append(tight)
        
        outs.append(enhanced)     # Enhanced contrast
        outs.append(tight_big)    # Bigger context
        
        # ← THÊM NUMBER-SPECIFIC CROP (FIX 5)
        number_enhanced = self._enhance_for_numbers(tight)
        outs.append(number_enhanced)

        # ← SPECIAL PROCESSING FOR FAST MOVING (NEW)
        if hasattr(self, '_current_obj_id') and is_fast_moving:
            # Method 1: Sharpened version (reduce motion blur)
            sharpened = self._sharpen_for_motion(tight)
            outs.append(sharpened)
            
            # Method 2: Edge-enhanced version
            edge_enhanced = self._enhance_edges_for_motion(tight)
            outs.append(edge_enhanced)
        
        return outs if outs else [tight]

    def _enhance_crop_contrast(self, crop):
        """Enhance crop contrast for better recognition"""
        try:
            if crop is None or crop.size == 0:
                return crop
                
            # Convert to LAB for better contrast enhancement
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except Exception:
            return crop
    def _enhance_for_numbers(self, crop):
        """Special enhancement for number recognition"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Sharpen for better edge definition
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            
            # High contrast threshold
            _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Ensure white text on black background
            if np.mean(binary) > 127:
                binary = 255 - binary
                
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        except:
            return crop
        
    def _sharpen_for_motion(self, crop):
        """Sharpen image to reduce motion blur effects"""
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Strong sharpening kernel for motion blur
            kernel = np.array([[-2,-2,-2], 
                            [-2,17,-2], 
                            [-2,-2,-2]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            
            return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        except:
            return crop

    def _enhance_edges_for_motion(self, crop):
        """Enhance edges for better feature detection during motion"""
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Edge detection + dilation to strengthen features
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((2,2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        except:
            return crop

    def _region_locked(self, box):
        now = time.time()
        while self.locked and self.locked[0][1] < now:
            self.locked.popleft()
        cx, cy = center(box)
        for b, _ in self.locked:
            if iou(box, b) >= self.region_iou_thr:
                return True
            bx, by = center(b)
            if dist2((cx,cy),(bx,by)) <= (self.region_center_thr**2):
                return True
        return False

    # ---- Pressing ----
    def _press(self, label, box, fused, detc, reason, obj_id=None):
        """Enhanced press with object-based tracking - MỖI OBJECT CHỈ PRESS 1 LẦN"""
        now = time.time()
        
        # ← THÊM CROP SAVING CHO DEBUG FALSE POSITIVES
        if self.print_logs:
            try:
                # Save crop for debugging false positives
                if hasattr(self, '_current_frame') and self._current_frame is not None:
                    H, W = self._current_frame.shape[:2]
                    x1, y1, x2, y2 = box
                    # Expand crop slightly for better context
                    margin = 5
                    x1_exp = max(0, x1 - margin)
                    y1_exp = max(0, y1 - margin)
                    x2_exp = min(W, x2 + margin)
                    y2_exp = min(H, y2 + margin)
                    
                    crop = self._current_frame[y1_exp:y2_exp, x1_exp:x2_exp]
                    if crop.size > 0:
                        timestamp = int(time.time() * 1000) % 1000000
                        crop_path = f"C:\\Users\\DatGo\\cauca\\debugs\\press\\debug_press_{label}_{timestamp}.png"
                        cv2.imwrite(crop_path, crop)
                        print(f"[DEBUG-CROP] 📸 Saved press crop: {crop_path} (box: {box})")
                        
                        # Also save original detection area
                        detection_crop = self._current_frame[y1:y2, x1:x2]
                        if detection_crop.size > 0:
                            tight_path = f"C:\\Users\\DatGo\\cauca\\debugs\\tight\\debug_tight_{label}_{timestamp}.png"
                            cv2.imwrite(tight_path, detection_crop)
                            print(f"[DEBUG-CROP] 📸 Saved tight crop: {tight_path}")
            except Exception as e:
                print(f"[DEBUG-CROP] ❌ Error saving crop: {e}")
        
        # ← CHECK POPUP ACTIVE - PAUSE KHI POPUP HIỆN
        if self.adaptive_learner.popup_active:
            if self.print_logs:
                print(f"[PRESS] 🛑 PAUSED - Popup active, not pressing '{label}'")
            return False

        # ← OBJECT-BASED PRESS TRACKING - CRITICAL FIX
        if not hasattr(self, 'pressed_objects'):
            self.pressed_objects = {}  # {obj_id: {label, timestamp, box}}

        # Check if this object was already pressed
        if obj_id in self.pressed_objects:
            prev_press = self.pressed_objects[obj_id]
            if self.print_logs:
                print(f"[PRESS] 🚫 OBJECT ID{obj_id} already pressed as '{prev_press['label']}' at {prev_press['timestamp']:.2f}")
                print(f"[PRESS] 🚫 BLOCKING duplicate press of '{label}' - EACH OBJECT PRESSED ONLY ONCE")
            return False    

        # ← GLOBAL RATE LIMIT - SHORTER
        global_rate_limit = 0.05  # ← 50ms between ANY presses
        if now - self.last_press_ts < global_rate_limit:
            if self.print_logs:
                print(f"[PRESS] ⏸️ Global rate limit ({global_rate_limit:.3f}s)")
            return False
        
        # ← REGION LOCK CHECK
        if self._region_locked(box):
            if self.print_logs:
                print(f"[PRESS] 🔒 Region locked for '{label}'")
            return False
        
        # ← EXECUTE PRESS
        try:
            if self.print_logs:
                print(f"[PRESS] 🎯 FIRST PRESS for Object ID{obj_id}: '{label}' at box {box}")
            
            pdi.keyDown(label)
            time.sleep(self.per_key_delay)
            pdi.keyUp(label)
            
            # ← RECORD THIS OBJECT AS PRESSED
            self.pressed_objects[obj_id] = {
                'label': label,
                'timestamp': now,
                'box': box,
                'fused': fused,
                'detection': detc,
                'reason': reason
            }
            
            if self.print_logs:
                print(f"[PRESS] ✅ ID{obj_id} REGISTERED: '{label}' fused={fused:.2f} det={detc:.2f} policy={reason}")
            
        except Exception as e:
            if self.print_logs:
                print(f"[PRESS][ERROR] ❌ Press failed: {e}")
            return False
        
        # ← UPDATE GLOBAL TRACKING
        self.last_press_ts = now
        self.locked.append((box, now + self.region_ttl))
        
        return True
    
    # ← CLEANUP PRESSED OBJECTS (ADD NEW METHOD)
    def _cleanup_pressed_objects(self):
        """Clean up pressed objects that are no longer being tracked"""
        if not hasattr(self, 'pressed_objects'):
            return
        
        current_time = time.time()
        
        # Get currently active object IDs
        active_obj_ids = set(getattr(self, 'state', {}).keys())
        
        # Remove pressed objects that are no longer active
        to_remove = []
        for obj_id in self.pressed_objects:
            if obj_id not in active_obj_ids:
                # Object disappeared, mark for removal
                to_remove.append(obj_id)
            elif current_time - self.pressed_objects[obj_id]['timestamp'] > 5.0:
                # Very old press, clean up
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            if self.print_logs:
                print(f"[CLEANUP] 🗑️ Removing pressed object ID{obj_id}")
            del self.pressed_objects[obj_id]

    # Thêm vào class AutoQTE (sau method _press):

    def _track_prediction(self, obj_id, predicted_char, confidence, margin, was_pressed=False):
        """Track prediction for performance analysis with enhanced metrics"""
        if not hasattr(self, 'accuracy_tracker'):
            return
            
        now = time.time()
        
        # Get character difficulty and thresholds for analysis
        difficulty = self.get_char_difficulty(predicted_char)
        thresholds = self.get_adjusted_thresholds(predicted_char, confidence)
        
        prediction_data = {
            "obj_id": obj_id,
            "char": predicted_char,
            "confidence": confidence,
            "margin": margin,
            "timestamp": now,
            "was_pressed": was_pressed,
            "difficulty": difficulty,
            "adjusted_thresholds": thresholds,
            "frame_number": getattr(self, '_frame_counter', 0)
        }
        
        self.accuracy_tracker["total_predictions"] += 1
        self.accuracy_tracker["recent_predictions"].append(prediction_data)
        self.accuracy_tracker["char_accuracy"][predicted_char].append(confidence)
        
        if was_pressed:
            self.accuracy_tracker["press_history"].append(prediction_data)
            if self.print_logs:
                print(f"[TRACK] 🎯 PRESSED: '{predicted_char}' conf={confidence:.2f} difficulty={difficulty}")
        
        # Performance metrics
        self.performance_monitor["detection_count"][predicted_char] += 1
        self.performance_monitor["confidence_trends"][predicted_char].append(confidence)
        
        # Auto-print stats every 30 seconds
        if now - self.accuracy_tracker["last_stats_print"] > 30.0:
            self._print_performance_stats()
            self.accuracy_tracker["last_stats_print"] = now

    def _print_performance_stats(self):
        """Enhanced performance statistics with character difficulty analysis"""
        recent = list(self.accuracy_tracker["recent_predictions"])
        if not recent:
            return
        
        print(f"\n[STATS] 📊 Performance Report (Last {len(recent)} predictions):")
        
        # Overall stats
        total_preds = self.accuracy_tracker["total_predictions"]
        avg_confidence = np.mean([p["confidence"] for p in recent])
        avg_margin = np.mean([p["margin"] for p in recent])
        press_rate = sum(1 for p in recent if p["was_pressed"]) / len(recent) * 100
        
        print(f"[STATS] Total predictions: {total_preds}")
        print(f"[STATS] Avg confidence: {avg_confidence:.2f}")
        print(f"[STATS] Avg margin: {avg_margin:.2f}")
        print(f"[STATS] Press rate: {press_rate:.1f}%")
        
        # Character breakdown by difficulty
        easy_chars = [p for p in recent if p["difficulty"] == "easy"]
        medium_chars = [p for p in recent if p["difficulty"] == "medium"]
        hard_chars = [p for p in recent if p["difficulty"] == "hard"]
        
        if easy_chars:
            easy_conf = np.mean([p["confidence"] for p in easy_chars])
            easy_press_rate = sum(1 for p in easy_chars if p["was_pressed"]) / len(easy_chars) * 100
            print(f"[STATS] Easy chars: {len(easy_chars)} detections, {easy_conf:.2f} avg conf, {easy_press_rate:.0f}% pressed")
        
        if medium_chars:
            medium_conf = np.mean([p["confidence"] for p in medium_chars])
            medium_press_rate = sum(1 for p in medium_chars if p["was_pressed"]) / len(medium_chars) * 100
            print(f"[STATS] Medium chars: {len(medium_chars)} detections, {medium_conf:.2f} avg conf, {medium_press_rate:.0f}% pressed")
        
        if hard_chars:
            hard_conf = np.mean([p["confidence"] for p in hard_chars])
            hard_press_rate = sum(1 for p in hard_chars if p["was_pressed"]) / len(hard_chars) * 100
            print(f"[STATS] Hard chars: {len(hard_chars)} detections, {hard_conf:.2f} avg conf, {hard_press_rate:.0f}% pressed")
        
        # Top characters detected
        char_counts = defaultdict(int)
        char_press_counts = defaultdict(int)
        for p in recent:
            char_counts[p["char"]] += 1
            if p["was_pressed"]:
                char_press_counts[p["char"]] += 1
        
        print(f"[STATS] Top detected characters:")
        top_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for char, count in top_chars:
            pressed = char_press_counts.get(char, 0)
            press_rate = pressed / count * 100 if count > 0 else 0
            difficulty = self.get_char_difficulty(char)
            print(f"[STATS]   '{char}': {count} detections, {pressed} pressed ({press_rate:.0f}%) - {difficulty}")
        
        # FPS performance
        if hasattr(self, 'fps') and self.fps > 0:
            self.performance_monitor["fps_history"].append(self.fps)
            if len(self.performance_monitor["fps_history"]) > 10:
                avg_fps = np.mean(list(self.performance_monitor["fps_history"])[-10:])
                min_fps = min(list(self.performance_monitor["fps_history"])[-10:])
                max_fps = max(list(self.performance_monitor["fps_history"])[-10:])
                print(f"[STATS] FPS: avg={avg_fps:.1f}, min={min_fps:.1f}, max={max_fps:.1f}")
        
        print("[STATS] 📊 End Report\n")

    def _track_fps_performance(self):
        """Track FPS and system performance"""
        if hasattr(self, 'fps') and self.fps > 0:
            self.performance_monitor["fps_history"].append(self.fps)
            
            # Keep only recent FPS data (last 5 minutes)
            if len(self.performance_monitor["fps_history"]) > 300:
                self.performance_monitor["fps_history"].popleft()
            
            # Detect performance issues
            if len(self.performance_monitor["fps_history"]) > 30:
                recent_fps = list(self.performance_monitor["fps_history"])[-30:]
                avg_fps = np.mean(recent_fps)
                if avg_fps < 20 and self.print_logs:
                    print(f"[PERF] ⚠️ Low FPS detected: {avg_fps:.1f} (last 30 frames)")

    # ← THÊM METHOD MỚI Ở ĐÂY
    def _is_falling_bubble(self, obj_id, current_box, state):
        """Detect falling bubble pattern"""
        if "y_history" not in state:
            state["y_history"] = []
        
        now = time.time()
        _, y1, _, y2 = current_box
        center_y = (y1 + y2) / 2
        
        # Track Y position over time
        state["y_history"].append((center_y, now))
        
        # Keep last 5 positions (last 0.3 seconds)
        cutoff = now - 0.3
        state["y_history"] = [(y, t) for (y, t) in state["y_history"] if t >= cutoff]
        
        if len(state["y_history"]) < 3:
            return False, 0
        
        # Calculate falling speed
        recent = state["y_history"]
        if len(recent) < 2:
            return False, 0
        
        y_start, t_start = recent[0]
        y_end, t_end = recent[-1]
        
        time_diff = t_end - t_start
        y_diff = y_end - y_start  # Positive = falling down
        
        if time_diff <= 0:
            return False, 0
        
        falling_speed = y_diff / time_diff  # pixels per second
        
        # Consider falling if moving down > 50 px/s
        is_falling = falling_speed > 50
        
        if is_falling and self.print_logs:
            print(f"[FALLING] ID{obj_id}: Falling at {falling_speed:.1f} px/s")
        
        return is_falling, falling_speed

    def _boost_falling_confidence(self, predicted_char, confidence, fall_speed):
        """Boost confidence for consistently falling bubbles"""
        
        # If bubble is falling consistently, boost confidence
        if fall_speed > 80:  # Consistent falling motion
            
            # Characters that are often confused when falling fast
            if predicted_char in ['g', '6', 'q', '9']:
                confidence_boost = 0.1
            elif predicted_char in ['b', 'd', 'p']:
                confidence_boost = 0.08
            elif predicted_char in ['m', 'w', 'n']:
                confidence_boost = 0.05
            else:
                confidence_boost = 0.03
            
            boosted = min(1.0, confidence + confidence_boost)
            
            if self.print_logs:
                print(f"[CONF-BOOST] '{predicted_char}': {confidence:.2f} → {boosted:.2f} (falling motion)")
            
            return boosted
        
        return confidence          

    # ← THÊM PARTIAL BUBBLE METHODS Ở ĐÂY
    def _get_bubble_visibility_ratio(self, box, frame_height):
        """Calculate how much of bubble is visible in frame"""
        x1, y1, x2, y2 = box
        
        # Check if bubble is cut off at top (just spawned)
        top_cutoff = max(0, -y1) / max(1, y2 - y1)
        
        # Check if bubble is cut off at bottom (leaving area)
        bottom_cutoff = max(0, y2 - frame_height) / max(1, y2 - y1)
        
        # Calculate visibility ratio
        visibility = 1.0 - top_cutoff - bottom_cutoff
        
        return visibility, top_cutoff > 0, bottom_cutoff > 0

    def _is_partial_bubble(self, box, frame_shape):
        """Check if bubble is partially visible"""
        H, W = frame_shape[:2]
        visibility, is_top_cut, is_bottom_cut = self._get_bubble_visibility_ratio(box, H)
        
        # Consider partial if less than 80% visible
        is_partial = visibility < 0.80
        
        if is_partial and self.print_logs:
            print(f"[PARTIAL] Bubble visibility: {visibility:.2f} (top_cut={is_top_cut}, bottom_cut={is_bottom_cut})")
        
        return is_partial, visibility

    def _should_wait_for_full_visibility(self, enriched_item):
        """Check if should wait for bubble to be fully visible"""
        e = enriched_item
        box = e["box"]
        
        # Get bubble position
        x1, y1, x2, y2 = box
        bubble_height = y2 - y1
        
        # If bubble is in top 20% of screen and partially cut
        frame_height = self._current_frame.shape[0]
        is_in_spawn_area = y1 < (frame_height * 0.2)
        is_partial, visibility = self._is_partial_bubble(box, self._current_frame.shape)
        
        # Wait if bubble just spawned and not fully visible
        should_wait = is_in_spawn_area and is_partial and visibility < 0.70
        
        if should_wait and self.print_logs:
            print(f"[WAIT] 🕐 Waiting for full visibility: '{e['label_cls']}' (vis={visibility:.2f})")
        
        return should_wait
    
    def _analyze_bubble_state(self, box, frame_shape):
        """Analyze if bubble is partial, full, or leaving"""
        H, W = frame_shape[:2]
        x1, y1, x2, y2 = box
        
        bubble_height = y2 - y1
        bubble_width = x2 - x1
        center_y = (y1 + y2) / 2
        
        # Check position relative to frame
        top_ratio = y1 / H  # How far from top (0 = very top)
        bottom_ratio = y2 / H  # How far to bottom (1 = bottom edge)
        
        # Classify bubble state
        if y1 < 10:  # Very close to top edge
            state = "spawning"
            visibility = max(0.3, (y2 - 10) / bubble_height)
        elif y2 > H - 10:  # Very close to bottom edge  
            state = "leaving"
            visibility = max(0.3, (H - 10 - y1) / bubble_height)
        elif top_ratio < 0.15:  # In spawn area but not cut off
            state = "entering"
            visibility = min(1.0, 0.6 + (top_ratio / 0.15) * 0.4)
        elif bottom_ratio > 0.85:  # In exit area
            state = "exiting" 
            visibility = min(1.0, 0.6 + ((1.0 - bottom_ratio) / 0.15) * 0.4)
        else:  # Middle area
            state = "full"
            visibility = 1.0
        
        if self.print_logs and state != "full":
            print(f"[BUBBLE-STATE] State: {state}, Visibility: {visibility:.2f}, Y-range: {y1}-{y2} (frame: {H})")
        
        return state, visibility

    def _get_state_requirements(self, bubble_state, visibility):
        """Get detection requirements based on bubble state"""
        if bubble_state == "spawning":
            # Very strict for spawning bubbles
            return {
                "min_confidence": 0.60,
                "min_detection": 0.85,
                "min_hits": 4,
                "min_run": 3,
                "skip_until_full": True
            }
        elif bubble_state == "entering":
            # Strict for entering bubbles
            return {
                "min_confidence": 0.45,
                "min_detection": 0.75,
                "min_hits": 3,
                "min_run": 2,
                "skip_until_full": False
            }
        elif bubble_state == "full":
            # Normal requirements for full bubbles
            return {
                "min_confidence": 0.25,
                "min_detection": 0.60,
                "min_hits": 1,
                "min_run": 1,
                "skip_until_full": False
            }
        elif bubble_state == "exiting":
            # Slightly higher for exiting
            return {
                "min_confidence": 0.35,
                "min_detection": 0.65,
                "min_hits": 2,
                "min_run": 1,
                "skip_until_full": False
            }
        else:  # leaving
            # Skip leaving bubbles entirely
            return {
                "min_confidence": 0.70,
                "min_detection": 0.90,
                "min_hits": 5,
                "min_run": 3,
                "skip_until_full": True
            }
        
    # ← THÊM MOTION DETECTION METHOD Ở ĐÂY
    def _is_moving_fast(self, obj_id, current_box, state):
        """Detect if bubble is moving fast (any direction)"""
        if "position_history" not in state:
            state["position_history"] = []
        
        now = time.time()
        x1, y1, x2, y2 = current_box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Track position over time
        state["position_history"].append((center_x, center_y, now))
        
        # Keep last 5 positions (last 0.4 seconds)
        cutoff = now - 0.6
        state["position_history"] = [(x, y, t) for (x, y, t) in state["position_history"] if t >= cutoff]
        
        if len(state["position_history"]) < 5:
            return False, 0.0, "unknown"
        
        # Calculate movement speed and direction
        recent = state["position_history"]
        total_distance = 0
        total_time = 0
        
        for i in range(1, len(recent)):
            x1, y1, t1 = recent[i-1]
            x2, y2, t2 = recent[i]
            
            distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
            time_diff = t2 - t1
            
            total_distance += distance
            total_time += time_diff
        
        if total_time <= 0:
            return False, 0.0, "unknown"
        
        speed = total_distance / total_time  # pixels per second

        # ← CALCULATE CONSISTENCY - NEW FEATURE
        # Check if movement is consistently fast (not just momentary)
        speeds = []
        for i in range(1, len(recent)):
            x1, y1, t1 = recent[i-1]
            x2, y2, t2 = recent[i]
            
            distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
            time_diff = t2 - t1
            
            if time_diff > 0:
                segment_speed = distance / time_diff
                speeds.append(segment_speed)
        
        # Movement must be consistently fast
        if speeds:
            avg_speed = np.mean(speeds)
            speed_consistency = 1.0 - (np.std(speeds) / max(avg_speed, 1))  # Higher = more consistent
            
            # ← STRICTER REQUIREMENTS
            is_fast = (avg_speed > 80) and (speed_consistency > 0.6)  # ← GIẢM threshold, THÊM consistency
            
            if is_fast and self.print_logs:
                print(f"[MOTION] ID{obj_id}: Consistent fast movement {avg_speed:.1f}px/s (consistency: {speed_consistency:.2f})")
            
            return is_fast, avg_speed, "consistent_fast"
        
        return False, 0.0, "inconsistent"

    def _get_motion_requirements(self, speed, movement_type, predicted_char):
        """Get requirements based on motion and character confusion patterns"""
        
        # Characters commonly confused when moving
        confusion_prone = {
            'h': ['b', 'n'],  # h loses ascender → becomes b/n
            'w': ['m', 'v', 'u'],  # w loses width definition → becomes m/v/u  
            'e': ['r', 'c'],  # e loses middle bar → becomes r/c
            'y': ['r', 'v'],  # y loses descender → becomes r/v
            'p': ['d', 'b'],  # p loses descender → becomes d/b
            'q': ['d', 'o'],  # q loses descender → becomes d/o
            'g': ['6', 'o'],  # g loses features → becomes 6/o
            'b': ['h', 'd'],  # b loses bottom → becomes h/d
            'd': ['b', 'p'],  # d loses features → becomes b/p
        }
        
        is_confusion_prone = predicted_char in confusion_prone
        
        # Base requirements by speed
        if speed > 200:  # Very fast
            base_conf = 0.65
            base_det = 0.80
            base_hits = 4
            base_stability = 3
        elif speed > 150:  # Fast
            base_conf = 0.55
            base_det = 0.75
            base_hits = 3
            base_stability = 2
        else:  # Medium fast (120-150)
            base_conf = 0.45
            base_det = 0.70
            base_hits = 2
            base_stability = 2
        
        # Increase requirements for confusion-prone characters
        if is_confusion_prone:
            base_conf += 0.10
            base_det += 0.05
            base_hits += 1
            base_stability += 1
            
            if self.print_logs:
                possible_confusions = confusion_prone[predicted_char]
                print(f"[MOTION] ⚠️ '{predicted_char}' is confusion-prone (could be: {possible_confusions})")
        
        # Movement type adjustments
        if movement_type == "horizontal":
            # Horizontal movement can cause more blur
            base_conf += 0.05
            base_hits += 1
        elif movement_type == "diagonal":
            # Diagonal movement is most problematic
            base_conf += 0.08
            base_det += 0.03
            base_hits += 1
        
        return {
            "min_confidence": min(0.85, base_conf),
            "min_detection": min(0.90, base_det),
            "min_hits": base_hits,
            "min_run": base_hits,
            "min_stability": base_stability
    }    

    # ---- Run loop ----
    def run(self):
        threading.Thread(target=self._hotkeys, daemon=True).start()

        t0 = time.time(); frames = 0
        while not self.stop:
            frame = self.cam.get_latest()
            if frame is None:
                frame = self.cam.grab(self.region)
            if frame is None:
                continue
            frame = np.ascontiguousarray(frame[..., :3])
            
            # ← SAVE CURRENT FRAME FOR DEBUG
            self._current_frame = frame.copy()

            frames += 1
            self._frame_counter = frames

            if frames % 20 == 0:
                t1 = time.time(); self.fps = 20/(t1-t0+1e-6); t0 = t1

            self._maintain_bank(frame)

            res = self.det.predict(
                frame,
                conf=self.cfg["inference"]["conf_thres"],
                imgsz=self.cfg["inference"]["imgsz"],
                half=self.cfg["inference"]["fp16"] and (self.device=="cuda"),
                verbose=False,
                max_det=self.cfg["inference"]["max_det"]
            )[0]

            if res.boxes is None or len(res.boxes)==0:
                self._draw(frame, [])
                continue

            boxes = res.boxes.xyxy.detach().cpu().numpy().astype(int)
            confs = res.boxes.conf.detach().cpu().numpy().astype(float)

            H, W = frame.shape[:2]
            boxes_f = self._filter_boxes(boxes, W, H)

            # ← DEBUG CHO MISSING CHARACTERS
            if self.print_logs and len(boxes) > 0:
                total_boxes = len(boxes)
                filtered_boxes = len(boxes_f)
                rejected_boxes = total_boxes - filtered_boxes
                print(f"[BOX-DEBUG] YOLO detected: {total_boxes} boxes, After filters: {filtered_boxes}")
                if rejected_boxes > 0:
                    print(f"[BOX-DEBUG] ⚠️ {rejected_boxes} boxes REJECTED by size/aspect ratio filters")

            now = time.time()
            items = []
            gate_rejections = {"exclude": 0, "color": 0, "shape": 0}
            
            for idx, ((x1,y1,x2,y2), conf) in enumerate(zip(boxes, confs)):
                if (x1,y1,x2,y2) not in boxes_f:
                    continue
                box = (x1,y1,x2,y2)
                
                # ← DEBUG GATE REJECTIONS
                if self._box_in_exclude(box):
                    gate_rejections["exclude"] += 1
                    if self.print_logs:
                        print(f"[GATE-DEBUG] Box {idx} REJECTED: in exclude zone")
                    continue
                if not self._ui_color_pass(frame, box):
                    gate_rejections["color"] += 1
                    if self.print_logs:
                        print(f"[GATE-DEBUG] Box {idx} REJECTED: failed color gate")
                    continue
                if not self._bubble_shape_pass(frame, box):
                    gate_rejections["shape"] += 1
                    if self.print_logs:
                        print(f"[GATE-DEBUG] Box {idx} REJECTED: failed shape gate")
                    continue

                obj_id = idx
                st = getattr(self, "state", {}).get(obj_id)
                if st is None:
                    st = {"first": now, "last": now, "hits": 0, "run": 0, "last_label": None,
                        "ema_prob": None, "pressed": 0, "pressed_at": 0.0,
                        "last_prob": 0.0, "last_margin": 0.0, "last_pmargin": 0.0,
                        "tm": 0.0, "fused": 0.0, "label": None, "det_ema": float(conf)}
                    if not hasattr(self, "state"):
                        self.state = {}
                    self.state[obj_id] = st
                else:
                    st["last"] = now
                    st["det_ema"] = 0.6*float(conf) + 0.4*st.get("det_ema", float(conf))
                st["hits"] += 1
                items.append((obj_id, box, st))

            # ← PRINT GATE REJECTION SUMMARY
            if self.print_logs and any(gate_rejections.values()):
                print(f"[GATE-DEBUG] Rejections - Exclude: {gate_rejections['exclude']}, Color: {gate_rejections['color']}, Shape: {gate_rejections['shape']}")

            # TTL cleanup (existing)
            for k in list(getattr(self, "state", {}).keys()):
                if now - self.state[k]["last"] > float(self.cfg.get("tracker", {}).get("state_ttl", 1.5)):
                    del self.state[k]

            # ← ADD PRESSED OBJECTS CLEANUP
            self._cleanup_pressed_objects()

            if self.active:
                # ← DEBUG FINAL ITEMS COUNT
                if self.print_logs:
                    print(f"[PIPELINE-DEBUG] Final items for classification: {len(items)}")

                enriched = []
                classification_debug = []
                
                for obj_id, box, st in items:
                    self._current_obj_id = obj_id  # Store for motion detection in _tta_crops
                    
                    crops = self._tta_crops(frame, box)
                    label_cls, prob1, prob2 = self.cls.predict_ensemble(crops, allowed_idx=self.allowed_idx if self.allowed_idx else None)

                    # ← TRACK ORIGINAL CLASSIFICATION
                    original_label = label_cls
                    classification_debug.append({
                        "obj_id": obj_id,
                        "original": original_label,
                        "conf": prob1,
                        "second": prob2,
                        "box": box
                    })

                    # ← THÊM FALLING MOTION CHECK VÀ CONFIDENCE BOOST
                    is_falling, fall_speed = self._is_falling_bubble(obj_id, box, st)
                    
                    # ← BOOST CONFIDENCE FOR FALLING BUBBLES
                    if is_falling:
                        prob1 = self._boost_falling_confidence(label_cls, prob1, fall_speed)

                    # Shape override với adaptive learning
                    glyph = crops[0] if len(crops)>0 and crops[0] is not None else None
                    if glyph is not None:
                        # ← THÊM CHECK FLAG
                        if self.shape_override_enabled:
                            label_cls = shape_override(label_cls, glyph, self.cfg, self.adaptive_learner)
                            # ← DEBUG SHAPE OVERRIDE
                            if label_cls != original_label and self.print_logs:
                                print(f"[SHAPE-DEBUG] ID{obj_id}: '{original_label}' → '{label_cls}' (shape override)")
                        else:
                            if self.print_logs:
                                print(f"[SHAPE-OVERRIDE] DISABLED - keeping original: '{label_cls}'")

                    st["ema_prob"] = prob1 if st.get("ema_prob") is None else (0.6*prob1 + 0.4*st["ema_prob"])
                    margin = max(0.0, prob1 - prob2)
                    pmargin = max(0.0, st["ema_prob"] - prob2)
                    if st["last_label"] == label_cls: st["run"] += 1
                    else: st["run"] = 1; st["last_label"] = label_cls

                    # ← TRACKING PREDICTION
                    self._track_prediction(obj_id, label_cls, prob1, margin, was_pressed=False)

                    # Debug info
                    print(f"[DEBUG] Processing ID{obj_id}: AI predicted '{label_cls}' (conf: {prob1:.2f}) difficulty: {self.get_char_difficulty(label_cls)}")

                    # Check if needs confirmation với obj_id tracking
                    final_label = label_cls
                    if (self.hitl_enable and glyph is not None):
                        final_label = self.adaptive_learner.should_confirm_non_blocking(
                            label_cls, prob1, margin, {}, obj_id, box, glyph
                        )

                    # Add retry check
                    if frames % 300 == 0:
                        self.adaptive_learner.check_retry_requests()

                    tm_view = glyph if glyph is not None else (crops[-1] if crops else None)
                    tm_best_label, tm_best = self.tbank.best_match(tm_view) if tm_view is not None else (None, 0.0)
                    tm_same = self.tbank.score(final_label, tm_view) if (final_label and tm_view is not None) else 0.0
                    fused = self.alpha_cls*st["ema_prob"] + (1.0-self.alpha_cls)*(tm_same if tm_same>0 else 0.0)

                    if (tm_best_label is not None and final_label and (tm_best_label != final_label) and
                        tm_best >= max(0.85, self.template_veto) and margin >= max(0.22, self.margin_thr)):
                        self._record_conflict(tm_best_label)

                    st.update({"label": final_label, "last_prob": st["ema_prob"], "last_margin": margin,
                            "last_pmargin": pmargin, "tm": tm_same, "fused": fused})

                    enriched.append({
                        "obj_id": obj_id, "box": box, "st": st, "tm_view": tm_view,
                        "label_cls": final_label, "prob1": prob1, "prob2": prob2,
                        "tm_best_label": tm_best_label, "tm_best": tm_best,
                        "tm_same": tm_same, "fused": fused
                    })

                # ← ENHANCED DETECTION DEBUGGING
                if self.print_logs and len(enriched) > 0:
                    detected_chars = [(e["label_cls"], f"{e['st']['ema_prob']:.2f}") for e in enriched]
                    print(f"[DETECT] Frame detected: {detected_chars}")
                    
                    # ← CHECK FOR MISSING CHARACTERS
                    detected_labels = [e["label_cls"] for e in enriched]
                    missing_chars = []
                    
                    # Check for common missing characters
                    for expected_char in ['w', 'm', 'z', 'k']:
                        if expected_char not in detected_labels:
                            # Check if it was originally classified as something else
                            for debug_info in classification_debug:
                                if debug_info["original"] in ['u', 'v', 'm', 'n', 'h'] and debug_info["conf"] > 0.3:
                                    print(f"[MISSING-DEBUG] 🔍 ID{debug_info['obj_id']}: '{debug_info['original']}' (conf: {debug_info['conf']:.2f}) - Could this be '{expected_char}'?")
                            
                            # Check wide aspect ratios for 'w' and 'm'
                            if expected_char in ['w', 'm']:
                                for e in enriched:
                                    x1, y1, x2, y2 = e["box"]
                                    ar = (x2-x1) / max(1, y2-y1)
                                    if ar >= 1.3 and e["label_cls"] in ['u', 'v', 'h']:
                                        print(f"[MISSING-DEBUG] 🔍 ID{e['obj_id']}: '{e['label_cls']}' has wide AR={ar:.2f} - Possible '{expected_char}'?")

                # ← SAVE CLASSIFICATION DEBUG CROPS
                if self.print_logs and classification_debug:
                    try:
                        timestamp = int(time.time() * 1000) % 1000000
                        for debug_info in classification_debug:
                            obj_id = debug_info["obj_id"]
                            original = debug_info["original"]
                            conf = debug_info["conf"]
                            box = debug_info["box"]
                            
                            x1, y1, x2, y2 = box
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                debug_path = f"C:\\Users\\DatGo\\cauca\\debugs\\classify\\debug_classify_ID{obj_id}_{original}_{conf:.2f}_{timestamp}.png"
                                cv2.imwrite(debug_path, crop)
                                print(f"[DEBUG-CLASSIFY] 📸 Saved: {debug_path}")
                    except Exception as e:
                        print(f"[DEBUG-CLASSIFY] ❌ Error saving classification crops: {e}")

                pressed_this_frame = 0

                # Template FORCE với tracking
                force_cands = [e for e in enriched if (e["tm_best_label"] is not None and e["tm_best"] >= self.template_force and e["st"]["det_ema"] >= self.det_gate_tm and not self._region_locked(e["box"]))]
                if force_cands:
                    def template_priority(e):
                        char = e["tm_best_label"]
                        score = e["tm_best"]
                        
                        if char == "1" and e["label_cls"] in ["r", "l"]:
                            score += 0.1
                        elif char == "r" and e["label_cls"] == "1":
                            score -= 0.05
                        elif char == "w":
                            score += 0.05
                            
                        return (score, e["st"]["det_ema"])
                    
                    best = max(force_cands, key=template_priority)
                    
                    # ← FIX: ADD obj_id parameter
                    if self._press(best["tm_best_label"], best["box"], max(best["fused"], best["tm_best"]), 
                                best["st"]["det_ema"], "tm_force", obj_id=best["obj_id"]):
                        best["st"]["pressed"] += 1
                        best["st"]["pressed_at"] = time.time()
                        self.tbank.add(best["tm_best_label"], best["tm_view"], ts=time.time())
                        
                        self._track_prediction(best["obj_id"], best["tm_best_label"], best["st"]["ema_prob"], 
                                            best["st"]["last_margin"], was_pressed=True)
                        
                        pressed_this_frame += 1

                # Main with smart thresholds
                if pressed_this_frame == 0:
                    cands = []
                    for e in enriched:
                        if self.should_press_with_smart_thresholds(e):
                            cands.append(e)
                    
                    def smart_score(e):
                        char = e["label_cls"]
                        difficulty = self.get_char_difficulty(char)
                        
                        difficulty_bonus = 0.1 if difficulty == "easy" else (-0.05 if difficulty == "hard" else 0.0)
                        w_bonus = 0.05 if char == "w" else 0.0
                        
                        return e["fused"] + e["st"]["det_ema"] + difficulty_bonus + w_bonus
                    
                    cands.sort(key=smart_score, reverse=True)
                    
                    for e in cands:
                        if pressed_this_frame >= self.max_presses_per_frame:
                            break
                        # ← FIX: ADD obj_id parameter
                        if self._press(e["label_cls"], e["box"], e["fused"], e["st"]["det_ema"], 
                                    "smart_main", obj_id=e["obj_id"]):
                            e["st"]["pressed"] += 1
                            e["st"]["pressed_at"] = time.time()
                            
                            self._track_prediction(e["obj_id"], e["label_cls"], e["st"]["ema_prob"], 
                                                e["st"]["last_margin"], was_pressed=True)
                            
                            pressed_this_frame += 1

                # Seed bank
                for e in enriched:
                    st = e["st"]
                    if (st["run"] >= 2) and (st["det_ema"] >= self.tm_seed_det) and (st["last_margin"] >= max(0.22, self.margin_thr)):
                        if e["tm_view"] is not None and not seed_veto(e["label_cls"], e["tm_view"]):
                            if self.tbank.score(e["label_cls"], e["tm_view"]) < 0.3:
                                self.tbank.add(e["label_cls"], e["tm_view"], ts=time.time())

                # Performance tracking
                if frames % 60 == 0:
                    self._track_fps_performance()
                    
            self._draw(frame, items)

        # Final performance summary
        if hasattr(self, 'accuracy_tracker') and self.accuracy_tracker["total_predictions"] > 0:
            print(f"\n[FINAL] 📊 Session Summary:")
            print(f"[FINAL] Total predictions: {self.accuracy_tracker['total_predictions']}")
            print(f"[FINAL] Total presses: {len(self.accuracy_tracker['press_history'])}")
            if self.accuracy_tracker["press_history"]:
                recent_presses = list(self.accuracy_tracker["press_history"])
                avg_press_conf = np.mean([p["confidence"] for p in recent_presses])
                print(f"[FINAL] Avg press confidence: {avg_press_conf:.2f}")
            print(f"[FINAL] 🎯 Session complete!")
        self.cam.stop()
        cv2.destroyAllWindows()

    # ---- Overlay ----
    def _draw(self, frame, items):
        if not self.show: return
        now = time.time()
        if (now - self.last_overlay_ts) < (1.0 / max(1, self.overlay_fps)):
            return
        self.last_overlay_ts = now
        disp = np.ascontiguousarray(frame)
        
        for obj_id, (x1,y1,x2,y2), st in items:
            if self.green_mode == "det":
                ok = (st.get("det_ema",0) >= self.det_gate)
            else:
                ok = (st.get("fused",0) >= self.fused_thr and st.get("det_ema",0) >= self.det_gate)
            color = (0,255,0) if ok else (0,200,255)
            cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
            txt = f"id{obj_id} {st.get('label','?')} f:{st.get('fused',0):.2f} tm:{st.get('tm',0):.2f} d:{st.get('det_ema',0):.2f}"
            cv2.putText(disp, txt, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.53, color, 2)
        
        if self.draw_exclude:
            for (ex,ey,ew,eh) in self.exclude_zones:
                cv2.rectangle(disp, (ex,ey), (ex+ew, ey+eh), (255,0,0), 2)
                cv2.putText(disp, "EX", (ex, max(0,ey-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0), 1)
        
        cv2.putText(disp, f"FPS:{self.fps:.1f} Active:{self.active}", (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        
        # ← THÊM SETUP WINDOW Ở ĐÂY:
        if not hasattr(self, 'window_created') or not self.window_created:
            cv2.namedWindow("auto-qte", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("auto-qte", 1200, 400)  # Kích thước ban đầu
            cv2.moveWindow("auto-qte", 100, 100)     # Vị trí ban đầu
            self.window_created = True
            print("[WINDOW] Created resizable window")
        
        cv2.imshow("auto-qte", disp)
        if cv2.waitKey(1) & 0xFF == 27:
            self.stop = True

    # ---- Hotkeys ----
    def _hotkeys(self):
        def on_press(key):
            try:
                if key == keyboard.Key.f8:
                    self.active = not self.active
                    print(f"[STATE] Active = {self.active}")
                elif key == keyboard.Key.f10:
                    # ← SỬA F10 THÀNH TOGGLE SHAPE OVERRIDE
                    self.shape_override_enabled = not self.shape_override_enabled
                    print(f"[SHAPE] 🔧 Shape Override = {self.shape_override_enabled}")
                    status = "ENABLED" if self.shape_override_enabled else "DISABLED"
                    print(f"[SHAPE] 🎛️ Shape Override is now {status}")
                elif key == keyboard.Key.f11:  # ← F11 CHO THRESHOLDS INFO
                    print(f"[STATE] det>={self.det_gate:.2f}/{self.det_gate_tm:.2f} hits={self.hits_req} run={self.run_req} fused>={self.fused_thr} margin>={self.margin_thr} pmargin>={self.pmargin_thr} tm_veto>={self.template_veto} tm_force>={self.template_force}")
                elif key == keyboard.Key.f7:
                    self.tbank.clear(); print("[BANK] cleared manually")
                elif key == keyboard.Key.f9:
                    print(f"[STATE] Exit requested"); self.stop = True; return False
                elif key == keyboard.Key.f1:
                    # Manual correction mode
                    print("[HITL] Manual correction mode - next detection will ask for confirmation")
                    # Set a flag to force confirmation on next detection
                    if hasattr(self, 'force_next_confirmation'):
                        self.force_next_confirmation = True
            except Exception as e:
                print("Hotkey error:", e)
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

if __name__ == "__main__":
    cfg = load_cfg(resource_path("config.yaml"))
    AutoQTE(cfg).run()