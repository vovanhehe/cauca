#!/usr/bin/env python3
"""
Performance test for SORT tracking integration.
Measures the overhead of SORT tracking vs simple index-based tracking.
"""

import time
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from sort import Sort

def test_iou(a, b):
    """IoU function from auto_qte.py"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6
    return inter / ua

def generate_random_detections(num_objects=5, frame_width=1920, frame_height=640):
    """Generate random bubble detections for testing"""
    detections = []
    for _ in range(num_objects):
        x1 = np.random.randint(0, frame_width - 100)
        y1 = np.random.randint(0, frame_height - 100) 
        x2 = x1 + np.random.randint(50, 100)
        y2 = y1 + np.random.randint(50, 100)
        conf = np.random.uniform(0.7, 0.95)
        detections.append([x1, y1, x2, y2, conf])
    return np.array(detections, dtype=np.float32)

def simulate_simple_tracking(detections_list):
    """Simulate the old simple index-based tracking"""
    start_time = time.time()
    
    for frame_idx, detections in enumerate(detections_list):
        # Simple index-based tracking (old method)
        for idx, det in enumerate(detections):
            obj_id = idx  # This was the problem - unstable IDs
            # Simulate some processing
            pass
    
    end_time = time.time()
    return end_time - start_time

def simulate_sort_tracking(detections_list):
    """Simulate the new SORT-based tracking"""
    start_time = time.time()
    
    sort_tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.2)
    
    for frame_idx, detections in enumerate(detections_list):
        # SORT tracking (new method)
        if len(detections) > 0:
            tracked_objects = sort_tracker.update(detections)
            
            # Simulate matching SORT results back to detections
            for track in tracked_objects:
                track_box = (int(track[0]), int(track[1]), int(track[2]), int(track[3]))
                sort_obj_id = int(track[4])
                
                # Find best matching detection
                best_iou = 0.0
                for det_idx, det in enumerate(detections):
                    det_box = (int(det[0]), int(det[1]), int(det[2]), int(det[3]))
                    overlap = test_iou(track_box, det_box)
                    if overlap > best_iou:
                        best_iou = overlap
                
                # Simulate processing with stable ID
                obj_id = sort_obj_id
        else:
            # Empty frame
            sort_tracker.update(np.empty((0, 5)))
    
    end_time = time.time()
    return end_time - start_time

def test_performance():
    """Test performance impact of SORT vs simple tracking"""
    print("🚀 Testing SORT tracking performance impact...")
    print("=" * 60)
    
    # Generate test data - simulate 120 frames (2 seconds at 60fps)
    num_frames = 120
    detections_per_frame = 5
    
    print(f"Generating {num_frames} frames with {detections_per_frame} detections each...")
    
    detections_list = []
    for frame_idx in range(num_frames):
        # Simulate varying number of detections (0-8 objects)
        num_dets = np.random.randint(0, 9)
        if num_dets > 0:
            dets = generate_random_detections(num_dets)
        else:
            dets = np.empty((0, 5), dtype=np.float32)
        detections_list.append(dets)
    
    # Test simple tracking performance
    print("\n📊 Testing simple index-based tracking (old method)...")
    simple_time = simulate_simple_tracking(detections_list)
    simple_fps = num_frames / simple_time
    print(f"Simple tracking: {simple_time:.4f}s for {num_frames} frames")
    print(f"Simple tracking FPS: {simple_fps:.1f}")
    
    # Test SORT tracking performance  
    print("\n📊 Testing SORT-based tracking (new method)...")
    sort_time = simulate_sort_tracking(detections_list)
    sort_fps = num_frames / sort_time
    print(f"SORT tracking: {sort_time:.4f}s for {num_frames} frames")
    print(f"SORT tracking FPS: {sort_fps:.1f}")
    
    # Calculate overhead
    overhead = ((sort_time - simple_time) / simple_time) * 100
    fps_impact = simple_fps - sort_fps
    
    print("\n📈 Performance Analysis:")
    print(f"Time overhead: {overhead:.1f}%")
    print(f"FPS impact: -{fps_impact:.1f} FPS")
    
    # Performance evaluation
    if overhead < 10:
        print("✅ EXCELLENT: Minimal performance impact (<10% overhead)")
    elif overhead < 25:
        print("✅ GOOD: Acceptable performance impact (<25% overhead)")
    elif overhead < 50:
        print("⚠️ MODERATE: Noticeable performance impact (<50% overhead)")
    else:
        print("❌ HIGH: Significant performance impact (>50% overhead)")
    
    # FPS evaluation for game requirements
    target_fps = 60
    if sort_fps >= target_fps:
        print(f"🎯 TARGET MET: SORT tracking achieves {sort_fps:.1f} FPS (≥{target_fps} FPS)")
    else:
        print(f"⚠️ TARGET MISSED: SORT tracking achieves {sort_fps:.1f} FPS (<{target_fps} FPS)")
    
    print("\n💡 Key Benefits of SORT (regardless of small overhead):")
    print("  ✅ Eliminates duplicate key presses (CRITICAL for gameplay)")
    print("  ✅ Stable object IDs throughout bubble lifecycle")
    print("  ✅ Better handling of fast-moving bubbles")
    print("  ✅ Support for duplicate characters (a, a, t, 2, 7)")
    print("  ✅ Kalman filter prediction for motion handling")
    
    return sort_fps >= 30  # Minimum acceptable FPS

if __name__ == "__main__":
    success = test_performance()
    print("\n" + "=" * 60)
    if success:
        print("🎉 Performance test PASSED!")
    else:
        print("❌ Performance test FAILED - FPS too low")
    sys.exit(0 if success else 1)