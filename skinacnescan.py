"""
=================================================================
  ADVANCED SKIN ACNE SCANNER v2.0
  Real-time face skin analysis with acne/blemish detection
  Using OpenCV + MediaPipe FaceLandmarker (Tasks API)
=================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import os
import urllib.request
from collections import deque

# ─────────────────────────────────────────────────────────
# MODEL DOWNLOAD
# ─────────────────────────────────────────────────────────

FACE_MODEL_PATH = "face_landmarker.task"

def download_model():
    """Download Face Landmarker model if not present."""
    if not os.path.exists(FACE_MODEL_PATH):
        print("[*] Mengunduh model Face Landmarker...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        urllib.request.urlretrieve(url, FACE_MODEL_PATH)
        print("[+] Model berhasil diunduh!")
    else:
        print("[+] Model sudah ada.")


# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

class Config:
    """All tunable parameters in one place."""
    
    # Camera
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    
    # Face Landmarker
    FACE_CONFIDENCE = 0.6
    FACE_TRACKING = 0.6
    
    # Color thresholds for acne/redness detection (in HSV)
    RED_LOW_H1, RED_HIGH_H1 = 0, 12
    RED_LOW_H2, RED_HIGH_H2 = 165, 180
    RED_SAT_MIN = 40
    RED_VAL_MIN = 60
    RED_VAL_MAX = 240
    
    # Acne blob detection min/max area (pixels)
    ACNE_MIN_AREA = 25
    ACNE_MAX_AREA = 2500
    ACNE_MIN_CIRCULARITY = 0.25
    
    # Severity thresholds
    MILD_THRESHOLD = 8
    MODERATE_THRESHOLD = 20
    SEVERE_THRESHOLD = 40
    
    # Smoothing (temporal averaging)
    HISTORY_SIZE = 15
    
    # UI Colors (BGR)
    COLOR_BG = (20, 20, 20)
    COLOR_PANEL = (35, 35, 40)
    COLOR_ACCENT = (255, 160, 50)      # Orange-amber
    COLOR_ACCENT2 = (100, 220, 255)    # Cyan
    COLOR_GREEN = (80, 220, 100)
    COLOR_YELLOW = (50, 220, 255)
    COLOR_ORANGE = (30, 140, 255)
    COLOR_RED = (60, 60, 255)
    COLOR_WHITE = (240, 240, 240)
    COLOR_GRAY = (140, 140, 140)
    COLOR_DARK_GRAY = (80, 80, 80)
    COLOR_BORDER = (60, 60, 65)
    
    ZONE_NAMES = {
        'forehead': 'Dahi',
        'left_cheek': 'Pipi Kiri',
        'right_cheek': 'Pipi Kanan',
        'nose': 'Hidung',
        'chin': 'Dagu',
        'left_jaw': 'Rahang Kiri',
        'right_jaw': 'Rahang Kanan',
    }


# ─────────────────────────────────────────────────────────
# FACE ZONE LANDMARKS (MediaPipe Face Mesh indices)
# ─────────────────────────────────────────────────────────

# Face outline for skin mask
FACE_OUTLINE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132,
    93, 234, 127, 162, 21, 54, 103, 67, 109
]

# Exclusion zones (eyes, mouth, eyebrows)
EYE_LEFT_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                    173, 157, 158, 159, 160, 161, 246]
EYE_RIGHT_INDICES = [362, 382, 381, 380, 374, 373, 390, 249,
                     263, 466, 388, 387, 386, 385, 384, 398]
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
EYEBROW_LEFT = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
EYEBROW_RIGHT = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

# Zone polygons for per-area analysis
ZONE_LANDMARKS = {
    'forehead': [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109],
    'left_cheek': [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152],
    'right_cheek': [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152],
    'nose': [168, 6, 197, 195, 5, 4, 1, 19, 94, 2],
    'chin': [18, 200, 199, 175, 152, 377, 400, 378, 379, 365, 397],
    'left_jaw': [234, 127, 162, 21, 54, 103, 67, 109, 10],
    'right_jaw': [454, 323, 361, 288, 397, 365, 379, 378, 400],
}


# ─────────────────────────────────────────────────────────
# ACNE DETECTION ENGINE
# ─────────────────────────────────────────────────────────

class AcneDetector:
    """
    Multi-method acne detection combining:
    1. Color-based redness detection (HSV)
    2. Texture analysis (Laplacian/gradient-based)
    3. Blob detection with circularity filtering
    4. Adaptive thresholding for dark spots
    5. LAB color anomaly detection (a-channel)
    """
    
    def __init__(self):
        self.history = deque(maxlen=Config.HISTORY_SIZE)
        self.zone_history = {zone: deque(maxlen=Config.HISTORY_SIZE) for zone in ZONE_LANDMARKS}
        self.smoothed_count = 0
        self.smoothed_severity = "Memindai..."
        self.detection_results = []
        self.zone_results = {}
        self.skin_health_score = 100
        self.skin_health_history = deque(maxlen=Config.HISTORY_SIZE)
        self.frame_count = 0
        
    def detect_redness(self, face_roi, mask):
        """Detect red/inflamed regions typical of acne."""
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([Config.RED_LOW_H1, Config.RED_SAT_MIN, Config.RED_VAL_MIN])
        upper_red1 = np.array([Config.RED_HIGH_H1, 255, Config.RED_VAL_MAX])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        lower_red2 = np.array([Config.RED_LOW_H2, Config.RED_SAT_MIN, Config.RED_VAL_MIN])
        upper_red2 = np.array([Config.RED_HIGH_H2, 255, Config.RED_VAL_MAX])
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)
        red_mask = cv2.bitwise_and(red_mask, mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return red_mask
    
    def detect_texture_anomalies(self, face_roi, mask):
        """Detect texture irregularities (bumps) via Laplacian."""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        laplacian_abs = np.uint8(np.absolute(laplacian))
        
        _, texture_mask = cv2.threshold(laplacian_abs, 30, 255, cv2.THRESH_BINARY)
        texture_mask = cv2.bitwise_and(texture_mask, mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return texture_mask
    
    def detect_dark_spots(self, face_roi, mask):
        """Detect dark spots / hyperpigmentation."""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 8
        )
        adaptive = cv2.bitwise_and(adaptive, mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return adaptive
    
    def detect_color_anomalies(self, face_roi, mask):
        """Detect color anomalies via LAB a-channel (redness deviation)."""
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        _, a_ch, _ = cv2.split(lab)
        
        skin_pixels = a_ch[mask > 0]
        if len(skin_pixels) < 100:
            return np.zeros(mask.shape, dtype=np.uint8)
        
        mean_a = np.mean(skin_pixels)
        std_a = np.std(skin_pixels)
        
        if std_a < 1:
            return np.zeros(mask.shape, dtype=np.uint8)
        
        threshold_val = mean_a + 1.5 * std_a
        anomaly_mask = np.zeros(mask.shape, dtype=np.uint8)
        anomaly_mask[a_ch > threshold_val] = 255
        anomaly_mask = cv2.bitwise_and(anomaly_mask, mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return anomaly_mask
    
    def find_acne_blobs(self, combined_mask):
        """Find individual acne spots from combined detection mask."""
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        acne_spots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < Config.ACNE_MIN_AREA or area > Config.ACNE_MAX_AREA:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            if circularity < Config.ACNE_MIN_CIRCULARITY:
                continue
            
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            if area < 80:
                severity = 'mild'
            elif area < 400:
                severity = 'moderate'
            else:
                severity = 'severe'
            
            acne_spots.append({
                'center': (int(cx), int(cy)),
                'radius': max(int(radius), 3),
                'area': area,
                'circularity': circularity,
                'bbox': (x, y, w, h),
                'severity': severity,
                'contour': contour,
            })
        
        return acne_spots
    
    def classify_spot_to_zone(self, spot_center, zone_polygons):
        """Determine which face zone an acne spot belongs to."""
        for zone_name, polygon in zone_polygons.items():
            if polygon is not None and len(polygon) > 2:
                result = cv2.pointPolygonTest(polygon, spot_center, False)
                if result >= 0:
                    return zone_name
        return 'other'
    
    def get_landmark_point(self, landmarks, idx, iw, ih, offset_x=0, offset_y=0):
        """Get pixel coordinates from a NormalizedLandmark."""
        lm = landmarks[idx]
        return (int(lm.x * iw) - offset_x, int(lm.y * ih) - offset_y)
    
    def analyze_frame(self, frame, landmarks, face_bbox):
        """Full analysis pipeline for one frame."""
        self.frame_count += 1
        
        x, y, w, h = face_bbox
        pad_x = int(w * 0.05)
        pad_y = int(h * 0.05)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)
        
        face_roi = frame[y1:y2, x1:x2].copy()
        if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
            return [], {}
        
        ih, iw = frame.shape[:2]
        
        # Build face skin mask from outline landmarks
        face_points = []
        for idx in FACE_OUTLINE_INDICES:
            px, py = self.get_landmark_point(landmarks, idx, iw, ih, x1, y1)
            face_points.append([px, py])
        
        face_points = np.array(face_points, dtype=np.int32)
        skin_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillConvexPoly(skin_mask, face_points, 255)
        
        # Exclude eyes, mouth, eyebrows
        for indices in [EYE_LEFT_INDICES, EYE_RIGHT_INDICES, MOUTH_INDICES,
                       EYEBROW_LEFT, EYEBROW_RIGHT]:
            pts = []
            for idx in indices:
                px, py = self.get_landmark_point(landmarks, idx, iw, ih, x1, y1)
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)
            cv2.fillConvexPoly(skin_mask, pts, 0)
        
        # Slight erosion to remove boundary artifacts
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.erode(skin_mask, kernel_erode, iterations=1)
        
        # Run all detection methods
        red_mask = self.detect_redness(face_roi, skin_mask)
        texture_mask = self.detect_texture_anomalies(face_roi, skin_mask)
        dark_mask = self.detect_dark_spots(face_roi, skin_mask)
        color_anomaly = self.detect_color_anomalies(face_roi, skin_mask)
        
        # Combine masks with weighted importance
        red_and_texture = cv2.bitwise_and(red_mask, texture_mask)
        red_and_color = cv2.bitwise_and(red_mask, color_anomaly)
        strong_signal = cv2.bitwise_or(red_and_texture, red_and_color)
        
        medium_signal = cv2.bitwise_or(red_mask, color_anomaly)
        
        # Dark spots near redness
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        red_expanded = cv2.dilate(red_mask, kernel_dilate, iterations=1)
        dark_near_red = cv2.bitwise_and(dark_mask, red_expanded)
        
        combined = cv2.bitwise_or(strong_signal, medium_signal)
        combined = cv2.bitwise_or(combined, dark_near_red)
        
        # Final morphological cleanup
        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_final, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_final, iterations=2)
        
        # Find acne blobs
        acne_spots = self.find_acne_blobs(combined)
        
        # Map spots back to frame coordinates
        for spot in acne_spots:
            cx, cy = spot['center']
            spot['center'] = (cx + x1, cy + y1)
            bx, by, bw, bh = spot['bbox']
            spot['bbox'] = (bx + x1, by + y1, bw, bh)
            spot['contour'] = spot['contour'] + np.array([x1, y1])
        
        # Build zone polygons
        zone_polygons = {}
        for zone_name, indices in ZONE_LANDMARKS.items():
            pts = []
            for idx in indices:
                px, py = self.get_landmark_point(landmarks, idx, iw, ih)
                pts.append([px, py])
            zone_polygons[zone_name] = np.array(pts, dtype=np.int32)
        
        # Classify spots into zones
        zone_counts = {z: 0 for z in ZONE_LANDMARKS.keys()}
        for spot in acne_spots:
            zone = self.classify_spot_to_zone(
                (float(spot['center'][0]), float(spot['center'][1])),
                zone_polygons
            )
            spot['zone'] = zone
            if zone in zone_counts:
                zone_counts[zone] += 1
        
        # Update history
        count = len(acne_spots)
        self.history.append(count)
        self.smoothed_count = int(np.mean(self.history))
        
        for zone, cnt in zone_counts.items():
            self.zone_history[zone].append(cnt)
        
        # Update severity
        if self.smoothed_count == 0:
            self.smoothed_severity = "Bersih"
            health = 100
        elif self.smoothed_count <= Config.MILD_THRESHOLD:
            self.smoothed_severity = "Ringan"
            health = max(70, 100 - self.smoothed_count * 3)
        elif self.smoothed_count <= Config.MODERATE_THRESHOLD:
            self.smoothed_severity = "Sedang"
            health = max(40, 70 - (self.smoothed_count - Config.MILD_THRESHOLD) * 2)
        elif self.smoothed_count <= Config.SEVERE_THRESHOLD:
            self.smoothed_severity = "Parah"
            health = max(15, 40 - (self.smoothed_count - Config.MODERATE_THRESHOLD))
        else:
            self.smoothed_severity = "Sangat Parah"
            health = max(5, 15 - (self.smoothed_count - Config.SEVERE_THRESHOLD))
        
        self.skin_health_history.append(health)
        self.skin_health_score = int(np.mean(self.skin_health_history))
        
        self.detection_results = acne_spots
        self.zone_results = zone_counts
        
        return acne_spots, zone_counts


# ─────────────────────────────────────────────────────────
# UI RENDERER
# ─────────────────────────────────────────────────────────

class UIRenderer:
    """Premium UI rendering for the scanner interface."""
    
    def __init__(self):
        self.animation_phase = 0
        self.scan_line_y = 0
        
    def draw_rounded_rect(self, img, pt1, pt2, color, radius=12, thickness=-1, alpha=1.0):
        """Draw a rounded rectangle with optional transparency."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        if alpha < 1.0:
            overlay = img.copy()
            cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
            cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
            cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
            cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
            cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
            cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        else:
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)
    
    def draw_acne_markers(self, frame, spots):
        """Draw detection markers on detected acne spots."""
        for spot in spots:
            cx, cy = spot['center']
            r = spot['radius']
            severity = spot['severity']
            
            if severity == 'mild':
                color = Config.COLOR_YELLOW
                ring_color = (80, 200, 255)
            elif severity == 'moderate':
                color = Config.COLOR_ORANGE
                ring_color = (40, 160, 255)
            else:
                color = Config.COLOR_RED
                ring_color = (50, 50, 255)
            
            pulse_r = r + 4 + int(3 * math.sin(self.animation_phase * 0.1))
            
            # Outer glow
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), pulse_r + 6, ring_color, 2)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # Main ring
            cv2.circle(frame, (cx, cy), pulse_r, ring_color, 2)
            
            # Inner dot
            cv2.circle(frame, (cx, cy), max(2, r // 2), color, -1)
            
            # Crosshair
            line_len = pulse_r + 6
            cv2.line(frame, (cx - line_len, cy), (cx - pulse_r - 2, cy), ring_color, 1)
            cv2.line(frame, (cx + pulse_r + 2, cy), (cx + line_len, cy), ring_color, 1)
            cv2.line(frame, (cx, cy - line_len), (cx, cy - pulse_r - 2), ring_color, 1)
            cv2.line(frame, (cx, cy + pulse_r + 2), (cx, cy + line_len), ring_color, 1)
    
    def draw_scan_line(self, frame, face_bbox):
        """Draw animated scanning line across face."""
        if face_bbox is None:
            return
        x, y, w, h = face_bbox
        
        self.scan_line_y = (self.scan_line_y + 3) % h
        scan_y = y + self.scan_line_y
        
        overlay = frame.copy()
        cv2.line(overlay, (x, scan_y), (x + w, scan_y), Config.COLOR_ACCENT2, 2)
        
        for i in range(1, 8):
            alpha_val = 0.15 - i * 0.02
            if alpha_val > 0:
                color = tuple(max(0, int(c * (1 - i * 0.1))) for c in Config.COLOR_ACCENT2)
                cv2.line(overlay, (x, scan_y + i), (x + w, scan_y + i), color, 1)
                cv2.line(overlay, (x, scan_y - i), (x + w, scan_y - i), color, 1)
        
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    def draw_face_mesh_overlay(self, frame, landmarks, ih, iw):
        """Draw subtle face mesh lines using tesselation connections."""
        connections = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
        
        overlay = frame.copy()
        for connection in connections:
            idx1 = connection.start
            idx2 = connection.end
            lm1 = landmarks[idx1]
            lm2 = landmarks[idx2]
            p1 = (int(lm1.x * iw), int(lm1.y * ih))
            p2 = (int(lm2.x * iw), int(lm2.y * ih))
            cv2.line(overlay, p1, p2, (100, 200, 180), 1)
        
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    
    def draw_info_panel(self, frame, detector):
        """Draw the main information panel on the right side."""
        h, w = frame.shape[:2]
        panel_w = 340
        panel_x = w - panel_w - 15
        panel_y = 15
        
        # Panel background
        self.draw_rounded_rect(frame, (panel_x, panel_y),
                              (panel_x + panel_w, h - 15),
                              Config.COLOR_PANEL, radius=16, alpha=0.85)
        
        # Border
        self.draw_rounded_rect(frame, (panel_x, panel_y),
                              (panel_x + panel_w, h - 15),
                              Config.COLOR_BORDER, radius=16, thickness=1)
        
        # Title
        title_y = panel_y + 35
        cv2.putText(frame, "SKIN SCANNER", (panel_x + 20, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_ACCENT, 2)
        cv2.putText(frame, "v2.0", (panel_x + panel_w - 60, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, Config.COLOR_GRAY, 1)
        
        # Divider
        div_y = title_y + 15
        cv2.line(frame, (panel_x + 20, div_y), (panel_x + panel_w - 20, div_y),
                Config.COLOR_BORDER, 1)
        
        # Health Score - circular gauge
        gauge_cx = panel_x + panel_w // 2
        gauge_cy = div_y + 70
        gauge_r = 45
        
        cv2.ellipse(frame, (gauge_cx, gauge_cy), (gauge_r, gauge_r),
                   0, 135, 405, Config.COLOR_DARK_GRAY, 4)
        
        score = detector.skin_health_score
        arc_angle = int(270 * score / 100)
        if score >= 70:
            gauge_color = Config.COLOR_GREEN
        elif score >= 40:
            gauge_color = Config.COLOR_YELLOW
        elif score >= 20:
            gauge_color = Config.COLOR_ORANGE
        else:
            gauge_color = Config.COLOR_RED
        
        cv2.ellipse(frame, (gauge_cx, gauge_cy), (gauge_r, gauge_r),
                   0, 135, 135 + arc_angle, gauge_color, 5)
        
        score_text = f"{score}"
        text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.putText(frame, score_text,
                   (gauge_cx - text_size[0] // 2, gauge_cy + text_size[1] // 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, Config.COLOR_WHITE, 2)
        cv2.putText(frame, "SKOR",
                   (gauge_cx - 18, gauge_cy + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, Config.COLOR_GRAY, 1)
        
        # ─── Stats section ───
        stats_y = gauge_cy + gauge_r + 35
        
        severity = detector.smoothed_severity
        sev_color = Config.COLOR_GREEN
        if severity == "Ringan":
            sev_color = Config.COLOR_YELLOW
        elif severity == "Sedang":
            sev_color = Config.COLOR_ORANGE
        elif severity in ("Parah", "Sangat Parah"):
            sev_color = Config.COLOR_RED
        
        cv2.putText(frame, "Status:", (panel_x + 20, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.COLOR_GRAY, 1)
        cv2.putText(frame, severity, (panel_x + 100, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, sev_color, 2)
        
        stats_y += 30
        cv2.putText(frame, "Jerawat:", (panel_x + 20, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.COLOR_GRAY, 1)
        cv2.putText(frame, f"{detector.smoothed_count} titik",
                   (panel_x + 100, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, Config.COLOR_WHITE, 1)
        
        mild_count = sum(1 for s in detector.detection_results if s['severity'] == 'mild')
        mod_count = sum(1 for s in detector.detection_results if s['severity'] == 'moderate')
        sev_count = sum(1 for s in detector.detection_results if s['severity'] == 'severe')
        
        stats_y += 30
        cv2.putText(frame, f"  Ringan: {mild_count}", (panel_x + 25, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, Config.COLOR_YELLOW, 1)
        cv2.putText(frame, f"Sedang: {mod_count}", (panel_x + 140, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, Config.COLOR_ORANGE, 1)
        cv2.putText(frame, f"Parah: {sev_count}", (panel_x + 250, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, Config.COLOR_RED, 1)
        
        # ─── Divider ───
        stats_y += 20
        cv2.line(frame, (panel_x + 20, stats_y), (panel_x + panel_w - 20, stats_y),
                Config.COLOR_BORDER, 1)
        
        # ─── Zone Analysis ───
        stats_y += 25
        cv2.putText(frame, "ANALISIS ZONA", (panel_x + 20, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.COLOR_ACCENT2, 1)
        
        for zone_key, zone_label in Config.ZONE_NAMES.items():
            stats_y += 25
            if zone_key in detector.zone_history and len(detector.zone_history[zone_key]) > 0:
                avg = int(np.mean(detector.zone_history[zone_key]))
            else:
                avg = 0
            
            cv2.putText(frame, zone_label, (panel_x + 25, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, Config.COLOR_GRAY, 1)
            
            bar_x = panel_x + 130
            bar_w = 140
            bar_h = 10
            bar_y_top = stats_y - 9
            
            cv2.rectangle(frame, (bar_x, bar_y_top),
                         (bar_x + bar_w, bar_y_top + bar_h),
                         Config.COLOR_DARK_GRAY, -1)
            
            fill_ratio = min(avg / 10, 1.0)
            fill_w = int(bar_w * fill_ratio)
            if fill_ratio > 0:
                if fill_ratio < 0.3:
                    bar_color = Config.COLOR_GREEN
                elif fill_ratio < 0.6:
                    bar_color = Config.COLOR_YELLOW
                else:
                    bar_color = Config.COLOR_RED
                cv2.rectangle(frame, (bar_x, bar_y_top),
                             (bar_x + fill_w, bar_y_top + bar_h),
                             bar_color, -1)
            
            cv2.putText(frame, str(avg), (bar_x + bar_w + 10, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, Config.COLOR_WHITE, 1)
        
        # ─── Divider ───
        stats_y += 25
        cv2.line(frame, (panel_x + 20, stats_y), (panel_x + panel_w - 20, stats_y),
                Config.COLOR_BORDER, 1)
        
        # ─── Recommendations ───
        stats_y += 25
        cv2.putText(frame, "REKOMENDASI", (panel_x + 20, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.COLOR_ACCENT, 1)
        stats_y += 5
        
        recommendations = self.get_recommendations(detector)
        for rec in recommendations:
            stats_y += 22
            if stats_y > h - 50:
                break
            cv2.putText(frame, rec, (panel_x + 25, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.33, Config.COLOR_WHITE, 1)
        
        # ─── Footer ───
        cv2.putText(frame, "Tekan 'Q' keluar | 'S' screenshot | 'M' mesh",
                   (panel_x + 15, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, Config.COLOR_DARK_GRAY, 1)
    
    def get_recommendations(self, detector):
        """Get skincare recommendations based on analysis."""
        recs = []
        severity = detector.smoothed_severity
        
        if severity == "Bersih":
            recs.append("* Kulit Anda sehat! Pertahankan")
            recs.append("  rutinitas perawatan Anda.")
            recs.append("* Gunakan sunscreen SPF 30+")
            recs.append("  setiap hari.")
        elif severity == "Ringan":
            recs.append("* Cuci muka 2x sehari dengan")
            recs.append("  pembersih lembut.")
            recs.append("* Gunakan toner salicylic acid")
            recs.append("  0.5-2% untuk jerawat ringan.")
            recs.append("* Hindari menyentuh wajah.")
        elif severity == "Sedang":
            recs.append("* Gunakan benzoyl peroxide 2.5%")
            recs.append("  pada area berjerawat.")
            recs.append("* Pertimbangkan niacinamide 5%")
            recs.append("  untuk meredakan kemerahan.")
            recs.append("* Hindari makanan berminyak.")
            recs.append("* Ganti sarung bantal 2x/minggu.")
        elif severity in ("Parah", "Sangat Parah"):
            recs.append("* KONSULTASI DERMATOLOG segera.")
            recs.append("* Jangan memencet jerawat!")
            recs.append("* Gunakan pembersih pH rendah.")
            recs.append("* Hindari produk berminyak.")
            recs.append("* Pertimbangkan retinoid topikal")
            recs.append("  (dengan resep dokter).")
        
        # Zone-specific advice
        forehead_avg = 0
        chin_avg = 0
        if 'forehead' in detector.zone_history and len(detector.zone_history['forehead']) > 0:
            forehead_avg = np.mean(detector.zone_history['forehead'])
        if 'chin' in detector.zone_history and len(detector.zone_history['chin']) > 0:
            chin_avg = np.mean(detector.zone_history['chin'])
        
        if forehead_avg > 3:
            recs.append("")
            recs.append("! Dahi: Mungkin terkait stres")
            recs.append("  atau minyak rambut.")
        if chin_avg > 3:
            recs.append("")
            recs.append("! Dagu: Mungkin terkait hormonal.")
            recs.append("  Cek pola makan & tidur.")
        
        return recs
    
    def draw_top_bar(self, frame):
        """Draw minimalistic top status bar."""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), Config.COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, "ACNE SKIN SCANNER", (15, 33),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, Config.COLOR_ACCENT, 2)
        
        pulse_r = 5 + int(2 * math.sin(self.animation_phase * 0.15))
        cv2.circle(frame, (260, 28), pulse_r, Config.COLOR_GREEN, -1)
        cv2.putText(frame, "LIVE", (272, 33),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, Config.COLOR_GREEN, 1)
        
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (w - 370, 33),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.COLOR_GRAY, 1)
        
        cv2.line(frame, (0, 50), (w, 50), Config.COLOR_ACCENT, 1)
    
    def draw_no_face(self, frame):
        """Draw message when no face is detected."""
        h, w = frame.shape[:2]
        
        msg = "Wajah tidak terdeteksi"
        sub_msg = "Posisikan wajah Anda di depan kamera"
        
        text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        sub_size = cv2.getTextSize(sub_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        cx = (w - 340) // 2
        cy = h // 2
        
        box_w = max(text_size[0], sub_size[0]) + 60
        box_h = 80
        self.draw_rounded_rect(frame,
                              (cx - box_w // 2, cy - box_h // 2),
                              (cx + box_w // 2, cy + box_h // 2),
                              Config.COLOR_PANEL, radius=12, alpha=0.85)
        
        alpha_val = 0.4 + 0.3 * math.sin(self.animation_phase * 0.1)
        self.draw_rounded_rect(frame,
                              (cx - box_w // 2, cy - box_h // 2),
                              (cx + box_w // 2, cy + box_h // 2),
                              Config.COLOR_ACCENT, radius=12, thickness=2, alpha=alpha_val)
        
        cv2.putText(frame, msg,
                   (cx - text_size[0] // 2, cy - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.COLOR_ACCENT, 2)
        cv2.putText(frame, sub_msg,
                   (cx - sub_size[0] // 2, cy + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_GRAY, 1)
    
    def draw_face_frame(self, frame, face_bbox):
        """Draw a sleek targeting frame around the detected face."""
        if face_bbox is None:
            return
        
        x, y, w, h = face_bbox
        pad = 20
        x1, y1 = x - pad, y - pad
        x2, y2 = x + w + pad, y + h + pad
        
        corner_len = 25
        color = Config.COLOR_ACCENT2
        thickness = 2
        
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)
        
        line_color = tuple(max(0, int(c * 0.3)) for c in color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), line_color, 1)
    
    def update(self):
        """Update animation state."""
        self.animation_phase += 1


# ─────────────────────────────────────────────────────────
# SCREENSHOT
# ─────────────────────────────────────────────────────────

def save_screenshot(frame):
    """Save current frame as screenshot with timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"skin_scan_{timestamp}.png"
    cv2.imwrite(filename, frame)
    print(f"[+] Screenshot disimpan: {filename}")
    return filename


# ─────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ADVANCED SKIN ACNE SCANNER v2.0")
    print("  Analisis kulit real-time dengan deteksi jerawat")
    print("=" * 60)
    print()
    
    # Download model
    download_model()
    
    print("[*] Menginisialisasi kamera...")
    
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("[!] ERROR: Kamera tidak dapat dibuka!")
        print("    Pastikan kamera terhubung dan tidak digunakan aplikasi lain.")
        return
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[+] Kamera aktif: {actual_w}x{actual_h}")
    
    # Initialize MediaPipe Face Landmarker (Tasks API)
    print("[*] Menginisialisasi Face Landmarker...")
    base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    face_options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=Config.FACE_CONFIDENCE,
        min_face_presence_confidence=Config.FACE_CONFIDENCE,
        min_tracking_confidence=Config.FACE_TRACKING,
        running_mode=vision.RunningMode.IMAGE,
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
    
    # Initialize detector and UI
    acne_detector = AcneDetector()
    ui = UIRenderer()
    
    print("[+] Siap! Tekan 'Q' untuk keluar, 'S' untuk screenshot, 'M' toggle mesh.")
    print()
    
    fps_deque = deque(maxlen=30)
    prev_time = time.time()
    show_mesh = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Gagal membaca frame dari kamera.")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        ih, iw = frame.shape[:2]
        
        # FPS
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 0.001)
        prev_time = curr_time
        fps_deque.append(fps)
        avg_fps = np.mean(fps_deque)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = face_landmarker.detect(mp_image)
        
        face_bbox = None
        
        if result.face_landmarks and len(result.face_landmarks) > 0:
            landmarks = result.face_landmarks[0]  # list of NormalizedLandmark
            
            # Calculate face bounding box from landmarks
            x_coords = [int(lm.x * iw) for lm in landmarks]
            y_coords = [int(lm.y * ih) for lm in landmarks]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            face_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            # Draw face mesh if enabled
            if show_mesh:
                ui.draw_face_mesh_overlay(frame, landmarks, ih, iw)
            
            # Run acne detection
            spots, zones = acne_detector.analyze_frame(frame, landmarks, face_bbox)
            
            # Draw visualizations
            ui.draw_face_frame(frame, face_bbox)
            ui.draw_scan_line(frame, face_bbox)
            ui.draw_acne_markers(frame, spots)
        else:
            ui.draw_no_face(frame)
        
        # Draw UI elements
        ui.draw_top_bar(frame)
        ui.draw_info_panel(frame, acne_detector)
        
        # FPS display
        cv2.putText(frame, f"FPS: {int(avg_fps)}", (iw - 460, 33),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.COLOR_GRAY, 1)
        
        mesh_status = "ON" if show_mesh else "OFF"
        mesh_color = Config.COLOR_GREEN if show_mesh else Config.COLOR_DARK_GRAY
        cv2.putText(frame, f"Mesh: {mesh_status} [M]", (iw - 560, 33),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, mesh_color, 1)
        
        ui.update()
        
        # Display
        cv2.imshow("Acne Skin Scanner v2.0", frame)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            save_screenshot(frame)
        elif key == ord('m') or key == ord('M'):
            show_mesh = not show_mesh
            print(f"[*] Face Mesh: {'ON' if show_mesh else 'OFF'}")
    
    # Cleanup
    print("\n[*] Menutup aplikasi...")
    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    print("[+] Selesai.")


if __name__ == "__main__":
    main()