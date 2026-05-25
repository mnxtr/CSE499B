"""Preprocessing pipeline to make pump display digits readable by OCR."""
import cv2
import numpy as np
from typing import Optional, Tuple


class ImageProcessor:
    def preprocess_for_ocr(self, frame: np.ndarray) -> np.ndarray:
        """Full pipeline: denoise → grayscale → threshold → morphology."""
        if frame is None:
            return np.zeros((100, 300), dtype=np.uint8)

        # Upscale small frames so OCR has enough resolution
        h, w = frame.shape[:2]
        if w < 400:
            scale = 400 / w
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        sharpened = self._sharpen(denoised)
        binary = self._adaptive_threshold(sharpened)
        return binary

    def extract_display_regions(self, frame: np.ndarray) -> dict:
        """
        Divide frame into labeled ROIs for each pump field.
        Assumes pump display layout (top to bottom):
          - gallons row  (top 33%)
          - price/gal row (middle 33%)
          - total row   (bottom 33%)
        """
        h, w = frame.shape[:2]
        third = h // 3
        return {
            "gallons": frame[0:third, :],
            "price":   frame[third:2*third, :],
            "total":   frame[2*third:h, :],
        }

    def detect_display_area(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Try to locate the LCD display panel automatically using contour detection.
        Falls back to using the full frame if detection fails.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0
        fh, fw = frame.shape[:2]
        min_area = (fw * fh) * 0.05

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            if 1.0 < aspect < 5.0 and area > best_area:
                best_area = area
                best = frame[y:y+h, x:x+w]

        return best if best is not None else frame

    def draw_overlay(self, frame: np.ndarray, pump_data: dict,
                     pump_name: str, status: str) -> np.ndarray:
        """Render pump readings as a semi-transparent overlay on the frame."""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Dark background strip at top
        cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        status_color = {
            "active": (0, 220, 0),
            "idle": (180, 180, 180),
            "maintenance": (0, 165, 255),
            "offline": (0, 0, 220),
        }.get(status, (180, 180, 180))

        cv2.putText(frame, pump_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status.upper()}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)

        gallons = pump_data.get("gallons", 0.0) or 0.0
        price   = pump_data.get("price",   0.0) or 0.0
        total   = pump_data.get("total",   0.0) or 0.0

        cv2.putText(frame, f"Gal: {gallons:.3f}  $/Gal: {price:.3f}  Total: ${total:.2f}",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)
        return frame

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _sharpen(img: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def _adaptive_threshold(img: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
