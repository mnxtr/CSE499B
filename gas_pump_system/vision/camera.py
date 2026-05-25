"""Camera abstraction: real webcam or simulated pump display frame."""
import cv2
import numpy as np
import logging
import random
import math
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class SimulatedPumpDisplay:
    """
    Renders a realistic seven-segment-style LCD pump display onto an OpenCV frame.
    Values animate along a sigmoidal fill curve so OCR testing is realistic.
    """

    BG_COLOR  = (15, 30, 15)
    LCD_COLOR = (0, 200, 80)
    DIM_COLOR = (0, 50, 20)
    TEXT_COLOR = (0, 255, 120)
    LABEL_COLOR = (150, 180, 150)

    def __init__(self, pump_id: int, price_per_gallon: float = 3.49, fuel_type: str = "Regular"):
        self.pump_id = pump_id
        self.price_per_gallon = price_per_gallon
        self.fuel_type = fuel_type
        self._start_time: Optional[float] = None
        self._max_gallons = random.uniform(8, 18)
        self._is_active = False

    def start_fill(self):
        self._start_time = time.time()
        self._max_gallons = random.uniform(8, 18)
        self._is_active = True

    def stop_fill(self):
        self._is_active = False

    def get_current_values(self) -> Tuple[float, float, float]:
        if not self._is_active or self._start_time is None:
            return 0.0, self.price_per_gallon, 0.0
        elapsed = time.time() - self._start_time
        # Sigmoidal: fills to ~95% over ~60 seconds
        fraction = min(0.99, 1 / (1 + math.exp(-0.12 * (elapsed - 25))))
        gallons = round(fraction * self._max_gallons, 3)
        total = round(gallons * self.price_per_gallon, 2)
        return gallons, self.price_per_gallon, total

    def render(self, width: int = 640, height: int = 480) -> np.ndarray:
        frame = np.full((height, width, 3), self.BG_COLOR, dtype=np.uint8)
        gallons, price, total = self.get_current_values()

        # Pump body
        cv2.rectangle(frame, (50, 30), (width - 50, height - 30), (40, 60, 40), -1)
        cv2.rectangle(frame, (50, 30), (width - 50, height - 30), (0, 120, 50), 3)

        # LCD panel
        lcd_x1, lcd_y1, lcd_x2, lcd_y2 = 80, 60, width - 80, height - 80
        cv2.rectangle(frame, (lcd_x1, lcd_y1), (lcd_x2, lcd_y2), (5, 15, 5), -1)
        cv2.rectangle(frame, (lcd_x1, lcd_y1), (lcd_x2, lcd_y2), (0, 160, 60), 2)

        # Fuel type badge
        cv2.putText(frame, f"  {self.fuel_type}  ",
                    (lcd_x1 + 10, lcd_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, self.LCD_COLOR, 1)

        # Divider lines
        row_h = (lcd_y2 - lcd_y1 - 40) // 3
        for i in range(1, 3):
            y = lcd_y1 + 40 + i * row_h
            cv2.line(frame, (lcd_x1 + 10, y), (lcd_x2 - 10, y), (0, 80, 30), 1)

        base_y = lcd_y1 + 40

        def draw_row(label: str, value: str, row_idx: int):
            ry = base_y + row_idx * row_h
            cv2.putText(frame, label, (lcd_x1 + 15, ry + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.LABEL_COLOR, 1)
            cv2.putText(frame, value, (lcd_x1 + 15, ry + row_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, self.TEXT_COLOR, 2)

        draw_row("GALLONS", f"{gallons:8.3f}", 0)
        draw_row("PRICE/GAL", f"${price:7.3f}", 1)
        draw_row("SALE TOTAL", f"${total:7.2f}", 2)

        # Status indicator
        status = "DISPENSING" if self._is_active else "READY"
        status_col = (0, 220, 80) if self._is_active else (120, 180, 120)
        cv2.putText(frame, f"PUMP {self.pump_id}  [{status}]",
                    (lcd_x1 + 10, height - 45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, status_col, 1)

        # Blinking dot when active
        if self._is_active and int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (lcd_x2 - 20, height - 50), 6, (0, 255, 60), -1)

        return frame


class Camera:
    """
    Wraps either a real OpenCV VideoCapture or a SimulatedPumpDisplay.
    Falls back to simulation automatically if the camera can't be opened.
    """

    def __init__(self, camera_index: int = 0, pump_id: int = 1,
                 price_per_gallon: float = 3.49, fuel_type: str = "Regular",
                 force_simulate: bool = False):
        self.camera_index = camera_index
        self.pump_id = pump_id
        self.simulated = SimulatedPumpDisplay(pump_id, price_per_gallon, fuel_type)
        self._cap: Optional[cv2.VideoCapture] = None
        self._use_simulation = force_simulate

        if not force_simulate:
            self._try_open_camera()

    def _try_open_camera(self):
        try:
            cap = cv2.VideoCapture(self.camera_index)
            if cap.isOpened():
                self._cap = cap
                logger.info(f"Camera {self.camera_index} opened for pump {self.pump_id}")
            else:
                logger.info(f"Camera {self.camera_index} unavailable — using simulation")
                self._use_simulation = True
        except Exception as e:
            logger.warning(f"Camera error: {e} — using simulation")
            self._use_simulation = True

    def read_frame(self) -> Optional[np.ndarray]:
        if self._use_simulation:
            return self.simulated.render()
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            return frame if ret else None
        return None

    def start_simulation(self):
        self.simulated.start_fill()

    def stop_simulation(self):
        self.simulated.stop_fill()

    def get_simulated_values(self) -> Tuple[float, float, float]:
        return self.simulated.get_current_values()

    def is_simulated(self) -> bool:
        return self._use_simulation

    def release(self):
        if self._cap:
            self._cap.release()

    def encode_frame_jpeg(self, frame: np.ndarray) -> bytes:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()
