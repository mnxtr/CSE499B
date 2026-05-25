"""OCR-based pump display reader with confidence scoring and fallback simulation."""
import re
import logging
import random
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)


@dataclass
class PumpReading:
    gallons: Optional[float] = None
    price_per_gallon: Optional[float] = None
    total_cost: Optional[float] = None
    confidence: float = 0.0
    raw_text: str = ""
    source: str = "ocr"  # "ocr" | "simulated"

    def is_valid(self) -> bool:
        return (
            self.gallons is not None and self.gallons >= 0 and
            self.price_per_gallon is not None and self.price_per_gallon > 0 and
            self.total_cost is not None and self.total_cost >= 0
        )

    def to_dict(self) -> dict:
        return {
            "gallons": self.gallons,
            "price": self.price_per_gallon,
            "total": self.total_cost,
            "confidence": self.confidence,
            "source": self.source,
        }


class PumpReader:
    def __init__(self):
        self.processor = ImageProcessor()
        self._tesseract_config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789."

    def read_frame(self, frame: np.ndarray, pump_price: float = 3.49) -> PumpReading:
        """Main entry: try OCR; fall back to simulation if OCR fails/unavailable."""
        if frame is None:
            return self._make_simulated_reading(pump_price)

        try:
            reading = self._ocr_read(frame, pump_price)
            if reading.is_valid() and reading.confidence > 0.4:
                return reading
        except Exception as e:
            logger.debug(f"OCR error: {e}")

        return self._make_simulated_reading(pump_price)

    # ── OCR path ──────────────────────────────────────────────────────────────

    def _ocr_read(self, frame: np.ndarray, pump_price: float) -> PumpReading:
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("pytesseract not available")

        preprocessed = self.processor.preprocess_for_ocr(frame)
        regions = self.processor.extract_display_regions(preprocessed)

        raw_parts = []
        values = {}
        confidences = []

        field_map = {
            "gallons": ("gallons", 0, 999),
            "price":   ("price_per_gallon", 0.5, 20),
            "total":   ("total_cost", 0, 9999),
        }

        for region_name, (field_key, lo, hi) in field_map.items():
            region = regions.get(region_name)
            if region is None:
                continue
            text, conf = self._run_tesseract(region)
            raw_parts.append(f"{region_name}:{text}")
            confidences.append(conf)
            val = self._parse_number(text)
            if val is not None and lo <= val <= hi:
                values[field_key] = val

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # Cross-validate: total ≈ gallons × price
        if all(k in values for k in ("gallons", "price_per_gallon", "total_cost")):
            expected = values["gallons"] * values["price_per_gallon"]
            if abs(expected - values["total_cost"]) / max(expected, 0.01) > 0.05:
                avg_conf *= 0.5  # penalise inconsistent reading

        return PumpReading(
            gallons=values.get("gallons"),
            price_per_gallon=values.get("price_per_gallon", pump_price),
            total_cost=values.get("total_cost"),
            confidence=avg_conf,
            raw_text=" | ".join(raw_parts),
            source="ocr",
        )

    def _run_tesseract(self, img: np.ndarray) -> Tuple[str, float]:
        data = pytesseract.image_to_data(
            img, config=self._tesseract_config, output_type=pytesseract.Output.DICT
        )
        texts, confs = [], []
        for txt, conf in zip(data["text"], data["conf"]):
            txt = txt.strip()
            if txt and conf > 0:
                texts.append(txt)
                confs.append(conf / 100.0)
        text = " ".join(texts)
        confidence = sum(confs) / len(confs) if confs else 0.0
        return text, confidence

    @staticmethod
    def _parse_number(text: str) -> Optional[float]:
        text = text.strip().replace(",", ".")
        match = re.search(r"\d+\.?\d*", text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return None

    # ── Simulation path ───────────────────────────────────────────────────────

    def _make_simulated_reading(self, pump_price: float) -> PumpReading:
        """Generate a realistic in-progress fueling reading for demo/testing."""
        t = SimulationState.get_elapsed()
        # Sigmoid-like fill curve: fast in middle, slow at start/end
        fill_fraction = 1 / (1 + math.exp(-0.15 * (t - 30)))
        max_gallons = random.uniform(8, 15)
        gallons = round(fill_fraction * max_gallons, 3)
        total = round(gallons * pump_price, 2)
        return PumpReading(
            gallons=gallons,
            price_per_gallon=pump_price,
            total_cost=total,
            confidence=0.99,
            raw_text=f"sim gal={gallons} price={pump_price} total={total}",
            source="simulated",
        )


class SimulationState:
    """Global monotonic timer for simulated fills (resets each transaction)."""
    _start: float = 0.0

    @classmethod
    def reset(cls):
        import time
        cls._start = time.time()

    @classmethod
    def get_elapsed(cls) -> float:
        import time
        if cls._start == 0:
            cls.reset()
        return time.time() - cls._start
