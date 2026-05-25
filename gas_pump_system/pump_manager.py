"""Central controller: manages pump lifecycle, vision loop, and DB writes."""
import threading
import logging
import time
from typing import Dict, Optional, List
from datetime import datetime

from database.db_manager import DatabaseManager
from vision.camera import Camera
from vision.pump_reader import PumpReader, SimulationState

logger = logging.getLogger(__name__)


class PumpController:
    """Manages one physical pump: camera, reader, and active transaction."""

    POLL_INTERVAL = 1.5  # seconds between vision captures

    def __init__(self, pump_info: dict, db: DatabaseManager, socketio=None):
        self.pump_id   = pump_info["id"]
        self.pump_name = pump_info["name"]
        self.fuel_type = pump_info["fuel_type"]
        self.price     = pump_info["price_per_gallon"]
        self.db        = db
        self.socketio  = socketio

        self.camera = Camera(
            camera_index=pump_info.get("camera_index", 0),
            pump_id=self.pump_id,
            price_per_gallon=self.price,
            fuel_type=self.fuel_type,
            force_simulate=True,  # Change to False to try real camera first
        )
        self.reader = PumpReader()

        self._active_txn_id: Optional[int] = None
        self._current_reading: dict = {}
        self._status: str = pump_info.get("status", "idle")
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def start_monitoring(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._vision_loop, daemon=True,
                                        name=f"pump-{self.pump_id}")
        self._thread.start()
        logger.info(f"Monitoring started for {self.pump_name}")

    def stop_monitoring(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def start_transaction(self, payment_method: str = "card") -> Optional[int]:
        with self._lock:
            if self._active_txn_id is not None:
                return None  # Already in progress
            txn_id = self.db.start_transaction(self.pump_id, payment_method)
            if txn_id:
                self._active_txn_id = txn_id
                self._status = "active"
                SimulationState.reset()
                self.camera.start_simulation()
                self._emit("pump_update", self._state_dict())
                logger.info(f"{self.pump_name}: transaction {txn_id} started")
            return txn_id

    def end_transaction(self) -> Optional[dict]:
        with self._lock:
            if self._active_txn_id is None:
                return None
            txn_id = self._active_txn_id
            reading = self._current_reading
            gallons = reading.get("gallons") or 0.0
            total   = reading.get("total")   or round(gallons * self.price, 2)
            result  = self.db.end_transaction(txn_id, gallons, total, "completed")
            self._active_txn_id = None
            self._status = "idle"
            self.camera.stop_simulation()
            self._current_reading = {}
            self._emit("pump_update", self._state_dict())
            self._emit("transaction_complete", result)
            logger.info(f"{self.pump_name}: transaction {txn_id} ended — ${total:.2f}")
            return result

    def set_maintenance(self, enabled: bool):
        self._status = "maintenance" if enabled else "idle"
        self.db.update_pump_status(self.pump_id, self._status)
        self._emit("pump_update", self._state_dict())

    def update_price(self, price: float):
        self.price = price
        self.camera.simulated.price_per_gallon = price
        self.db.update_pump_price(self.pump_id, price)

    def get_state(self) -> dict:
        return self._state_dict()

    def get_latest_reading(self) -> dict:
        return dict(self._current_reading)

    # ── Vision loop ────────────────────────────────────────────────────────

    def _vision_loop(self):
        while not self._stop_event.is_set():
            try:
                frame = self.camera.read_frame()
                if frame is not None:
                    reading = self.reader.read_frame(frame, self.price)
                    with self._lock:
                        self._current_reading = reading.to_dict()
                        if self._active_txn_id:
                            self.db.update_transaction(
                                self._active_txn_id,
                                reading.gallons or 0.0,
                                reading.total_cost or 0.0,
                            )
                            self.db.record_reading(
                                pump_id=self.pump_id,
                                txn_id=self._active_txn_id,
                                gallons=reading.gallons,
                                price=reading.price_per_gallon,
                                total=reading.total_cost,
                                confidence=reading.confidence,
                                raw_text=reading.raw_text,
                            )
                    self._emit("pump_reading", {
                        "pump_id": self.pump_id,
                        "pump_name": self.pump_name,
                        **self._current_reading,
                    })
            except Exception as e:
                logger.error(f"{self.pump_name} vision loop error: {e}")
            time.sleep(self.POLL_INTERVAL)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _state_dict(self) -> dict:
        r = self._current_reading
        return {
            "pump_id":   self.pump_id,
            "pump_name": self.pump_name,
            "fuel_type": self.fuel_type,
            "price":     self.price,
            "status":    self._status,
            "txn_id":    self._active_txn_id,
            "gallons":   r.get("gallons", 0.0),
            "total":     r.get("total", 0.0),
            "source":    r.get("source", "idle"),
        }

    def _emit(self, event: str, data: dict):
        if self.socketio:
            try:
                self.socketio.emit(event, data)
            except Exception:
                pass


class PumpManager:
    """Manages all pump controllers for the station."""

    def __init__(self, db: DatabaseManager, socketio=None):
        self.db = db
        self.socketio = socketio
        self._controllers: Dict[int, PumpController] = {}

    def initialize(self):
        self.db.seed_pumps()
        pumps = self.db.get_all_pumps()
        for p in pumps:
            ctrl = PumpController(p, self.db, self.socketio)
            ctrl.start_monitoring()
            self._controllers[p["id"]] = ctrl
        logger.info(f"PumpManager initialized with {len(self._controllers)} pumps")

    def shutdown(self):
        for ctrl in self._controllers.values():
            ctrl.stop_monitoring()
            ctrl.camera.release()

    def get_controller(self, pump_id: int) -> Optional[PumpController]:
        return self._controllers.get(pump_id)

    def get_all_states(self) -> List[dict]:
        return [c.get_state() for c in self._controllers.values()]

    def start_transaction(self, pump_id: int, payment_method: str = "card") -> Optional[int]:
        ctrl = self._controllers.get(pump_id)
        return ctrl.start_transaction(payment_method) if ctrl else None

    def end_transaction(self, pump_id: int) -> Optional[dict]:
        ctrl = self._controllers.get(pump_id)
        return ctrl.end_transaction() if ctrl else None

    def set_maintenance(self, pump_id: int, enabled: bool):
        ctrl = self._controllers.get(pump_id)
        if ctrl:
            ctrl.set_maintenance(enabled)

    def update_price(self, pump_id: int, price: float):
        ctrl = self._controllers.get(pump_id)
        if ctrl:
            ctrl.update_price(price)
