from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

from .models import Base, Pump, Transaction, FuelReading, FuelPrice, Alert, PumpStatus, FuelType

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db_url: str = "sqlite:///gas_pump_station.db"):
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        logger.info(f"Database initialized: {db_url}")

    @contextmanager
    def get_session(self) -> Session:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Pump CRUD ────────────────────────────────────────────────────────────

    def seed_pumps(self):
        """Insert default pumps if the table is empty."""
        with self.get_session() as session:
            if session.query(Pump).count() > 0:
                return
            defaults = [
                Pump(name="Pump 1", location="Lane A", fuel_type=FuelType.REGULAR,  price_per_gallon=3.49, camera_index=0),
                Pump(name="Pump 2", location="Lane A", fuel_type=FuelType.MIDGRADE, price_per_gallon=3.69, camera_index=1),
                Pump(name="Pump 3", location="Lane B", fuel_type=FuelType.PREMIUM,  price_per_gallon=3.99, camera_index=2),
                Pump(name="Pump 4", location="Lane B", fuel_type=FuelType.DIESEL,   price_per_gallon=3.79, camera_index=3),
            ]
            session.add_all(defaults)
            # Seed current fuel prices
            for p in defaults:
                session.add(FuelPrice(fuel_type=p.fuel_type, price_per_gallon=p.price_per_gallon))
            logger.info("Seeded 4 default pumps")

    def get_all_pumps(self) -> List[Dict]:
        with self.get_session() as session:
            pumps = session.query(Pump).filter_by(is_active=True).all()
            return [p.to_dict() for p in pumps]

    def get_pump(self, pump_id: int) -> Optional[Dict]:
        with self.get_session() as session:
            p = session.get(Pump, pump_id)
            return p.to_dict() if p else None

    def update_pump_status(self, pump_id: int, status: str) -> bool:
        with self.get_session() as session:
            p = session.get(Pump, pump_id)
            if not p:
                return False
            p.status = status
            p.updated_at = datetime.utcnow()
            return True

    def update_pump_price(self, pump_id: int, price: float, set_by: str = "admin") -> bool:
        with self.get_session() as session:
            p = session.get(Pump, pump_id)
            if not p:
                return False
            old_price = p.price_per_gallon
            p.price_per_gallon = price
            p.updated_at = datetime.utcnow()
            # Record price history
            session.query(FuelPrice).filter(
                FuelPrice.fuel_type == p.fuel_type, FuelPrice.effective_to.is_(None)
            ).update({"effective_to": datetime.utcnow()})
            session.add(FuelPrice(fuel_type=p.fuel_type, price_per_gallon=price, set_by=set_by))
            logger.info(f"Pump {pump_id} price updated {old_price} -> {price}")
            return True

    # ── Transaction CRUD ─────────────────────────────────────────────────────

    def start_transaction(self, pump_id: int, payment_method: str = "card") -> Optional[int]:
        with self.get_session() as session:
            p = session.get(Pump, pump_id)
            if not p:
                return None
            txn = Transaction(
                pump_id=pump_id,
                price_per_gallon=p.price_per_gallon,
                fuel_type=p.fuel_type,
                payment_method=payment_method,
                payment_status="pending",
            )
            session.add(txn)
            p.status = PumpStatus.ACTIVE
            session.flush()
            logger.info(f"Transaction {txn.id} started on pump {pump_id}")
            return txn.id

    def update_transaction(self, txn_id: int, gallons: float, total: float) -> bool:
        with self.get_session() as session:
            txn = session.get(Transaction, txn_id)
            if not txn:
                return False
            txn.gallons_dispensed = gallons
            txn.total_cost = total
            return True

    def end_transaction(self, txn_id: int, gallons: float, total: float,
                        payment_status: str = "completed") -> Optional[Dict]:
        with self.get_session() as session:
            txn = session.get(Transaction, txn_id)
            if not txn:
                return None
            txn.end_time = datetime.utcnow()
            txn.gallons_dispensed = gallons
            txn.total_cost = total
            txn.payment_status = payment_status
            pump = session.get(Pump, txn.pump_id)
            if pump:
                pump.status = PumpStatus.IDLE
            logger.info(f"Transaction {txn_id} ended: {gallons:.3f} gal, ${total:.2f}")
            return txn.to_dict()

    def get_transactions(self, pump_id: Optional[int] = None, limit: int = 100,
                         hours: Optional[int] = None) -> List[Dict]:
        with self.get_session() as session:
            q = session.query(Transaction)
            if pump_id:
                q = q.filter_by(pump_id=pump_id)
            if hours:
                cutoff = datetime.utcnow() - timedelta(hours=hours)
                q = q.filter(Transaction.start_time >= cutoff)
            txns = q.order_by(desc(Transaction.start_time)).limit(limit).all()
            return [t.to_dict() for t in txns]

    def get_active_transaction(self, pump_id: int) -> Optional[Dict]:
        with self.get_session() as session:
            txn = (
                session.query(Transaction)
                .filter_by(pump_id=pump_id, payment_status="pending")
                .filter(Transaction.end_time.is_(None))
                .order_by(desc(Transaction.start_time))
                .first()
            )
            return txn.to_dict() if txn else None

    # ── FuelReading CRUD ─────────────────────────────────────────────────────

    def record_reading(self, pump_id: int, gallons: Optional[float],
                       price: Optional[float], total: Optional[float],
                       confidence: float = 0.0, raw_text: str = "",
                       txn_id: Optional[int] = None, frame_path: str = "") -> int:
        with self.get_session() as session:
            reading = FuelReading(
                pump_id=pump_id,
                transaction_id=txn_id,
                gallons_reading=gallons,
                price_reading=price,
                total_reading=total,
                ocr_confidence=confidence,
                raw_ocr_text=raw_text,
                frame_path=frame_path,
            )
            session.add(reading)
            session.flush()
            return reading.id

    # ── Analytics ────────────────────────────────────────────────────────────

    def get_revenue_summary(self, hours: int = 24) -> Dict:
        with self.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            result = session.query(
                func.count(Transaction.id).label("total_transactions"),
                func.sum(Transaction.total_cost).label("total_revenue"),
                func.sum(Transaction.gallons_dispensed).label("total_gallons"),
                func.avg(Transaction.total_cost).label("avg_transaction"),
            ).filter(
                Transaction.start_time >= cutoff,
                Transaction.payment_status == "completed",
            ).first()
            return {
                "period_hours": hours,
                "total_transactions": result.total_transactions or 0,
                "total_revenue": round(result.total_revenue or 0, 2),
                "total_gallons": round(result.total_gallons or 0, 3),
                "avg_transaction": round(result.avg_transaction or 0, 2),
            }

    def get_revenue_by_pump(self, hours: int = 24) -> List[Dict]:
        with self.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            rows = (
                session.query(
                    Pump.name,
                    Pump.fuel_type,
                    func.count(Transaction.id).label("count"),
                    func.sum(Transaction.total_cost).label("revenue"),
                    func.sum(Transaction.gallons_dispensed).label("gallons"),
                )
                .join(Transaction, Transaction.pump_id == Pump.id)
                .filter(
                    Transaction.start_time >= cutoff,
                    Transaction.payment_status == "completed",
                )
                .group_by(Pump.id)
                .all()
            )
            return [
                {
                    "pump_name": r.name,
                    "fuel_type": r.fuel_type,
                    "transactions": r.count,
                    "revenue": round(r.revenue or 0, 2),
                    "gallons": round(r.gallons or 0, 3),
                }
                for r in rows
            ]

    def get_hourly_revenue(self, hours: int = 24) -> List[Dict]:
        """Return per-hour revenue for charting."""
        with self.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            txns = (
                session.query(Transaction)
                .filter(Transaction.start_time >= cutoff, Transaction.payment_status == "completed")
                .all()
            )
            buckets: Dict[str, float] = {}
            for txn in txns:
                hour_key = txn.start_time.strftime("%Y-%m-%d %H:00")
                buckets[hour_key] = buckets.get(hour_key, 0) + (txn.total_cost or 0)
            return [{"hour": k, "revenue": round(v, 2)} for k, v in sorted(buckets.items())]

    # ── Alerts ───────────────────────────────────────────────────────────────

    def create_alert(self, alert_type: str, message: str, severity: str = "info",
                     pump_id: Optional[int] = None) -> int:
        with self.get_session() as session:
            alert = Alert(pump_id=pump_id, alert_type=alert_type,
                          severity=severity, message=message)
            session.add(alert)
            session.flush()
            return alert.id

    def get_alerts(self, unresolved_only: bool = True, limit: int = 50) -> List[Dict]:
        with self.get_session() as session:
            q = session.query(Alert)
            if unresolved_only:
                q = q.filter_by(is_resolved=False)
            alerts = q.order_by(desc(Alert.created_at)).limit(limit).all()
            return [a.to_dict() for a in alerts]

    def resolve_alert(self, alert_id: int) -> bool:
        with self.get_session() as session:
            a = session.get(Alert, alert_id)
            if not a:
                return False
            a.is_resolved = True
            a.resolved_at = datetime.utcnow()
            return True
