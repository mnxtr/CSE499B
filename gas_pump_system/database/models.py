from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Enum, Boolean, Text
)
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime
import enum


class Base(DeclarativeBase):
    pass


class FuelType(str, enum.Enum):
    REGULAR = "Regular"
    MIDGRADE = "Midgrade"
    PREMIUM = "Premium"
    DIESEL = "Diesel"


class PumpStatus(str, enum.Enum):
    IDLE = "idle"
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class PaymentStatus(str, enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class Pump(Base):
    __tablename__ = "pumps"

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False, unique=True)
    location = Column(String(100), nullable=True)
    status = Column(String(20), default=PumpStatus.IDLE)
    fuel_type = Column(String(20), default=FuelType.REGULAR)
    price_per_gallon = Column(Float, nullable=False, default=3.50)
    camera_index = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    last_calibrated = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    transactions = relationship("Transaction", back_populates="pump", lazy="dynamic")
    fuel_readings = relationship("FuelReading", back_populates="pump", lazy="dynamic")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "location": self.location,
            "status": self.status,
            "fuel_type": self.fuel_type,
            "price_per_gallon": self.price_per_gallon,
            "camera_index": self.camera_index,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)
    pump_id = Column(Integer, ForeignKey("pumps.id"), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    gallons_dispensed = Column(Float, default=0.0)
    price_per_gallon = Column(Float, nullable=False)
    total_cost = Column(Float, default=0.0)
    fuel_type = Column(String(20), nullable=False)
    payment_method = Column(String(30), default="card")
    payment_status = Column(String(20), default=PaymentStatus.PENDING)
    customer_id = Column(String(50), nullable=True)
    notes = Column(Text, nullable=True)

    pump = relationship("Pump", back_populates="transactions")
    readings = relationship("FuelReading", back_populates="transaction", lazy="dynamic")

    @property
    def duration_seconds(self):
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self):
        return {
            "id": self.id,
            "pump_id": self.pump_id,
            "pump_name": self.pump.name if self.pump else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "gallons_dispensed": round(self.gallons_dispensed, 3),
            "price_per_gallon": self.price_per_gallon,
            "total_cost": round(self.total_cost, 2),
            "fuel_type": self.fuel_type,
            "payment_method": self.payment_method,
            "payment_status": self.payment_status,
            "duration_seconds": self.duration_seconds,
        }


class FuelReading(Base):
    """Raw vision readings captured from pump display at intervals."""
    __tablename__ = "fuel_readings"

    id = Column(Integer, primary_key=True)
    pump_id = Column(Integer, ForeignKey("pumps.id"), nullable=False)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    gallons_reading = Column(Float, nullable=True)
    price_reading = Column(Float, nullable=True)
    total_reading = Column(Float, nullable=True)
    ocr_confidence = Column(Float, nullable=True)
    raw_ocr_text = Column(Text, nullable=True)
    frame_path = Column(String(255), nullable=True)

    pump = relationship("Pump", back_populates="fuel_readings")
    transaction = relationship("Transaction", back_populates="readings")

    def to_dict(self):
        return {
            "id": self.id,
            "pump_id": self.pump_id,
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "gallons_reading": self.gallons_reading,
            "price_reading": self.price_reading,
            "total_reading": self.total_reading,
            "ocr_confidence": self.ocr_confidence,
        }


class FuelPrice(Base):
    """Historical fuel price records."""
    __tablename__ = "fuel_prices"

    id = Column(Integer, primary_key=True)
    fuel_type = Column(String(20), nullable=False)
    price_per_gallon = Column(Float, nullable=False)
    effective_from = Column(DateTime, default=datetime.utcnow)
    effective_to = Column(DateTime, nullable=True)
    set_by = Column(String(100), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "fuel_type": self.fuel_type,
            "price_per_gallon": self.price_per_gallon,
            "effective_from": self.effective_from.isoformat() if self.effective_from else None,
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
        }


class Alert(Base):
    """System alerts and events."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    pump_id = Column(Integer, ForeignKey("pumps.id"), nullable=True)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), default="info")
    message = Column(Text, nullable=False)
    is_resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "pump_id": self.pump_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "is_resolved": self.is_resolved,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
