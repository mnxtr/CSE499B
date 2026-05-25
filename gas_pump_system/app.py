"""Flask + SocketIO application for the gas pump management dashboard."""
import sys
import os
import logging
from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO

# Allow imports from this package root
sys.path.insert(0, os.path.dirname(__file__))

from database.db_manager import DatabaseManager
from pump_manager import PumpManager

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"] = "gaspump-dev-secret"

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

DB_PATH = os.path.join(os.path.dirname(__file__), "gas_pump_station.db")
db = DatabaseManager(f"sqlite:///{DB_PATH}")
manager = PumpManager(db, socketio)


# ── Page routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/transactions")
def transactions_page():
    return render_template("transactions.html")

@app.route("/reports")
def reports_page():
    return render_template("reports.html")

@app.route("/settings")
def settings_page():
    return render_template("settings.html")


# ── REST API ──────────────────────────────────────────────────────────────────

@app.route("/api/pumps")
def api_pumps():
    return jsonify(manager.get_all_states())

@app.route("/api/pumps/<int:pump_id>")
def api_pump(pump_id):
    ctrl = manager.get_controller(pump_id)
    if not ctrl:
        return jsonify({"error": "Pump not found"}), 404
    return jsonify(ctrl.get_state())

@app.route("/api/pumps/<int:pump_id>/start", methods=["POST"])
def api_start_transaction(pump_id):
    data = request.get_json(silent=True) or {}
    payment = data.get("payment_method", "card")
    txn_id = manager.start_transaction(pump_id, payment)
    if txn_id is None:
        return jsonify({"error": "Cannot start transaction (pump busy or not found)"}), 400
    return jsonify({"transaction_id": txn_id, "pump_id": pump_id})

@app.route("/api/pumps/<int:pump_id>/stop", methods=["POST"])
def api_stop_transaction(pump_id):
    result = manager.end_transaction(pump_id)
    if result is None:
        return jsonify({"error": "No active transaction on this pump"}), 400
    return jsonify(result)

@app.route("/api/pumps/<int:pump_id>/maintenance", methods=["POST"])
def api_maintenance(pump_id):
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled", True)
    manager.set_maintenance(pump_id, enabled)
    return jsonify({"pump_id": pump_id, "maintenance": enabled})

@app.route("/api/pumps/<int:pump_id>/price", methods=["POST"])
def api_update_price(pump_id):
    data = request.get_json(silent=True) or {}
    price = data.get("price")
    if not price or float(price) <= 0:
        return jsonify({"error": "Invalid price"}), 400
    manager.update_price(pump_id, float(price))
    return jsonify({"pump_id": pump_id, "new_price": float(price)})

@app.route("/api/transactions")
def api_transactions():
    pump_id = request.args.get("pump_id", type=int)
    limit   = request.args.get("limit", 50, type=int)
    hours   = request.args.get("hours", type=int)
    txns    = db.get_transactions(pump_id=pump_id, limit=limit, hours=hours)
    return jsonify(txns)

@app.route("/api/analytics/summary")
def api_summary():
    hours = request.args.get("hours", 24, type=int)
    return jsonify(db.get_revenue_summary(hours))

@app.route("/api/analytics/by-pump")
def api_by_pump():
    hours = request.args.get("hours", 24, type=int)
    return jsonify(db.get_revenue_by_pump(hours))

@app.route("/api/analytics/hourly")
def api_hourly():
    hours = request.args.get("hours", 24, type=int)
    return jsonify(db.get_hourly_revenue(hours))

@app.route("/api/alerts")
def api_alerts():
    unresolved = request.args.get("unresolved", "true").lower() == "true"
    return jsonify(db.get_alerts(unresolved_only=unresolved))

@app.route("/api/alerts/<int:alert_id>/resolve", methods=["POST"])
def api_resolve_alert(alert_id):
    db.resolve_alert(alert_id)
    return jsonify({"resolved": alert_id})

# ── Video stream ──────────────────────────────────────────────────────────────

def generate_stream(pump_id: int):
    ctrl = manager.get_controller(pump_id)
    if not ctrl:
        return
    while True:
        frame = ctrl.camera.read_frame()
        if frame is not None:
            from vision.image_processor import ImageProcessor
            proc = ImageProcessor()
            annotated = proc.draw_overlay(
                frame, ctrl.get_latest_reading(),
                ctrl.pump_name, ctrl._status
            )
            jpg = ctrl.camera.encode_frame_jpeg(annotated)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        import time; time.sleep(0.1)

@app.route("/video/<int:pump_id>")
def video_feed(pump_id):
    return Response(generate_stream(pump_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ── SocketIO events ───────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    logger.info("Client connected")
    socketio.emit("init", {"pumps": manager.get_all_states()})

@socketio.on("request_start")
def on_request_start(data):
    pump_id = data.get("pump_id")
    payment = data.get("payment_method", "card")
    txn_id  = manager.start_transaction(pump_id, payment)
    socketio.emit("transaction_started", {"pump_id": pump_id, "txn_id": txn_id})

@socketio.on("request_stop")
def on_request_stop(data):
    pump_id = data.get("pump_id")
    result  = manager.end_transaction(pump_id)
    socketio.emit("transaction_stopped", result or {"error": "no active txn"})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        manager.initialize()
        logger.info("Starting Gas Pump Management System on http://0.0.0.0:5000")
        socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
    finally:
        manager.shutdown()
