"""Railgate web UI.

Spins a ROS2 node (subscribes to mobile train GPS + IMU, publishes a
crossing-status string) alongside a Flask server that streams live data to a
browser dashboard via Server-Sent Events.

Run with:
    ros2 run railgate web_ui
or directly:
    python3 -m railgate.web_ui
"""

from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import String

from flask import Flask, Response, jsonify, render_template, request


# ---------- shared state ---------------------------------------------------

@dataclass
class GpsSample:
    lat: float
    lon: float
    alt: float
    stamp: float


@dataclass
class ImuSample:
    qx: float
    qy: float
    qz: float
    qw: float
    ang_vel: tuple  # (x, y, z)
    lin_acc: tuple  # (x, y, z)
    stamp: float


@dataclass
class Settings:
    crossing_lat: Optional[float] = None
    crossing_lon: Optional[float] = None
    incoming_threshold_m: float = 200.0   # train within this distance -> "incoming"
    passing_threshold_m: float = 30.0     # train within this distance -> "passing"


@dataclass
class SharedState:
    gps: Optional[GpsSample] = None
    imu: Optional[ImuSample] = None
    status: str = "unknown"                # safe / incoming / passing / outgoing / unknown
    distance_m: Optional[float] = None
    has_passed: bool = False               # latched once the train clears the crossing
    settings: Settings = field(default_factory=Settings)
    lock: threading.Lock = field(default_factory=threading.Lock)
    subscribers: list = field(default_factory=list)  # list[queue.Queue]

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "gps": asdict(self.gps) if self.gps else None,
                "imu": asdict(self.imu) if self.imu else None,
                "status": self.status,
                "distance_m": self.distance_m,
                "settings": asdict(self.settings),
                "armed": self.settings.crossing_lat is not None
                         and self.settings.crossing_lon is not None,
            }

    def broadcast(self) -> None:
        payload = self.snapshot()
        with self.lock:
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass


STATE = SharedState()


# ---------- geometry -------------------------------------------------------

_EARTH_R = 6_371_000.0  # meters

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_R * math.asin(math.sqrt(a))


def classify(distance_m: float, s: Settings, has_passed: bool = False) -> tuple[str, bool]:
    """Return (status, has_passed).

    `incoming` and `outgoing` share the same distance band (between the passing
    and incoming thresholds); they differ only by direction. `has_passed`
    latches True once the train is within the passing threshold, so the band
    reads `outgoing` while the train recedes. It clears once the train is
    `safe` (fully clear of the crossing).
    """
    if distance_m <= s.passing_threshold_m:
        return "passing", True
    if distance_m <= s.incoming_threshold_m:
        return ("outgoing" if has_passed else "incoming"), has_passed
    return "safe", False


# ---------- ROS2 node ------------------------------------------------------

class RailgateNode(Node):
    def __init__(self) -> None:
        super().__init__("railgate_web_ui")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(NavSatFix, "/mobile_sensor/gps", self._on_gps, sensor_qos)
        self.create_subscription(Imu, "/mobile_sensor/imu", self._on_imu, sensor_qos)
        self.status_pub = self.create_publisher(String, "/status", 10)

        # Re-publish status periodically so a late-joining actuator sees it.
        self.create_timer(0.5, self._tick)

        self.get_logger().info("railgate_web_ui node up; subscribing to /mobile_sensor/{gps,imu}")

    def _on_gps(self, msg: NavSatFix) -> None:
        sample = GpsSample(
            lat=float(msg.latitude),
            lon=float(msg.longitude),
            alt=float(msg.altitude),
            stamp=time.time(),
        )
        with STATE.lock:
            STATE.gps = sample
            s = STATE.settings
            armed = s.crossing_lat is not None and s.crossing_lon is not None
            if armed:
                d = haversine_m(sample.lat, sample.lon, s.crossing_lat, s.crossing_lon)
                STATE.distance_m = d
                STATE.status, STATE.has_passed = classify(d, s, STATE.has_passed)
            else:
                STATE.distance_m = None
                STATE.status = "unknown"
                STATE.has_passed = False
            status_to_pub = STATE.status if armed else None

        if status_to_pub is not None:
            self.status_pub.publish(String(data=status_to_pub))

        STATE.broadcast()

    def _on_imu(self, msg: Imu) -> None:
        sample = ImuSample(
            qx=float(msg.orientation.x),
            qy=float(msg.orientation.y),
            qz=float(msg.orientation.z),
            qw=float(msg.orientation.w),
            ang_vel=(
                float(msg.angular_velocity.x),
                float(msg.angular_velocity.y),
                float(msg.angular_velocity.z),
            ),
            lin_acc=(
                float(msg.linear_acceleration.x),
                float(msg.linear_acceleration.y),
                float(msg.linear_acceleration.z),
            ),
            stamp=time.time(),
        )
        with STATE.lock:
            STATE.imu = sample
        STATE.broadcast()

    def _tick(self) -> None:
        # Heartbeat publish of current status so the actuator stays in sync
        # even if GPS messages stop coming for a moment.
        with STATE.lock:
            armed = (STATE.settings.crossing_lat is not None
                     and STATE.settings.crossing_lon is not None)
            status = STATE.status
        if armed and status in ("safe", "incoming", "passing", "outgoing"):
            self.status_pub.publish(String(data=status))


# ---------- Flask app ------------------------------------------------------

def _here(*parts: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


app = Flask(
    __name__,
    template_folder=_here("templates"),
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    return jsonify(STATE.snapshot())


@app.route("/api/set_crossing", methods=["POST"])
def api_set_crossing():
    """Set the crossing's GPS coordinate.

    Body:
        {"mode": "current"}  -> latch the latest GPS reading as the crossing
        {"mode": "manual", "lat": <float>, "lon": <float>}
    """
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "manual")

    with STATE.lock:
        if mode == "current":
            if STATE.gps is None:
                return jsonify({"ok": False, "error": "no GPS fix yet"}), 400
            STATE.settings.crossing_lat = STATE.gps.lat
            STATE.settings.crossing_lon = STATE.gps.lon
        else:
            try:
                lat = float(body["lat"])
                lon = float(body["lon"])
            except (KeyError, TypeError, ValueError):
                return jsonify({"ok": False, "error": "lat and lon required"}), 400
            STATE.settings.crossing_lat = lat
            STATE.settings.crossing_lon = lon

    STATE.broadcast()
    return jsonify({"ok": True, "settings": asdict(STATE.settings)})


@app.route("/api/clear_crossing", methods=["POST"])
def api_clear_crossing():
    with STATE.lock:
        STATE.settings.crossing_lat = None
        STATE.settings.crossing_lon = None
        STATE.status = "unknown"
        STATE.distance_m = None
        STATE.has_passed = False
    STATE.broadcast()
    return jsonify({"ok": True})


@app.route("/api/set_thresholds", methods=["POST"])
def api_set_thresholds():
    body = request.get_json(silent=True) or {}
    try:
        incoming = float(body["incoming_m"])
        passing = float(body["passing_m"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "error": "incoming_m and passing_m required"}), 400
    if passing <= 0 or incoming <= 0 or passing >= incoming:
        return jsonify({"ok": False, "error": "need 0 < passing < incoming"}), 400
    with STATE.lock:
        STATE.settings.incoming_threshold_m = incoming
        STATE.settings.passing_threshold_m = passing
    STATE.broadcast()
    return jsonify({"ok": True, "settings": asdict(STATE.settings)})


@app.route("/stream")
def stream():
    """Server-Sent Events feed of the full shared state."""
    q: queue.Queue = queue.Queue(maxsize=64)
    with STATE.lock:
        STATE.subscribers.append(q)
    # Push the current snapshot immediately so a fresh tab gets state.
    q.put_nowait(STATE.snapshot())

    def gen():
        try:
            while True:
                try:
                    payload = q.get(timeout=15.0)
                    yield f"data: {json.dumps(payload)}\n\n"
                except queue.Empty:
                    # SSE keep-alive comment
                    yield ": keep-alive\n\n"
        finally:
            with STATE.lock:
                if q in STATE.subscribers:
                    STATE.subscribers.remove(q)

    return Response(gen(), mimetype="text/event-stream")


# ---------- entry point ----------------------------------------------------

def _spin_ros(node: Node) -> None:
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RailgateNode()

    ros_thread = threading.Thread(target=_spin_ros, args=(node,), daemon=True)
    ros_thread.start()

    host = os.environ.get("RAILGATE_HOST", "0.0.0.0")
    port = int(os.environ.get("RAILGATE_PORT", "5000"))
    try:
        # threaded=True so SSE clients don't block other requests.
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    finally:
        rclpy.shutdown()
        ros_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
