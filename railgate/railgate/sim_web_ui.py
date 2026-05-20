"""Railgate web UI — simulation build.

Same dashboard as :mod:`railgate.web_ui`, with an added ``DUMMY`` switch:

* ``DUMMY = True``  -> no GPS sensors required. The node animates a synthetic
  "train" (the circle on the map) running from a fixed start point, past a
  level crossing, and a bit further on. This exercises the SAFE / INCOMING /
  PASSING LEDs end to end: the SAFE LED comes on once the train has passed the
  crossing and moved clear of it.
* ``DUMMY = False`` -> behaves exactly like :mod:`railgate.web_ui` and consumes
  the real ``/mobile_sensor/{gps,imu}`` topics.

Override the flag at launch with the ``RAILGATE_DUMMY`` env var, e.g.::

    RAILGATE_DUMMY=false ros2 run railgate sim_web_ui

Run with::

    ros2 run railgate sim_web_ui
    python3 -m railgate.sim_web_ui
"""

from __future__ import annotations

import math
import os
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Reuse the shared state, geometry and the whole Flask app (routes + SSE) so the
# simulated and the real dashboards are byte-for-byte identical in the browser.
# Relative import when imported as a package (ros2 run / python3 -m railgate.*);
# plain import as a fallback when this file is run directly (python3 sim_web_ui.py).
try:
    from .web_ui import (
        STATE,
        app,
        GpsSample,
        ImuSample,
        haversine_m,
        classify,
        _spin_ros,
    )
except ImportError:
    from web_ui import (
        STATE,
        app,
        GpsSample,
        ImuSample,
        haversine_m,
        classify,
        _spin_ros,
    )


# --- simulation flag -------------------------------------------------------
# DUMMY=True  -> synthetic train data, no GPS sensors needed.
# DUMMY=False -> real-sensor behaviour, identical to railgate.web_ui.
DUMMY = True
_dummy_env = os.environ.get("RAILGATE_DUMMY")
if _dummy_env is not None:
    DUMMY = _dummy_env.strip().lower() not in ("0", "false", "no", "off")


# --- simulation scenario ---------------------------------------------------
SIM_START    = (23.726897, 90.388352)   # train spawn point
SIM_END      = (23.726881, 90.388167)   # nominal end of the run
SIM_CROSSING = (23.726789, 90.388241)   # level crossing being guarded

SIM_OVERSHOOT    = 0.8    # travel this fraction past SIM_END so the train
                          # clearly clears the crossing -> SAFE LED turns on
SIM_INCOMING_M   = 18.0   # thresholds tuned to this ~19 m demo path
SIM_PASSING_M    = 12.0
SIM_RUN_SECONDS  = 20.0   # wall-clock time for one start -> overshoot pass
SIM_HOLD_SECONDS = 3.0    # pause at the far (SAFE) end before looping
SIM_TICK_HZ      = 10.0

_T_MAX = 1.0 + SIM_OVERSHOOT


# --- simulation geometry ---------------------------------------------------

def _sim_position(p: float) -> tuple[float, float]:
    """Linear-interpolate the train position at path parameter ``p``.

    ``p`` runs 0 -> 1 from SIM_START to SIM_END, then on to ``_T_MAX``
    (the overshoot) so the train moves clear of the crossing.
    """
    lat = SIM_START[0] + p * (SIM_END[0] - SIM_START[0])
    lon = SIM_START[1] + p * (SIM_END[1] - SIM_START[1])
    return lat, lon


def _sim_progress(elapsed: float) -> float:
    """Map elapsed wall-clock seconds to a looping path parameter."""
    period = SIM_RUN_SECONDS + SIM_HOLD_SECONDS
    phase = elapsed % period
    if phase >= SIM_RUN_SECONDS:
        return _T_MAX                       # hold at the far (SAFE) end
    return (phase / SIM_RUN_SECONDS) * _T_MAX


def _path_yaw() -> float:
    lat0 = math.radians(SIM_START[0])
    d_north = SIM_END[0] - SIM_START[0]
    d_east = (SIM_END[1] - SIM_START[1]) * math.cos(lat0)
    return math.atan2(d_east, d_north)


_SIM_YAW = _path_yaw()


def _sim_imu(now: float) -> ImuSample:
    """A gentle synthetic IMU so the dashboard's IMU panel looks alive."""
    wobble = 0.04 * math.sin(now * 1.3)
    yaw = _SIM_YAW + wobble
    return ImuSample(
        qx=0.0,
        qy=0.0,
        qz=math.sin(yaw / 2.0),
        qw=math.cos(yaw / 2.0),
        ang_vel=(0.0, 0.0, 0.052 * math.cos(now * 1.3)),
        lin_acc=(0.0, 0.0, 9.81),
        stamp=now,
    )


# --- simulation ROS2 node --------------------------------------------------

class SimWebUiNode(Node):
    """Drives the dashboard from a synthetic train run (DUMMY mode)."""

    def __init__(self) -> None:
        super().__init__("railgate_sim_web_ui")

        self.status_pub = self.create_publisher(String, "/status", 10)

        # Pre-arm the crossing and tune thresholds for the ~19 m demo path.
        with STATE.lock:
            STATE.settings.crossing_lat = SIM_CROSSING[0]
            STATE.settings.crossing_lon = SIM_CROSSING[1]
            STATE.settings.incoming_threshold_m = SIM_INCOMING_M
            STATE.settings.passing_threshold_m = SIM_PASSING_M

        self._t0 = time.time()
        self.create_timer(1.0 / SIM_TICK_HZ, self._tick)

        self.get_logger().info(
            f"DUMMY simulation up: train {SIM_START} -> {SIM_END} "
            f"(+{SIM_OVERSHOOT:.0%} overshoot), crossing at {SIM_CROSSING}"
        )

    def _tick(self) -> None:
        now = time.time()
        p = _sim_progress(now - self._t0)
        lat, lon = _sim_position(p)

        with STATE.lock:
            STATE.gps = GpsSample(lat=lat, lon=lon, alt=10.0, stamp=now)
            STATE.imu = _sim_imu(now)
            s = STATE.settings
            armed = s.crossing_lat is not None and s.crossing_lon is not None
            if armed:
                d = haversine_m(lat, lon, s.crossing_lat, s.crossing_lon)
                STATE.distance_m = d
                STATE.status, STATE.has_passed = classify(d, s, STATE.has_passed)
            else:
                STATE.distance_m = None
                STATE.status = "unknown"
                STATE.has_passed = False
            status = STATE.status if armed else None

        if status is not None:
            self.status_pub.publish(String(data=status))
        STATE.broadcast()


# --- entry point -----------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)

    if DUMMY:
        node = SimWebUiNode()
    else:
        # Real-sensor path: reuse the production node verbatim.
        try:
            from .web_ui import RailgateNode
        except ImportError:
            from web_ui import RailgateNode
        node = RailgateNode()
        node.get_logger().info("DUMMY=False -> using real /mobile_sensor topics")

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
