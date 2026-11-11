"""
bridge_client.py
================
Client library used by the Streamlit dashboard to communicate
with the serial_firebase_bridge.py FastAPI control server.

The bridge runs as a SEPARATE Python process (not inside Streamlit)
because Streamlit reruns the script on every interaction,
which would restart the serial connection every time.

The dashboard talks to the bridge over HTTP on localhost:8765.

Simple explanation
------------------
Think of this like a TV remote control:
  - The bridge is the TV (holds the serial connection, records data).
  - The dashboard is the remote (sends start/stop/status commands).
  - This file is the signal that travels between them.

If the bridge is not running, all methods return safe defaults
so the dashboard can still display historical data.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Bridge server address (runs locally on the same machine)
BRIDGE_HOST = "http://127.0.0.1"
BRIDGE_PORT = 8765
BRIDGE_BASE = f"{BRIDGE_HOST}:{BRIDGE_PORT}"


def _http_get(path: str, timeout: float = 2.0) -> Optional[Dict]:
    """Make a GET request to the bridge. Returns parsed JSON or None."""
    try:
        import urllib.request, json as _json
        url = f"{BRIDGE_BASE}{path}"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("Bridge GET %s failed: %s", path, exc)
        return None


def _http_post(path: str, body: Dict, timeout: float = 5.0) -> Optional[Dict]:
    """Make a POST request to the bridge. Returns parsed JSON or None."""
    try:
        import urllib.request, urllib.error, json as _json
        url = f"{BRIDGE_BASE}{path}"
        data = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("Bridge POST %s failed: %s", path, exc)
        return None


# =============================================================================
# Public API — used by dashboard.py
# =============================================================================

def bridge_is_running() -> bool:
    """Return True if the serial_firebase_bridge control server is reachable."""
    result = _http_get("/status", timeout=1.5)
    return result is not None


def get_live_status() -> Dict[str, Any]:
    """
    Return detailed live status including sample counts.
    Used to poll during active recording.
    """
    result = _http_get("/live_status")
    if result is None:
        return {
            "bridge_running": False,
            "status": "bridge_offline",
            "session_id": None,
            "patient_id": None,
            "valid_samples": 0,
            "rejected_samples": 0,
            "demo_mode": False,
        }
    return result


def get_bridge_status() -> Dict[str, Any]:
    """
    Return the bridge status dict.

    Fields:
        status         "recording" | "idle"
        session_id     Current session ID or None
        patient_id     Patient currently recording or None
        valid_samples  Number of valid samples received so far
    """
    result = _http_get("/status")
    if result is None:
        return {
            "status": "bridge_offline",
            "session_id": None,
            "patient_id": None,
            "valid_samples": 0,
        }
    return result


def start_recording(
    patient_id: str,
    serial_port: str = "",
    baud_rate: int = 115200,
    demo_mode: bool = False,
) -> Dict[str, Any]:
    """
    Tell the bridge to start a new recording session.

    Returns dict with keys: success (bool), message (str), patient_id (str).
    """
    result = _http_post("/start", {
        "patient_id":  patient_id,
        "serial_port": serial_port,
        "baud_rate":   baud_rate,
        "demo_mode":   demo_mode,
    })
    if result is None:
        return {
            "success": False,
            "message": (
                "Could not reach the bridge server. "
                "Run:  python serial_firebase_bridge.py --server"
            ),
        }
    return result


def stop_recording() -> Dict[str, Any]:
    """Tell the bridge to stop the current recording session."""
    result = _http_post("/stop", {})
    if result is None:
        return {"success": False, "message": "Bridge offline"}
    return result


def get_patients_from_bridge() -> List[str]:
    """Return patient IDs from Firestore via the bridge."""
    result = _http_get("/patients")
    if result and "patients" in result:
        return result["patients"]
    return []


def read_latest_from_rtdb(patient_id: str, session_id: str) -> Optional[Dict[str, Any]]:
    """
    Read the most recent sensor reading for a session.
    Talks directly to Firebase Realtime Database via the bridge.
    Returns None if unavailable.
    """
    result = _http_get(f"/latest/{patient_id}/{session_id}")
    return result


def get_session_history(patient_id: str) -> List[Dict[str, Any]]:
    """Return past gait sessions for a patient from Firestore via the bridge."""
    result = _http_get(f"/sessions/{patient_id}")
    if result and "sessions" in result:
        return result["sessions"]
    return []
