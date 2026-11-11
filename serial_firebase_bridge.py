"""
serial_firebase_bridge.py
=========================
ESP32 → Python → Firebase Bridge for the OA Patient Monitoring System.

What this script does (simple explanation)
-------------------------------------------
1. Opens the USB serial port where your ESP32 is connected.
2. Reads the JSON sensor data the ESP32 sends 50 times per second.
3. Sends that data to Firebase in two ways:
   a) Firebase Realtime Database  — for LIVE display on the website.
   b) Firebase Firestore          — for PERMANENT session records.
4. Saves a local CSV backup so data is never lost if the internet drops.
5. Extracts gait features and saves them to Firestore when the session ends.

How to run
----------
    # Activate your virtual environment first:
    source venv/bin/activate

    # Then run:
    python serial_firebase_bridge.py

    # With options:
    python serial_firebase_bridge.py --port /dev/cu.usbserial-0001 --patient patient_001
    python serial_firebase_bridge.py --demo         # run without hardware

Required .env variables (copy from .env.example)
-------------------------------------------------
    FIREBASE_CREDENTIALS_PATH = secrets/firebase-key.json
    FIREBASE_DATABASE_URL      = https://oa-screening-system-default-rtdb.firebaseio.com
    SERIAL_PORT                = /dev/cu.usbserial-0001   (or COM3 on Windows)
    BAUD_RATE                  = 115200
    PATIENT_ID                 = patient_001
    DEMO_MODE                  = false

Architecture note
-----------------
    Browser buttons cannot directly stop/start a Python process.
    This bridge exposes a tiny FastAPI control server on port 8765
    so the Streamlit dashboard (or any local client) can send
    start/stop/status commands via HTTP:

        POST http://localhost:8765/start
        POST http://localhost:8765/stop
        GET  http://localhost:8765/status
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Load environment variables from .env
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    # Fallback parser — works without python-dotenv installed
    _env_path = _PROJECT_ROOT / ".env"
    if _env_path.exists():
        with open(_env_path, "r") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k = _k.strip()
                    _v = _v.strip().strip("'\"")
                    if _k not in os.environ:
                        os.environ[_k] = _v

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_PROJECT_ROOT / "bridge.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("bridge")

# ---------------------------------------------------------------------------
# Configuration — read from environment / command line
# ---------------------------------------------------------------------------

FIREBASE_CREDENTIALS_PATH = os.environ.get(
    "FIREBASE_CREDENTIALS_PATH", "secrets/firebase-key.json"
)
FIREBASE_DATABASE_URL = os.environ.get(
    "FIREBASE_DATABASE_URL",
    "https://oa-screening-system-default-rtdb.firebaseio.com",
)
SERIAL_PORT  = os.environ.get("SERIAL_PORT", "")
BAUD_RATE    = int(os.environ.get("BAUD_RATE", os.environ.get("SERIAL_BAUD_RATE", "115200")))
PATIENT_ID   = os.environ.get("PATIENT_ID", "patient_001")
DEMO_MODE    = os.environ.get("DEMO_MODE", "false").lower() in ("true", "1", "yes")

# How often to push "latest" to Realtime DB (seconds)
LATEST_PUSH_INTERVAL = 0.20   # 200 ms → ~5 updates/second

# Batch size: collect N samples then push them together as one "chunk"
CHUNK_BATCH_SIZE = 50          # ~1 second of data at 50 Hz

# ---------------------------------------------------------------------------
# Firebase initialisation
# ---------------------------------------------------------------------------

_firebase_app = None
_rtdb = None           # Realtime Database module
_fs_client = None      # Firestore client

def _init_firebase() -> bool:
    """
    Initialise Firebase Admin SDK.

    Uses the service account JSON file — credentials are NEVER sent to the browser.
    Returns True on success.
    """
    global _firebase_app, _rtdb, _fs_client

    cred_path = _PROJECT_ROOT / FIREBASE_CREDENTIALS_PATH
    if not cred_path.exists():
        logger.error(
            "Firebase credentials not found at: %s\n"
            "  1. Go to https://console.firebase.google.com\n"
            "  2. Project Settings → Service Accounts → Generate new private key\n"
            "  3. Save the JSON file as: %s\n"
            "  4. This file is in .gitignore — it will NOT be uploaded to GitHub.",
            cred_path, cred_path,
        )
        return False

    try:
        import firebase_admin
        from firebase_admin import credentials, db as rtdb_module, firestore

        if not firebase_admin._apps:
            cred = credentials.Certificate(str(cred_path))
            _firebase_app = firebase_admin.initialize_app(cred, {
                "databaseURL": FIREBASE_DATABASE_URL,
            })
            logger.info("Firebase Admin SDK initialised (project: oa-screening-system).")
        else:
            _firebase_app = firebase_admin.get_app()
            logger.info("Re-using existing Firebase app.")

        _rtdb = rtdb_module   # used for Realtime Database writes
        _fs_client = firestore.client()
        logger.info("Firebase Realtime Database + Firestore ready.")
        return True

    except ImportError:
        logger.error(
            "firebase-admin is not installed.\n"
            "Run:  pip install firebase-admin"
        )
        return False
    except Exception as exc:
        logger.error("Firebase init failed: %s", exc)
        return False


# ===========================================================================
# Session ID generator
# ===========================================================================

def make_session_id(patient_id: str) -> str:
    """
    Create a unique, human-readable session ID.

    Example: patient_001_20261111_143022
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{patient_id}_{ts}"


# ===========================================================================
# Gait feature extraction
# ===========================================================================

def extract_gait_features(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute biomechanical summary features from a list of sensor samples.

    This processes raw ADC readings — it does NOT convert to Newtons
    (which requires individual sensor calibration) and does NOT diagnose OA.

    Parameters
    ----------
    samples : list of dicts with keys:
        heel, midfoot, forefoot, acc_x_g, acc_y_g, acc_z_g,
        gyro_x_dps, gyro_y_dps, gyro_z_dps, timestamp_ms

    Returns
    -------
    dict  Feature dictionary ready for Firestore storage.
    """
    if not samples:
        return {}

    try:
        import numpy as np
    except ImportError:
        logger.warning("numpy not installed — skipping feature extraction.")
        return {}

    n = len(samples)

    heel     = np.array([s.get("heel", 0)     for s in samples], dtype=float)
    midfoot  = np.array([s.get("midfoot", 0)  for s in samples], dtype=float)
    forefoot = np.array([s.get("forefoot", 0) for s in samples], dtype=float)

    ax = np.array([s.get("acc_x_g", 0)    for s in samples], dtype=float)
    ay = np.array([s.get("acc_y_g", 0)    for s in samples], dtype=float)
    az = np.array([s.get("acc_z_g", 0)    for s in samples], dtype=float)
    gx = np.array([s.get("gyro_x_dps", 0) for s in samples], dtype=float)
    gy = np.array([s.get("gyro_y_dps", 0) for s in samples], dtype=float)
    gz = np.array([s.get("gyro_z_dps", 0) for s in samples], dtype=float)

    # Acceleration magnitude per sample
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

    # Relative loading distribution (percentage of total foot load)
    total_load = heel + midfoot + forefoot
    total_load_sum = float(total_load.sum()) if total_load.sum() > 0 else 1.0
    heel_pct    = float(heel.sum()     / total_load_sum * 100)
    midfoot_pct = float(midfoot.sum()  / total_load_sum * 100)
    fore_pct    = float(forefoot.sum() / total_load_sum * 100)

    # Loading threshold: ADC > 100 considered "foot loaded"
    LOAD_THRESHOLD = 100
    heel_contact_frames    = int(np.sum(heel     > LOAD_THRESHOLD))
    midfoot_contact_frames = int(np.sum(midfoot  > LOAD_THRESHOLD))
    fore_contact_frames    = int(np.sum(forefoot > LOAD_THRESHOLD))

    # Timestamps
    timestamps = [s.get("timestamp_ms", 0) for s in samples]
    duration_ms = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0

    features = {
        # --- FSR pressure statistics (raw ADC 0-4095) ---
        "heel_mean_adc":       float(np.mean(heel)),
        "heel_max_adc":        float(np.max(heel)),
        "heel_min_adc":        float(np.min(heel)),
        "heel_std_adc":        float(np.std(heel)),
        "midfoot_mean_adc":    float(np.mean(midfoot)),
        "midfoot_max_adc":     float(np.max(midfoot)),
        "midfoot_min_adc":     float(np.min(midfoot)),
        "midfoot_std_adc":     float(np.std(midfoot)),
        "forefoot_mean_adc":   float(np.mean(forefoot)),
        "forefoot_max_adc":    float(np.max(forefoot)),
        "forefoot_min_adc":    float(np.min(forefoot)),
        "forefoot_std_adc":    float(np.std(forefoot)),

        # --- Loading distribution (%) ---
        "heel_load_percent":     round(heel_pct,    1),
        "midfoot_load_percent":  round(midfoot_pct, 1),
        "forefoot_load_percent": round(fore_pct,    1),

        # --- Contact frame counts ---
        "heel_contact_frames":     heel_contact_frames,
        "midfoot_contact_frames":  midfoot_contact_frames,
        "forefoot_contact_frames": fore_contact_frames,

        # --- Accelerometer (g-force) ---
        "acc_x_mean_g":    float(np.mean(ax)),
        "acc_x_std_g":     float(np.std(ax)),
        "acc_y_mean_g":    float(np.mean(ay)),
        "acc_y_std_g":     float(np.std(ay)),
        "acc_z_mean_g":    float(np.mean(az)),
        "acc_z_std_g":     float(np.std(az)),
        "acc_magnitude_mean_g": float(np.mean(acc_mag)),
        "acc_magnitude_std_g":  float(np.std(acc_mag)),
        "acc_magnitude_max_g":  float(np.max(acc_mag)),

        # --- Gyroscope (degrees per second) ---
        "gyro_x_mean_dps": float(np.mean(gx)),
        "gyro_x_std_dps":  float(np.std(gx)),
        "gyro_y_mean_dps": float(np.mean(gy)),
        "gyro_y_std_dps":  float(np.std(gy)),
        "gyro_z_mean_dps": float(np.mean(gz)),
        "gyro_z_std_dps":  float(np.std(gz)),

        # --- Session metadata ---
        "total_samples":      n,
        "duration_ms":        int(duration_ms),
        "nominal_sample_hz":  50,

        # --- Disclaimer (always present) ---
        "disclaimer": (
            "Raw ADC values only. No Newton conversion without calibration. "
            "Research prototype — not a clinical diagnostic tool."
        ),
    }

    logger.info(
        "Features extracted: %d samples | duration %.1f s | "
        "heel %.0f%% | midfoot %.0f%% | forefoot %.0f%%",
        n, duration_ms / 1000.0, heel_pct, midfoot_pct, fore_pct,
    )
    return features


# ===========================================================================
# CSV backup writer
# ===========================================================================

class CSVBackupWriter:
    """
    Writes each validated sensor sample to a local CSV file.

    The CSV is created the moment recording starts so data is preserved
    even if Firebase is unreachable or the script crashes.
    """

    FIELDNAMES = [
        "seq", "host_timestamp_utc", "timestamp_ms",
        "heel", "midfoot", "forefoot",
        "acc_x_g", "acc_y_g", "acc_z_g",
        "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
        "patient_id", "session_id", "data_source",
    ]

    def __init__(self, filepath: Path, patient_id: str, session_id: str):
        self.filepath   = filepath
        self.patient_id = patient_id
        self.session_id = session_id
        self._file      = None
        self._writer    = None

    def open(self) -> None:
        """Open the CSV file and write the header row."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(self.filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        logger.info("CSV backup opened: %s", self.filepath)

    def write(self, sample: Dict[str, Any], seq: int, data_source: str = "real") -> None:
        """Append one sample row to the CSV."""
        if self._writer is None:
            return
        row = {
            "seq":               seq,
            "host_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timestamp_ms":      sample.get("timestamp_ms", 0),
            "heel":              sample.get("heel",         sample.get("Heel_FSR", 0)),
            "midfoot":           sample.get("midfoot",      sample.get("Ball_FSR", 0)),
            "forefoot":          sample.get("forefoot",     sample.get("Toe_FSR", 0)),
            "acc_x_g":           sample.get("acc_x_g",     sample.get("Accel_X", 0.0)),
            "acc_y_g":           sample.get("acc_y_g",     sample.get("Accel_Y", 0.0)),
            "acc_z_g":           sample.get("acc_z_g",     sample.get("Accel_Z", 0.0)),
            "gyro_x_dps":        sample.get("gyro_x_dps",  sample.get("Gyro_X", 0.0)),
            "gyro_y_dps":        sample.get("gyro_y_dps",  sample.get("Gyro_Y", 0.0)),
            "gyro_z_dps":        sample.get("gyro_z_dps",  sample.get("Gyro_Z", 0.0)),
            "patient_id":        self.patient_id,
            "session_id":        self.session_id,
            "data_source":       data_source,
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            logger.info("CSV backup closed: %s", self.filepath)


# ===========================================================================
# Realtime Database uploader (runs in a background thread)
# ===========================================================================

class RealtimeDatabaseUploader(threading.Thread):
    """
    Background thread that pushes sensor data to Firebase Realtime Database.

    Two update frequencies:
      - latest/  : Updated every 200 ms with the most recent reading.
      - chunks/  : Updated every 50 samples (~1 s) with a batch of readings.

    Simple explanation: "latest" is like a live scoreboard that keeps changing.
    "chunks" is like saving snapshots to a photo album — each snapshot is kept forever.
    """

    def __init__(
        self,
        patient_id: str,
        session_id: str,
        demo_mode: bool = False,
    ):
        super().__init__(name="RTDBUploader", daemon=True)
        self.patient_id  = patient_id
        self.session_id  = session_id
        self.demo_mode   = demo_mode

        # Queues for communicating with the main thread
        self._latest_queue: queue.Queue = queue.Queue(maxsize=10)
        self._chunk_queue:  queue.Queue = queue.Queue(maxsize=200)
        self._stop_event = threading.Event()

        self._chunk_index = 0   # monotonically increasing chunk counter

        # Path prefix in Realtime Database
        self._base_path = f"live_sessions/{patient_id}/{session_id}"

    # ------------------------------------------------------------------
    # API — called from the main thread
    # ------------------------------------------------------------------

    def push_latest(self, sample_dict: Dict[str, Any]) -> None:
        """Non-blocking: drop oldest if queue is full (rate limiting)."""
        try:
            self._latest_queue.put_nowait(sample_dict)
        except queue.Full:
            try:
                self._latest_queue.get_nowait()   # discard oldest
                self._latest_queue.put_nowait(sample_dict)
            except queue.Empty:
                pass

    def push_to_chunk_buffer(self, sample_dict: Dict[str, Any]) -> None:
        """Buffer a sample for the next batch upload."""
        try:
            self._chunk_queue.put_nowait(sample_dict)
        except queue.Full:
            logger.warning("RTDB chunk queue full — sample dropped.")

    def stop(self) -> None:
        """Signal the thread to finish."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def run(self) -> None:
        last_latest_push = 0.0
        chunk_buffer: List[Dict] = []

        while not self._stop_event.is_set():
            now = time.time()

            # --- Drain latest queue, keep only the most recent ---
            latest_sample = None
            while True:
                try:
                    latest_sample = self._latest_queue.get_nowait()
                except queue.Empty:
                    break

            # --- Push latest to Realtime DB every 200 ms ---
            if latest_sample and (now - last_latest_push) >= LATEST_PUSH_INTERVAL:
                self._write_latest(latest_sample)
                last_latest_push = now

            # --- Drain chunk buffer ---
            while True:
                try:
                    chunk_buffer.append(self._chunk_queue.get_nowait())
                except queue.Empty:
                    break

            # --- Upload chunk when batch is full ---
            if len(chunk_buffer) >= CHUNK_BATCH_SIZE:
                self._write_chunk(chunk_buffer[:CHUNK_BATCH_SIZE])
                chunk_buffer = chunk_buffer[CHUNK_BATCH_SIZE:]

            time.sleep(0.05)   # 50 ms loop interval

        # Flush remaining chunk buffer before exit
        if chunk_buffer:
            self._write_chunk(chunk_buffer)
            logger.info("Flushed %d remaining samples to RTDB chunk.", len(chunk_buffer))

    # ------------------------------------------------------------------
    # Private RTDB write helpers
    # ------------------------------------------------------------------

    def _write_latest(self, sample: Dict[str, Any]) -> None:
        """Write a single sample to live_sessions/{patient}/{session}/latest."""
        if self.demo_mode:
            return   # don't write simulated data to production DB

        if _rtdb is None:
            return

        try:
            ref = _rtdb.reference(f"{self._base_path}/latest")
            ref.set({
                "timestamp_ms":  sample.get("timestamp_ms", 0),
                "server_ts":     {".sv": "timestamp"},   # server-side timestamp
                "heel":          sample.get("heel",        0),
                "midfoot":       sample.get("midfoot",     0),
                "forefoot":      sample.get("forefoot",    0),
                "acc_x_g":       sample.get("acc_x_g",    0.0),
                "acc_y_g":       sample.get("acc_y_g",    0.0),
                "acc_z_g":       sample.get("acc_z_g",    0.0),
                "gyro_x_dps":    sample.get("gyro_x_dps", 0.0),
                "gyro_y_dps":    sample.get("gyro_y_dps", 0.0),
                "gyro_z_dps":    sample.get("gyro_z_dps", 0.0),
                "patient_id":    self.patient_id,
                "session_id":    self.session_id,
            })
        except Exception as exc:
            logger.debug("RTDB latest write failed: %s", exc)

    def _write_chunk(self, samples: List[Dict[str, Any]]) -> None:
        """Append a batch of samples to live_sessions/{patient}/{session}/chunks."""
        if self.demo_mode:
            return

        if _rtdb is None or not samples:
            return

        try:
            chunk_key = f"chunk_{self._chunk_index:06d}"
            self._chunk_index += 1

            chunk_data = {
                "chunk_index":    self._chunk_index,
                "sample_count":   len(samples),
                "server_received": {".sv": "timestamp"},
                "samples": {
                    f"s{i:04d}": {
                        "seq":          s.get("seq", i),
                        "timestamp_ms": s.get("timestamp_ms", 0),
                        "heel":         s.get("heel",        0),
                        "midfoot":      s.get("midfoot",     0),
                        "forefoot":     s.get("forefoot",    0),
                        "acc_x_g":      s.get("acc_x_g",    0.0),
                        "acc_y_g":      s.get("acc_y_g",    0.0),
                        "acc_z_g":      s.get("acc_z_g",    0.0),
                        "gyro_x_dps":   s.get("gyro_x_dps", 0.0),
                        "gyro_y_dps":   s.get("gyro_y_dps", 0.0),
                        "gyro_z_dps":   s.get("gyro_z_dps", 0.0),
                    }
                    for i, s in enumerate(samples)
                },
            }

            ref = _rtdb.reference(f"{self._base_path}/chunks/{chunk_key}")
            ref.set(chunk_data)
            logger.debug("RTDB chunk %s written (%d samples).", chunk_key, len(samples))

        except Exception as exc:
            logger.error("RTDB chunk write failed: %s", exc)


# ===========================================================================
# Firestore session manager
# ===========================================================================

class FirestoreSession:
    """
    Manages the Firestore session document lifecycle.

    Uses the EXISTING patients collection — does not create a new one.

    Firestore path: patients/{patientId}/gait_sessions/{sessionId}
    """

    def __init__(self, patient_id: str, session_id: str, demo_mode: bool = False):
        self.patient_id  = patient_id
        self.session_id  = session_id
        self.demo_mode   = demo_mode
        self._start_time = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _session_ref(self):
        """Return the Firestore document reference for this session."""
        if _fs_client is None:
            return None
        return (
            _fs_client
            .collection("patients")
            .document(self.patient_id)
            .collection("gait_sessions")
            .document(self.session_id)
        )

    def _safe_write(self, operation_name: str, fn) -> bool:
        """Call fn() with up to 3 retries."""
        if self.demo_mode or _fs_client is None:
            return True   # skip writes in demo mode / no Firebase

        for attempt in range(1, 4):
            try:
                fn()
                return True
            except Exception as exc:
                logger.warning(
                    "[Firestore %s] Attempt %d/3 failed: %s",
                    operation_name, attempt, exc,
                )
                time.sleep(1.5)
        logger.error("[Firestore %s] All 3 attempts failed.", operation_name)
        return False

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create(self, csv_filename: str, data_source: str = "real") -> bool:
        """Create the Firestore session document at recording start."""
        # Skip completely in demo mode
        if self.demo_mode or _fs_client is None:
            logger.info("[Demo/Offline] Skipping Firestore session create.")
            return True

        try:
            from firebase_admin import firestore as _fsa
        except ImportError:
            logger.warning("firebase_admin not installed — skipping Firestore create.")
            return True

        def _do():
            ref = self._session_ref()
            if ref is None:
                return
            # Check if already exists (prevent duplicate sessions)
            existing = ref.get()
            if existing.exists:
                logger.warning(
                    "Session '%s' already exists in Firestore — skipping create.",
                    self.session_id,
                )
                return

            ref.set({
                "patient_id":       self.patient_id,
                "session_id":       self.session_id,
                "start_time":       _fsa.SERVER_TIMESTAMP,
                "end_time":         None,
                "duration_seconds": 0,
                "sampling_hz":      50,
                "total_samples":    0,
                "valid_samples":    0,
                "connection_status": "recording",
                "data_source":      data_source,
                "csv_filename":     csv_filename,
                "features":         None,
                "data_quality":     {},
                "created_at":       _fsa.SERVER_TIMESTAMP,
                "disclaimer": (
                    "Research prototype. Not a clinical diagnostic device. "
                    "Raw sensor data only."
                ),
            })
            logger.info(
                "Firestore session created: patients/%s/gait_sessions/%s",
                self.patient_id, self.session_id,
            )

        return self._safe_write("create_session", _do)

    def update_sample_count(self, total: int, valid: int) -> bool:
        """Periodically update the sample counter during recording."""
        def _do():
            ref = self._session_ref()
            if ref:
                ref.update({"total_samples": total, "valid_samples": valid})
        return self._safe_write("update_count", _do)

    def complete(
        self,
        total_samples: int,
        valid_samples: int,
        features: Optional[Dict[str, Any]] = None,
        data_quality: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Mark the session as completed and save final statistics."""
        if self.demo_mode or _fs_client is None:
            logger.info("[Demo/Offline] Skipping Firestore session complete.")
            return True

        try:
            from firebase_admin import firestore as _fsa
        except ImportError:
            return True

        end_time = datetime.now(timezone.utc)
        duration = (end_time - self._start_time).total_seconds()

        def _do():
            ref = self._session_ref()
            if ref is None:
                return

            update_data: Dict[str, Any] = {
                "end_time":         end_time.isoformat(),
                "duration_seconds": round(duration, 2),
                "total_samples":    total_samples,
                "valid_samples":    valid_samples,
                "connection_status": "completed",
                "completed_at":     _fsa.SERVER_TIMESTAMP,
            }
            if features:
                update_data["features"] = features
            if data_quality:
                update_data["data_quality"] = data_quality

            ref.update(update_data)
            logger.info(
                "Firestore session completed: %s (%.1f s, %d valid samples)",
                self.session_id, duration, valid_samples,
            )

        return self._safe_write("complete_session", _do)

    def mark_failed(self, reason: str = "unknown") -> bool:
        """Mark the session as failed in Firestore."""
        from firebase_admin import firestore as _fsa

        def _do():
            ref = self._session_ref()
            if ref:
                ref.update({
                    "connection_status": "failed",
                    "failure_reason":    reason,
                    "completed_at":      _fsa.SERVER_TIMESTAMP,
                })
        return self._safe_write("mark_failed", _do)

    def set_rtdb_status(self, status: str) -> None:
        """Update the status field in Realtime Database."""
        if self.demo_mode or _rtdb is None:
            return
        try:
            ref = _rtdb.reference(
                f"live_sessions/{self.patient_id}/{self.session_id}/status"
            )
            ref.set({
                "value":      status,
                "updated_at": {".sv": "timestamp"},
                "patient_id": self.patient_id,
                "session_id": self.session_id,
            })
        except Exception as exc:
            logger.debug("RTDB status write failed: %s", exc)


# ===========================================================================
# CSV parser for prompt-specified field names
# ===========================================================================

# Map from prompt field names → firmware JSON field names
_FIELD_MAP = {
    "timestamp_ms": "timestamp_ms",
    "Heel_FSR":     "heel",
    "Ball_FSR":     "midfoot",
    "Toe_FSR":      "forefoot",
    "Accel_X":      "acc_x_g",
    "Accel_Y":      "acc_y_g",
    "Accel_Z":      "acc_z_g",
    "Gyro_X":       "gyro_x_dps",
    "Gyro_Y":       "gyro_y_dps",
    "Gyro_Z":       "gyro_z_dps",
}

EXPECTED_NUMERIC_FIELDS = [
    "timestamp_ms", "heel", "midfoot", "forefoot",
    "acc_x_g", "acc_y_g", "acc_z_g",
    "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
]

def _parse_line(raw_line: str) -> Optional[Dict[str, Any]]:
    """
    Parse one line from the ESP32.

    The firmware outputs JSON.
    This function also handles the CSV format specified in the prompt
    in case you switch firmware later.

    Returns a normalised dict with EXPECTED_NUMERIC_FIELDS, or None.
    """
    raw_line = raw_line.strip()

    # Skip empty lines and comments
    if not raw_line or raw_line.startswith("//") or raw_line.startswith("#"):
        return None

    # Try JSON first (the firmware outputs JSON)
    if raw_line.startswith("{"):
        try:
            data = json.loads(raw_line)
            if not isinstance(data, dict):
                return None

            # Normalise field names (handle both firmware names and prompt CSV names)
            normalised: Dict[str, Any] = {}
            for prompt_key, internal_key in _FIELD_MAP.items():
                if prompt_key in data:
                    normalised[internal_key] = data[prompt_key]
            # Also copy firmware fields directly
            for key in EXPECTED_NUMERIC_FIELDS:
                if key in data and key not in normalised:
                    normalised[key] = data[key]

            # Validate all required fields are present and numeric
            for field in EXPECTED_NUMERIC_FIELDS:
                if field not in normalised:
                    return None
                try:
                    normalised[field] = float(normalised[field])
                except (ValueError, TypeError):
                    return None

            return normalised

        except json.JSONDecodeError:
            pass   # fall through to CSV parser

    # Try CSV format: timestamp_ms, Heel_FSR, Ball_FSR, Toe_FSR, Accel_X/Y/Z, Gyro_X/Y/Z
    parts = raw_line.split(",")
    if len(parts) == 10:
        try:
            keys_ordered = list(_FIELD_MAP.keys())
            vals = [float(p.strip()) for p in parts]
            raw = dict(zip(keys_ordered, vals))
            # Rename to internal field names
            normalised = {_FIELD_MAP[k]: v for k, v in raw.items()}
            return normalised
        except ValueError:
            pass

    return None


# ===========================================================================
# Main bridge class
# ===========================================================================

class ESP32FirebaseBridge:
    """
    Connects the ESP32 USB serial stream to Firebase.

    Usage:
        bridge = ESP32FirebaseBridge(patient_id="patient_001", port="/dev/cu.usbserial-0001")
        bridge.run()   # blocks until Ctrl+C or bridge.stop()
    """

    def __init__(
        self,
        patient_id: str,
        port: str,
        baud: int = 115200,
        demo_mode: bool = False,
    ):
        self.patient_id = patient_id
        self.port       = port
        self.baud       = baud
        self.demo_mode  = demo_mode

        self._stop_event = threading.Event()

        # Stats
        self._total_lines   = 0
        self._valid_samples = 0
        self._rejected      = 0
        self._latest_sample: Optional[Dict[str, Any]] = None

        # Set during run()
        self._session_id: Optional[str] = None
        self._csv_writer: Optional[CSVBackupWriter] = None
        self._rtdb_uploader: Optional[RealtimeDatabaseUploader] = None
        self._fs_session: Optional[FirestoreSession] = None

        logger.info(
            "Bridge initialised | patient=%s | port=%s | demo=%s",
            patient_id, port or "(auto)", demo_mode,
        )

    def stop(self) -> None:
        """Signal the bridge to shut down cleanly."""
        self._stop_event.set()
        logger.info("Stop signal received.")

    def run(self) -> None:
        """Main blocking loop — runs until stop() is called or Ctrl+C."""
        # 1. Firebase — skip entirely in demo mode
        if not self.demo_mode:
            if not _init_firebase():
                logger.error("Cannot connect to Firebase. Exiting.")
                return

        # 2. Session setup
        self._session_id = make_session_id(self.patient_id)
        logger.info("Session ID: %s", self._session_id)

        # 3. CSV backup
        csv_path = (
            _PROJECT_ROOT / "data" / "raw"
            / f"{self._session_id}.csv"
        )
        data_source = "simulated" if self.demo_mode else "real"
        self._csv_writer = CSVBackupWriter(csv_path, self.patient_id, self._session_id)
        self._csv_writer.open()

        # 4. Realtime Database uploader thread
        self._rtdb_uploader = RealtimeDatabaseUploader(
            self.patient_id, self._session_id, self.demo_mode
        )
        self._rtdb_uploader.start()

        # 5. Firestore session document
        self._fs_session = FirestoreSession(
            self.patient_id, self._session_id, self.demo_mode
        )
        self._fs_session.create(csv_filename=csv_path.name, data_source=data_source)
        self._fs_session.set_rtdb_status("recording")

        # 6. Serial / simulated receiver
        all_samples: List[Dict[str, Any]] = []
        last_count_update = time.time()

        try:
            if self.demo_mode:
                self._run_demo(all_samples, last_count_update)
            else:
                self._run_real(all_samples, last_count_update)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down.")
        except Exception as exc:
            logger.exception("Unexpected error in bridge loop: %s", exc)
            if self._fs_session:
                self._fs_session.mark_failed(str(exc))
        finally:
            self._shutdown(all_samples)

    # ------------------------------------------------------------------
    # Real hardware loop
    # ------------------------------------------------------------------

    def _run_real(self, all_samples: List, last_count_update: float) -> None:
        import serial
        import serial.tools.list_ports

        chosen_port = self.port
        if not chosen_port:
            ports = [p.device for p in serial.tools.list_ports.comports()]
            if not ports:
                raise RuntimeError(
                    "No serial port found. Connect the ESP32 and set SERIAL_PORT in .env"
                )
            chosen_port = ports[0]
            logger.info("Auto-selected serial port: %s", chosen_port)

        logger.info("Opening serial port %s at %d baud …", chosen_port, self.baud)
        ser = serial.Serial(port=chosen_port, baudrate=self.baud, timeout=2.0)
        time.sleep(2.0)       # allow ESP32 to reset after DTR toggle
        ser.reset_input_buffer()
        logger.info("ESP32 connected. Recording started.")

        self._fs_session.set_rtdb_status("connected")

        while not self._stop_event.is_set():
            try:
                raw_bytes = ser.readline()
            except serial.SerialException as exc:
                logger.error("Serial read error: %s", exc)
                self._fs_session.set_rtdb_status("disconnected")
                # Reconnect loop
                ser.close()
                for _ in range(10):
                    if self._stop_event.is_set():
                        break
                    time.sleep(2.0)
                    try:
                        ser.open()
                        ser.reset_input_buffer()
                        self._fs_session.set_rtdb_status("reconnected")
                        logger.info("Serial reconnected.")
                        break
                    except Exception:
                        pass
                else:
                    logger.error("Could not reconnect after 10 attempts. Stopping.")
                    break
                continue

            if not raw_bytes:
                continue

            self._total_lines += 1
            raw_line = raw_bytes.decode("utf-8", errors="replace")
            sample = _parse_line(raw_line)

            if sample is None:
                self._rejected += 1
                continue

            self._valid_samples += 1
            sample["seq"] = self._valid_samples
            self._latest_sample = dict(sample, server_ts=int(time.time() * 1000))
            all_samples.append(sample)

            # Write CSV backup immediately
            self._csv_writer.write(sample, self._valid_samples, "real")

            # Push to Realtime DB
            self._rtdb_uploader.push_latest(sample)
            self._rtdb_uploader.push_to_chunk_buffer(sample)

            # Periodically update Firestore sample count (every 5 s)
            if time.time() - last_count_update > 5.0:
                self._fs_session.update_sample_count(
                    self._total_lines, self._valid_samples
                )
                last_count_update = time.time()
                logger.info(
                    "Recording … valid=%d | rejected=%d",
                    self._valid_samples, self._rejected,
                )

        ser.close()

    # ------------------------------------------------------------------
    # Demo (simulated) loop
    # ------------------------------------------------------------------

    def _run_demo(self, all_samples: List, last_count_update: float) -> None:
        """Generate synthetic sensor readings for testing without hardware."""
        import math
        import random

        logger.info("[DEMO MODE] Generating synthetic gait data at 50 Hz.")
        logger.info("[DEMO MODE] Data is NOT written to Firebase (keeping production DB clean).")
        logger.info("[DEMO MODE] Press Ctrl+C to stop.")

        t = 0
        interval = 1.0 / 50.0   # 50 Hz

        while not self._stop_event.is_set():
            phase = (t % 50) / 50.0  # gait cycle phase 0→1

            sample = {
                "timestamp_ms": t * 20,
                "heel":         max(0, min(4095, int(800 * math.exp(-((phase - 0.15)**2) / 0.01) + random.gauss(0, 15)))),
                "midfoot":      max(0, min(4095, int(600 * math.exp(-((phase - 0.45)**2) / 0.02) + random.gauss(0, 12)))),
                "forefoot":     max(0, min(4095, int(850 * math.exp(-((phase - 0.75)**2) / 0.01) + random.gauss(0, 15)))),
                "acc_x_g":      round(0.05 * math.sin(2 * math.pi * phase)      + random.gauss(0, 0.02), 4),
                "acc_y_g":      round(0.08 * math.sin(2 * math.pi * phase + 0.5) + random.gauss(0, 0.02), 4),
                "acc_z_g":      round(0.98                                        + random.gauss(0, 0.02), 4),
                "gyro_x_dps":   round(2.0 * math.sin(2 * math.pi * phase)        + random.gauss(0, 0.3), 4),
                "gyro_y_dps":   round(1.5 * math.cos(2 * math.pi * phase)        + random.gauss(0, 0.2), 4),
                "gyro_z_dps":   round(0.8 * math.sin(4 * math.pi * phase)        + random.gauss(0, 0.15), 4),
                "seq":          t + 1,
            }

            self._valid_samples += 1
            self._latest_sample = dict(sample, server_ts=int(time.time() * 1000))
            all_samples.append(sample)
            self._csv_writer.write(sample, self._valid_samples, "simulated")

            if t % 50 == 0:
                logger.info(
                    "[DEMO] Sample %d | heel=%d | midfoot=%d | forefoot=%d | acc_z=%.2f g",
                    self._valid_samples,
                    sample["heel"], sample["midfoot"], sample["forefoot"],
                    sample["acc_z_g"],
                )

            t += 1
            time.sleep(interval)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self, all_samples: List[Dict[str, Any]]) -> None:
        """Clean up resources and save final session data."""
        logger.info("Shutting down bridge …")

        # 1. Stop RTDB uploader thread
        if self._rtdb_uploader:
            self._rtdb_uploader.stop()
            self._rtdb_uploader.join(timeout=10)

        # 2. Close CSV
        if self._csv_writer:
            self._csv_writer.close()

        # 3. Extract features
        features = None
        if self._valid_samples >= 10:
            logger.info("Extracting gait features from %d samples …", self._valid_samples)
            features = extract_gait_features(all_samples)
        else:
            logger.warning(
                "Only %d valid samples — skipping feature extraction (minimum 10 required).",
                self._valid_samples,
            )

        # 4. Data quality stats
        total_possible = max(self._total_lines, 1)
        data_quality = {
            "total_lines_received": self._total_lines,
            "valid_samples":        self._valid_samples,
            "rejected_lines":       self._rejected,
            "parse_success_rate":   round(self._valid_samples / total_possible * 100, 1),
        }

        # 5. Complete Firestore session
        if self._fs_session:
            self._fs_session.complete(
                total_samples=self._total_lines,
                valid_samples=self._valid_samples,
                features=features,
                data_quality=data_quality,
            )
            self._fs_session.set_rtdb_status("completed")

        # 6. Summary
        logger.info(
            "Session complete: %s\n"
            "  Valid samples : %d\n"
            "  Rejected      : %d\n"
            "  Parse success : %.1f%%\n"
            "  CSV saved to  : data/raw/%s.csv",
            self._session_id,
            self._valid_samples,
            self._rejected,
            data_quality["parse_success_rate"],
            self._session_id,
        )


# ===========================================================================
# FastAPI control server  (optional — allows dashboard to start/stop recording)
# ===========================================================================

_active_bridge: Optional[ESP32FirebaseBridge] = None
_bridge_thread: Optional[threading.Thread] = None


def start_fastapi_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """
    Start a lightweight HTTP control server.

    Endpoints:
        GET  /status    — returns bridge status as JSON
        POST /start     — start recording (body: {patient_id, serial_port})
        POST /stop      — stop recording
        GET  /patients  — list patients from Firestore

    Why do we need this?
    --------------------
    A browser button cannot directly start or stop a Python process.
    The dashboard sends HTTP requests to this local server instead.
    This server runs on the same laptop as the Python bridge.
    """
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        from pydantic import BaseModel
    except ImportError:
        logger.error(
            "fastapi and uvicorn are required for the control server.\n"
            "Install them with:  pip install fastapi uvicorn"
        )
        return

    app = FastAPI(title="OA Bridge Control Server", version="1.0")

    # Allow requests from the Streamlit frontend (localhost)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://localhost:8502", "http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class StartRequest(BaseModel):
        patient_id:   str = "patient_001"
        serial_port:  str = ""
        baud_rate:    int = 115200
        demo_mode:    bool = False

    @app.get("/status")
    def get_status():
        global _active_bridge, _bridge_thread
        is_running = (
            _bridge_thread is not None
            and _bridge_thread.is_alive()
        )
        return {
            "status":       "recording" if is_running else "idle",
            "session_id":   _active_bridge._session_id if _active_bridge else None,
            "patient_id":   _active_bridge.patient_id  if _active_bridge else None,
            "valid_samples": _active_bridge._valid_samples if _active_bridge else 0,
        }

    @app.post("/start")
    def start_recording(req: StartRequest):
        global _active_bridge, _bridge_thread

        if _bridge_thread and _bridge_thread.is_alive():
            return {"success": False, "message": "Already recording"}

        _active_bridge = ESP32FirebaseBridge(
            patient_id=req.patient_id,
            port=req.serial_port,
            baud=req.baud_rate,
            demo_mode=req.demo_mode,
        )
        _bridge_thread = threading.Thread(
            target=_active_bridge.run, name="BridgeMain", daemon=True
        )
        _bridge_thread.start()
        logger.info("Recording started via API for patient: %s", req.patient_id)
        return {"success": True, "message": "Recording started", "patient_id": req.patient_id}

    @app.post("/stop")
    def stop_recording():
        global _active_bridge
        if _active_bridge:
            _active_bridge.stop()
            return {"success": True, "message": "Stop signal sent"}
        return {"success": False, "message": "No active recording"}

    @app.get("/patients")
    def list_patients():
        """Return patient list from Firestore."""
        if _fs_client is None:
            return {"patients": [], "source": "firebase_unavailable"}
        try:
            docs = _fs_client.collection("patients").stream()
            patients = [d.to_dict().get("patient_id", d.id) for d in docs]
            return {"patients": patients, "source": "firestore"}
        except Exception as exc:
            return {"patients": [], "error": str(exc)}

    @app.get("/latest/{patient_id}/{session_id}")
    def get_latest(patient_id: str, session_id: str):
        """
        Read the most recent sensor reading from Firebase Realtime Database.
        Falls back to the active bridge's in-memory sample in demo mode or
        before the first 200ms RTDB write completes.
        """
        global _active_bridge
        if _rtdb is not None and not (_active_bridge and _active_bridge.demo_mode):
            try:
                ref = _rtdb.reference(f"live_sessions/{patient_id}/{session_id}/latest")
                data = ref.get()
                if data is not None:
                    return data
            except Exception as exc:
                logger.debug("RTDB get_latest error: %s", exc)

        if (
            _active_bridge is not None
            and _active_bridge._session_id == session_id
            and _active_bridge._latest_sample is not None
        ):
            return _active_bridge._latest_sample

        if _rtdb is None:
            return {"error": "rtdb_unavailable"}
        return {"error": "no_data_yet"}

    @app.get("/rtdb_chunks/{patient_id}/{session_id}")
    def get_rtdb_chunks(patient_id: str, session_id: str):
        """
        Retrieve all recorded 50 Hz sample chunks & server timestamps
        directly from Firebase Realtime Database for a given session.
        Path: live_sessions/{patientId}/{sessionId}/chunks
        """
        if _rtdb is None:
            return {"samples": [], "error": "rtdb_unavailable"}
        try:
            ref = _rtdb.reference(f"live_sessions/{patient_id}/{session_id}/chunks")
            chunks_raw = ref.get()
            if not chunks_raw:
                return {"samples": [], "chunk_count": 0}

            flat_samples = []
            for chunk_key in sorted(chunks_raw.keys()):
                chunk = chunks_raw[chunk_key]
                srv_ts = chunk.get("server_received")
                samples_dict = chunk.get("samples", {})
                for s_key in sorted(samples_dict.keys()):
                    row = dict(samples_dict[s_key])
                    row["chunk_id"] = chunk_key
                    row["server_received"] = srv_ts
                    flat_samples.append(row)

            return {
                "samples": flat_samples,
                "chunk_count": len(chunks_raw),
                "sample_count": len(flat_samples),
                "source": "firebase_rtdb",
            }
        except Exception as exc:
            return {"samples": [], "error": str(exc)}

    @app.get("/sessions/{patient_id}")
    def get_sessions(patient_id: str):
        """
        Read historical gait sessions from Firestore for a patient.
        Path: patients/{patientId}/gait_sessions/{sessionId}
        """
        if _fs_client is None:
            return {"sessions": [], "source": "firebase_unavailable"}
        try:
            docs = (
                _fs_client
                .collection("patients")
                .document(patient_id)
                .collection("gait_sessions")
                .order_by("created_at", direction="DESCENDING")
                .limit(20)
                .stream()
            )
            sessions = []
            for d in docs:
                data = d.to_dict()
                # Convert Firestore timestamps to ISO strings for JSON serialisation
                for key in ("start_time", "end_time", "created_at", "completed_at"):
                    if key in data and hasattr(data[key], "isoformat"):
                        data[key] = data[key].isoformat()
                sessions.append(data)
            return {"sessions": sessions, "source": "firestore"}
        except Exception as exc:
            return {"sessions": [], "error": str(exc)}

    @app.get("/live_status")
    def live_status():
        """
        Returns both bridge recording state AND current sample count.
        Used by the dashboard for polling during active recording.
        """
        global _active_bridge, _bridge_thread
        is_running = _bridge_thread is not None and _bridge_thread.is_alive()
        return {
            "bridge_running":   is_running,
            "status":           "recording" if is_running else "idle",
            "session_id":       _active_bridge._session_id  if _active_bridge else None,
            "patient_id":       _active_bridge.patient_id   if _active_bridge else None,
            "valid_samples":    _active_bridge._valid_samples if _active_bridge else 0,
            "rejected_samples": _active_bridge._rejected     if _active_bridge else 0,
            "demo_mode":        _active_bridge.demo_mode     if _active_bridge else False,
        }

    logger.info("Control server starting at http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")


# ===========================================================================
# Command-line entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ESP32 → Firebase bridge for OA patient gait monitoring"
    )
    p.add_argument(
        "--port", default=SERIAL_PORT,
        help="Serial port (e.g. /dev/cu.usbserial-0001 or COM3). Default: auto-detect.",
    )
    p.add_argument(
        "--baud", type=int, default=BAUD_RATE,
        help="Baud rate (default: 115200).",
    )
    p.add_argument(
        "--patient", default=PATIENT_ID,
        help="Patient ID (default: patient_001).",
    )
    p.add_argument(
        "--demo", action="store_true", default=DEMO_MODE,
        help="Run in demo mode (no hardware, no Firebase writes).",
    )
    p.add_argument(
        "--server", action="store_true",
        help="Run the FastAPI control server instead of the bridge directly.",
    )
    p.add_argument(
        "--server-port", type=int, default=8765,
        help="Port for the FastAPI control server (default: 8765).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Validate patient ID
    if not args.patient or args.patient.strip() == "":
        logger.error("Patient ID is required. Use --patient patient_001 or set PATIENT_ID in .env")
        sys.exit(1)

    # Graceful Ctrl+C
    def _sigint(sig, frame):
        logger.info("Ctrl+C received — stopping bridge.")
        if 'bridge' in dir():
            bridge.stop()

    signal.signal(signal.SIGINT, _sigint)

    if args.server:
        # Start FastAPI control server + init Firebase in background
        if not args.demo:
            _init_firebase()
        start_fastapi_server(port=args.server_port)
    else:
        # Run bridge directly
        bridge = ESP32FirebaseBridge(
            patient_id=args.patient,
            port=args.port,
            baud=args.baud,
            demo_mode=args.demo,
        )
        bridge.run()


if __name__ == "__main__":
    main()

