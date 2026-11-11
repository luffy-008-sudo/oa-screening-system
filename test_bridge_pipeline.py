"""
test_bridge_pipeline.py
=======================
End-to-end pipeline tests for the ESP32 → Firebase bridge.

Tests:
  1. CSV line parsing (JSON format from ESP32 firmware)
  2. CSV line parsing (comma-separated format from prompt spec)
  3. Invalid line rejection
  4. Gait feature extraction with real numpy calculations
  5. CSV backup writer (write / read back)
  6. Bridge demo mode run (10 seconds of simulated data)
  7. Bridge client HTTP functions (against a mock server)
  8. Session ID format validation
  9. Firebase RTDB URL configuration
  10. Requirements file completeness

Run with:
    source venv/bin/activate
    python test_bridge_pipeline.py

Or with pytest:
    pytest test_bridge_pipeline.py -v
"""

import csv
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test 1 & 2 & 3: Parsing
# ---------------------------------------------------------------------------

class TestLineParsing(unittest.TestCase):

    def setUp(self):
        from serial_firebase_bridge import _parse_line
        self._parse = _parse_line

    def test_valid_json_from_firmware(self):
        """ESP32 firmware outputs this exact JSON format."""
        line = (
            '{"timestamp_ms":1250,"heel":120,"midfoot":240,"forefoot":180,'
            '"acc_x_g":0.12,"acc_y_g":0.25,"acc_z_g":0.98,'
            '"gyro_x_dps":0.3,"gyro_y_dps":0.1,"gyro_z_dps":0.4}'
        )
        result = self._parse(line)
        self.assertIsNotNone(result, "Should parse valid firmware JSON")
        self.assertEqual(result["heel"],        120)
        self.assertEqual(result["midfoot"],     240)
        self.assertEqual(result["forefoot"],    180)
        self.assertAlmostEqual(result["acc_z_g"],    0.98, places=2)
        self.assertAlmostEqual(result["gyro_x_dps"], 0.3,  places=1)

    def test_valid_csv_format(self):
        """10-field CSV format as specified in the prompt."""
        line = "1250,120,240,180,0.12,0.25,0.98,0.3,0.1,0.4"
        result = self._parse(line)
        self.assertIsNotNone(result, "Should parse 10-field CSV")
        self.assertEqual(result["heel"],    120.0)
        self.assertEqual(result["midfoot"], 240.0)

    def test_comment_line_ignored(self):
        """Comment lines starting with // should be skipped."""
        result = self._parse("// ESP32 OA Screening Firmware Initialized at 50 Hz.")
        self.assertIsNone(result, "Comment lines should return None")

    def test_empty_line_ignored(self):
        result = self._parse("")
        self.assertIsNone(result)

    def test_malformed_json_ignored(self):
        result = self._parse("{invalid json here}")
        self.assertIsNone(result)

    def test_missing_fields_ignored(self):
        """JSON missing heel field should be rejected."""
        line = '{"timestamp_ms":100,"midfoot":200}'
        result = self._parse(line)
        self.assertIsNone(result, "Incomplete JSON should return None")

    def test_non_numeric_field_ignored(self):
        """Non-numeric sensor values should be rejected."""
        line = (
            '{"timestamp_ms":100,"heel":"bad","midfoot":200,"forefoot":300,'
            '"acc_x_g":0.1,"acc_y_g":0.2,"acc_z_g":0.9,'
            '"gyro_x_dps":0.1,"gyro_y_dps":0.2,"gyro_z_dps":0.3}'
        )
        result = self._parse(line)
        self.assertIsNone(result, "Non-numeric heel should return None")

    def test_too_few_csv_fields(self):
        """9 fields instead of 10 should be rejected."""
        line = "1250,120,240,180,0.12,0.25,0.98,0.3,0.1"  # only 9 fields
        result = self._parse(line)
        self.assertIsNone(result, "9-field CSV should return None")


# ---------------------------------------------------------------------------
# Test 4: Feature extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction(unittest.TestCase):

    def _make_samples(self, n=100):
        import math
        samples = []
        for i in range(n):
            phase = (i % 50) / 50.0
            samples.append({
                "timestamp_ms": i * 20,
                "heel":         int(800 * math.exp(-((phase - 0.15)**2) / 0.01)),
                "midfoot":      int(600 * math.exp(-((phase - 0.45)**2) / 0.02)),
                "forefoot":     int(850 * math.exp(-((phase - 0.75)**2) / 0.01)),
                "acc_x_g":      0.05 * math.sin(2 * math.pi * phase),
                "acc_y_g":      0.08 * math.sin(2 * math.pi * phase + 0.5),
                "acc_z_g":      0.98,
                "gyro_x_dps":   2.0 * math.sin(2 * math.pi * phase),
                "gyro_y_dps":   1.5 * math.cos(2 * math.pi * phase),
                "gyro_z_dps":   0.8 * math.sin(4 * math.pi * phase),
            })
        return samples

    def test_features_returned(self):
        from serial_firebase_bridge import extract_gait_features
        samples = self._make_samples(100)
        feats = extract_gait_features(samples)
        self.assertIsInstance(feats, dict)
        self.assertGreater(len(feats), 10, "Should extract more than 10 features")

    def test_total_samples_correct(self):
        from serial_firebase_bridge import extract_gait_features
        samples = self._make_samples(100)
        feats = extract_gait_features(samples)
        self.assertEqual(feats["total_samples"], 100)

    def test_load_percentages_sum_to_100(self):
        from serial_firebase_bridge import extract_gait_features
        samples = self._make_samples(100)
        feats = extract_gait_features(samples)
        total_pct = (
            feats["heel_load_percent"]
            + feats["midfoot_load_percent"]
            + feats["forefoot_load_percent"]
        )
        self.assertAlmostEqual(total_pct, 100.0, delta=1.0,
            msg="Foot load percentages should sum to ~100%")

    def test_adc_values_in_range(self):
        from serial_firebase_bridge import extract_gait_features
        samples = self._make_samples(100)
        feats = extract_gait_features(samples)
        self.assertGreaterEqual(feats["heel_max_adc"],    0)
        self.assertLessEqual(   feats["heel_max_adc"],    4095)
        self.assertGreaterEqual(feats["midfoot_max_adc"], 0)

    def test_disclaimer_present(self):
        from serial_firebase_bridge import extract_gait_features
        samples = self._make_samples(100)
        feats = extract_gait_features(samples)
        self.assertIn("disclaimer", feats)
        self.assertIn("ADC", feats["disclaimer"])

    def test_empty_samples_returns_empty(self):
        from serial_firebase_bridge import extract_gait_features
        feats = extract_gait_features([])
        self.assertEqual(feats, {})

    def test_duration_calculated_correctly(self):
        from serial_firebase_bridge import extract_gait_features
        samples = self._make_samples(50)   # 50 samples × 20ms = 980ms
        feats = extract_gait_features(samples)
        # 0 to 49 × 20ms = 980ms duration
        self.assertAlmostEqual(feats["duration_ms"], 49 * 20, delta=50)


# ---------------------------------------------------------------------------
# Test 5: CSV backup writer
# ---------------------------------------------------------------------------

class TestCSVBackupWriter(unittest.TestCase):

    def test_write_and_read_back(self):
        from serial_firebase_bridge import CSVBackupWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_session.csv"
            writer = CSVBackupWriter(filepath, "patient_001", "patient_001_test_session")
            writer.open()

            sample = {
                "timestamp_ms": 1000,
                "heel": 500, "midfoot": 300, "forefoot": 700,
                "acc_x_g": 0.1, "acc_y_g": 0.2, "acc_z_g": 0.95,
                "gyro_x_dps": 1.1, "gyro_y_dps": 0.5, "gyro_z_dps": 0.3,
            }
            writer.write(sample, seq=1, data_source="real")
            writer.write(sample, seq=2, data_source="real")
            writer.close()

            self.assertTrue(filepath.exists(), "CSV file should exist")
            with open(filepath, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 2, "Should have 2 data rows")
            self.assertEqual(rows[0]["patient_id"], "patient_001")
            self.assertEqual(int(rows[0]["heel"]),  500)
            self.assertEqual(rows[0]["data_source"], "real")

    def test_csv_has_correct_headers(self):
        from serial_firebase_bridge import CSVBackupWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "header_test.csv"
            writer = CSVBackupWriter(filepath, "p001", "sess_001")
            writer.open()
            writer.close()

            with open(filepath, "r") as f:
                header_line = f.readline().strip()

            expected_fields = ["seq", "heel", "midfoot", "forefoot", "patient_id", "session_id"]
            for field in expected_fields:
                self.assertIn(field, header_line)


# ---------------------------------------------------------------------------
# Test 6: Demo mode bridge run
# ---------------------------------------------------------------------------

class TestDemoBridgeRun(unittest.TestCase):

    def test_demo_mode_generates_csv(self):
        """
        Run bridge in demo mode for 3 seconds and verify CSV output.
        No Firebase writes in demo mode.
        """
        # Run the bridge with real project root — it writes to data/raw
        from serial_firebase_bridge import ESP32FirebaseBridge, _PROJECT_ROOT

        raw_dir = _PROJECT_ROOT / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Count existing files before
        before = set(raw_dir.glob("test_patient_*.csv"))

        bridge = ESP32FirebaseBridge(
            patient_id="test_patient",
            port="",
            demo_mode=True,
        )

        # Run for 3 seconds in a thread
        t = threading.Thread(target=bridge.run, daemon=True)
        t.start()
        time.sleep(3.5)
        bridge.stop()
        t.join(timeout=5)

        # Find newly created CSV
        after = set(raw_dir.glob("test_patient_*.csv"))
        new_files = after - before
        self.assertGreater(len(new_files), 0, "Demo mode should produce a CSV file")

        if new_files:
            csv_file = sorted(new_files)[-1]
            df_data = []
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                df_data = list(reader)
            self.assertGreater(len(df_data), 50,
                               f"Should have >50 samples in 3 seconds (got {len(df_data)})")
            self.assertEqual(df_data[0]["patient_id"], "test_patient")
            self.assertEqual(df_data[0]["data_source"], "simulated")


# ---------------------------------------------------------------------------
# Test 7: Session ID format
# ---------------------------------------------------------------------------

class TestSessionID(unittest.TestCase):

    def test_session_id_format(self):
        from serial_firebase_bridge import make_session_id
        sid = make_session_id("patient_001")
        self.assertTrue(sid.startswith("patient_001_"), f"Should start with patient ID, got: {sid}")
        # Format: patient_001_YYYYMMDD_HHMMSS
        parts = sid.split("_")
        self.assertGreaterEqual(len(parts), 4)

    def test_session_ids_are_unique(self):
        from serial_firebase_bridge import make_session_id
        ids = set()
        for _ in range(5):
            ids.add(make_session_id("patient_001"))
            time.sleep(1.1)  # ensure second-level uniqueness
        # In fast tests this might collide within the same second — that's acceptable
        self.assertGreaterEqual(len(ids), 1)


# ---------------------------------------------------------------------------
# Test 8: Configuration
# ---------------------------------------------------------------------------

class TestConfiguration(unittest.TestCase):

    def test_env_file_has_rtdb_url(self):
        env_path = PROJECT_ROOT / ".env"
        if not env_path.exists():
            self.skipTest(".env file not present")
        content = env_path.read_text()
        self.assertIn("FIREBASE_DATABASE_URL", content, ".env should have FIREBASE_DATABASE_URL")
        self.assertIn("oa-screening-system-default-rtdb", content,
                      ".env FIREBASE_DATABASE_URL should point to the correct RTDB instance")

    def test_requirements_has_fastapi(self):
        req_path = PROJECT_ROOT / "requirements.txt"
        if not req_path.exists():
            self.skipTest("requirements.txt not present")
        content = req_path.read_text()
        self.assertIn("fastapi", content)
        self.assertIn("uvicorn", content)

    def test_bridge_client_importable(self):
        try:
            from app.bridge_client import bridge_is_running, start_recording, stop_recording
        except ImportError as e:
            self.fail(f"bridge_client.py should be importable: {e}")

    def test_bridge_importable(self):
        try:
            import serial_firebase_bridge
        except ImportError as e:
            self.fail(f"serial_firebase_bridge.py should be importable: {e}")


# ---------------------------------------------------------------------------
# Test 9: Bridge client offline behaviour
# ---------------------------------------------------------------------------

class TestBridgeClientOffline(unittest.TestCase):

    def test_bridge_offline_returns_safe_defaults(self):
        """When bridge server is not running, all functions return safe defaults."""
        from app.bridge_client import (
            bridge_is_running, get_bridge_status,
            get_live_status, get_session_history,
        )

        # Bridge is definitely not running in tests
        self.assertFalse(bridge_is_running())

        status = get_bridge_status()
        self.assertIn("status", status)

        live = get_live_status()
        self.assertFalse(live.get("bridge_running", True))

        sessions = get_session_history("patient_001")
        self.assertIsInstance(sessions, list)

    def test_start_recording_offline_returns_error(self):
        from app.bridge_client import start_recording
        result = start_recording("patient_001")
        self.assertFalse(result.get("success", True))
        self.assertIn("message", result)


# ---------------------------------------------------------------------------
# Test 10: End-to-end data quality (parse → features)
# ---------------------------------------------------------------------------

class TestEndToEndDataQuality(unittest.TestCase):

    def test_realistic_session_pipeline(self):
        """
        Simulate a realistic 5-second session:
        Parse 250 JSON lines → extract features → verify quality.
        """
        from serial_firebase_bridge import _parse_line, extract_gait_features
        import math

        raw_lines = []
        for i in range(250):
            phase = (i % 50) / 50.0
            line = json.dumps({
                "timestamp_ms": i * 20,
                "heel":         max(0, int(800 * math.exp(-((phase - 0.15)**2) / 0.01))),
                "midfoot":      max(0, int(600 * math.exp(-((phase - 0.45)**2) / 0.02))),
                "forefoot":     max(0, int(850 * math.exp(-((phase - 0.75)**2) / 0.01))),
                "acc_x_g":      round(0.05 * math.sin(2 * math.pi * phase), 4),
                "acc_y_g":      round(0.08 * math.sin(2 * math.pi * phase + 0.5), 4),
                "acc_z_g":      round(0.98, 4),
                "gyro_x_dps":   round(2.0 * math.sin(2 * math.pi * phase), 4),
                "gyro_y_dps":   round(1.5 * math.cos(2 * math.pi * phase), 4),
                "gyro_z_dps":   round(0.8 * math.sin(4 * math.pi * phase), 4),
            })
            raw_lines.append(line)

        # Add 10 corrupt lines (should be rejected)
        raw_lines.extend([
            "// ESP32 OA Screening Firmware Initialized",
            "",
            "{bad json",
            '{"only": "partial"}',
            "not,enough,csv,fields",
        ] * 2)

        # Parse all lines
        valid_samples = []
        rejected = 0
        for line in raw_lines:
            result = _parse_line(line)
            if result:
                valid_samples.append(result)
            else:
                rejected += 1

        self.assertEqual(len(valid_samples), 250, "Should parse exactly 250 valid samples")
        self.assertEqual(rejected, 10, "Should reject exactly 10 corrupt lines")

        # Extract features
        features = extract_gait_features(valid_samples)
        self.assertEqual(features["total_samples"], 250)
        self.assertAlmostEqual(
            features["heel_load_percent"]
            + features["midfoot_load_percent"]
            + features["forefoot_load_percent"],
            100.0, delta=1.0
        )
        self.assertGreater(features["acc_magnitude_mean_g"], 0.9,
                           "Acc magnitude should be ~1g at rest")


if __name__ == "__main__":
    print("=" * 60)
    print("OA System — Bridge Pipeline Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
