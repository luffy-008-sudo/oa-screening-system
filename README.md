# KINETIX OA — AI-Powered Osteoarthritis Patient Assistance & Monitoring System
### Wearable Insole Foot Pressure & Gait Kinematics with Gemini AI Clinical Assistance

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit%20Dark-FF6B00.svg)](https://streamlit.io/)
[![Hardware](https://img.shields.io/badge/Hardware-ESP32%20DevKit%20%2B%20MPU6050-orange.svg)](https://www.espressif.com/)
[![Firebase](https://img.shields.io/badge/Database-Cloud%20Firestore-yellow.svg)](https://firebase.google.com/)
[![AI Engine](https://img.shields.io/badge/AI-Gemini%202.5%20Flash-4285F4.svg)](https://ai.google.dev/)
[![Tests](https://img.shields.io/badge/Tests-156%20%2B%2027%20Passing-brightgreen.svg)](run_tests.py)

---

> **⚠️ Mandatory Research Prototype Notice**  
> This system is an academic research and patient assistance prototype designed for continuous lifestyle, nutrition, symptom, and gait monitoring of individuals **already diagnosed** with knee osteoarthritis.  
> It is **not** a diagnostic device and does not claim to diagnose osteoarthritis or replace treating physicians, orthopedists, or physical therapists.

---

## 📖 Table of Contents

1. [System Architecture](#system-architecture)
2. [Hardware & Sensor Insole Specifications](#hardware--sensor-insole-specifications)
3. [Key Software Modules](#key-software-modules)
4. [Visual Identity — Orange & Black UI](#visual-identity--orange--black-ui)
5. [Installation & Quickstart](#installation--quickstart)
6. [Hardware Wiring & Firmware Setup](#hardware-wiring--firmware-setup)
7. [Running the System](#running-the-system)
8. [Automated Verification & Tests](#automated-verification--tests)
9. [Project Directory Structure](#project-directory-structure)
10. [Security & Privacy Guardrails](#security--privacy-guardrails)

---

## 🏗️ System Architecture

```
 Wearable Insole Sensors (Foot Pressure & Kinematics)
  ├── Heel FSR (GPIO 34)
  ├── Midfoot FSR (GPIO 35)
  ├── Forefoot FSR (GPIO 32)
  └── MPU6050 6-DOF IMU (I2C: GPIO 21 / 22)
                 │
                 ▼ (USB Serial @ 115,200 Baud, ~50 Hz JSON)
         ESP32 DevKit V1
                 │
                 ▼
  Standalone Bridge & FastAPI Control Server (`serial_firebase_bridge.py`)
  ├── Validates 50 Hz JSON/CSV packets & writes local CSV backups (`data/raw/`)
  ├── Pushes live telemetry every ~200 ms & 50-sample chunks to Realtime Database
  ├── Persists session metadata & extracted gait features to Cloud Firestore
  └── Serves local control & telemetry API at `http://127.0.0.1:8765`
                 │
        ┌────────┴────────────────────────┐
        ▼                                 ▼
Firebase Cloud                    Python Analytics & Streamlit Dashboard
├── Realtime Database             ├── `bridge_client.py` (HTTP client to bridge)
│   (`live_sessions/{pid}/{sid}`) ├── `DataProcessor` & `FeatureExtractor`
└── Cloud Firestore               ├── `BaselineEngine` & `AnomalyDetector`
    (`patients/{pid}/gait_sessions`) ├── `AIReportGenerator` (Gemini 2.5 Flash)
                                  └── 9-Module Orange & Black Dashboard UI
```

---

## ⚡ Hardware & Sensor Insole Specifications

| Component | Quantity | Role / Specification |
|---|---|---|
| **ESP32 DevKit V1** | 1 | Microcontroller running non-blocking 50 Hz acquisition loop |
| **FSR 402 Sensors** | 3 | Force Sensing Resistors (Heel, Midfoot, Forefoot) |
| **MPU-6050 IMU** | 1 | 3-Axis Accelerometer ($\pm 2g$) + 3-Axis Gyroscope ($\pm 250^\circ/\text{s}$) |
| **Voltage Dividers** | 3 | $10\text{ k}\Omega$ pull-down resistors for linearised ADC response |
| **Micro-USB Cable** | 1 | Direct laptop connection (powers the board and streams data) |

---

## 🎨 Visual Identity — Orange & Black UI

The user interface follows a modern fitness-tech and health tracking aesthetic:
- **Canvas Base:** `#101010` (Deep black)
- **Surfaces & Cards:** `#1A1A1A` and `#242424` (`border: 1px solid #282828`)
- **Primary Highlights:** `#FF6B00` (Vibrant orange)
- **Secondary Accents:** `#FF8A24` (Warm orange)
- **Typography:** `#FFFFFF` (High contrast) and `#9E9E9E` (Muted labels)

### Dashboard Navigation:
1. **Dashboard Overview:** 4 Orange KPI cards (Steps, Protein, Pain scale, Physical Therapy), 14-day activity trends, baseline profile status.
2. **Patient Profile & Wizard:** Multi-step clinical onboarding wizard (Demographics, KL Severity, Goals) + active profile management.
3. **Medical Reports Vault:** Upload, store, and categorize radiographs (X-Rays), MRI scans, and lab panels.
4. **Nutrition Tracking:** Protein goal progress bar, quick-add preset buttons, daily food diary with auto-summation.
5. **Physical Activity:** Step counter, walking duration tracker, and 30-day baseline adherence chart.
6. **Gait Monitoring:** Real-time ESP32 sensor telemetry with live FSR pressure gauges, kinematics meters, and playback of recorded trials.
7. **Symptom Journal:** Self-reported pain slider (0-10), morning stiffness duration, fatigue score, and dual-axis symptom progression charts.
8. **AI Insights:** Gemini 2.5 Flash clinical summaries, baseline deviation signals table, and suggested physician discussion questions.
9. **System Settings:** Serial port auto-detection, cloud synchronization health, and Gemini API diagnostics.

---

## 🚀 Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/luffy-008-sudo/oa-screening-system.git
cd oa-screening-system
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the example configuration file:
```bash
cp .env.example .env
```

Open `.env` and fill in your details:
```ini
# Google Gemini AI
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL_NAME=gemini-2.5-flash

# Firebase Firestore & Realtime Database (optional — local mode works without cloud)
FIREBASE_CREDENTIALS_PATH=secrets/firebase-key.json
FIREBASE_PROJECT_ID=your-firebase-project-id
FIREBASE_DATABASE_URL=https://your-firebase-project-id-default-rtdb.asia-southeast1.firebasedatabase.app

# Hardware Serial Port (leave blank to auto-detect or use the UI)
SERIAL_PORT=
SERIAL_BAUD_RATE=115200
BAUD_RATE=115200
PATIENT_ID=patient_001
DEMO_MODE=false
```

---

## 🔌 Hardware Wiring & Firmware Setup

### Pin Mapping:
| Sensor | ESP32 Pin | Details |
|---|---|---|
| **FSR Heel** | GPIO 34 | Analog Input (ADC1) |
| **FSR Midfoot** | GPIO 35 | Analog Input (ADC1) |
| **FSR Forefoot** | GPIO 32 | Analog Input (ADC1) |
| **MPU6050 SDA** | GPIO 21 | I2C Data |
| **MPU6050 SCL** | GPIO 22 | I2C Clock |
| **VCC & GND** | 3.3V & GND | Connected to all sensors |

### Flash the Firmware:
1. Open Arduino IDE.
2. Open `firmware/esp32_gait_acquisition/esp32_gait_acquisition.ino`.
3. Select board: **ESP32 Dev Module**.
4. Select your USB port.
5. Click **Upload** (115,200 baud).

---

## 🖥️ Running the Application

Run the ESP32-to-Firebase bridge server and the Streamlit dashboard in two separate terminals:

**Terminal 1 — ESP32 $\rightarrow$ Firebase Bridge Server:**
```bash
source venv/bin/activate
python serial_firebase_bridge.py --server
```
*(Or run standalone CLI capture without the UI: `python serial_firebase_bridge.py --patient patient_001`)*

**Terminal 2 — Streamlit Web Dashboard:**
```bash
source venv/bin/activate
streamlit run app/dashboard.py
```
Open your browser at **`http://localhost:8501`**.

> **Note on Demo Mode:** If no physical ESP32 is plugged in, enable **Demo Mode (no hardware)** on the Gait Monitoring page or pass `--demo` to `serial_firebase_bridge.py` to test safely without writing simulated data to production Firebase.

---

## 🧪 Automated Verification & Tests

Run the core test suite and the ESP32-Firebase bridge pipeline tests:
```bash
python run_tests.py
python test_bridge_pipeline.py
```
**Expected Result:**
- `run_tests.py`: 156 tests passing
- `test_bridge_pipeline.py`: 27 tests passing

---

## 📁 Project Directory Structure

```
OA_Screening_System/
├── .env.example                    # Environment template
├── .gitignore                      # Security-audited git exclusions
├── .python-version                 # Python 3.11 pin for cloud deployment
├── firebase_database_rules.json    # Realtime Database security rules
├── requirements.txt                # Python package manifest
├── serial_firebase_bridge.py       # Standalone ESP32 -> Firebase RTDB & Firestore bridge
├── test_bridge_pipeline.py         # End-to-end bridge pipeline tests
├── run_tests.py                    # Master test runner
├── README.md                       # Documentation
├── .streamlit/
│   └── config.toml                 # Streamlit dark theme configuration
├── app/
│   ├── config.py                   # Central configuration (st.secrets + .env)
│   ├── ui_theme.py                 # Orange & Black CSS & component system
│   ├── patient_profile_service.py  # Profile & medical records persistence
│   ├── serial_receiver.py          # ESP32 USB serial acquisition
│   ├── bridge_client.py            # HTTP client for serial_firebase_bridge.py
│   ├── firebase_rtdb_listener.js   # Browser Realtime Database live listener
│   ├── data_logger.py              # CSV recording
│   ├── data_processor.py           # Signal cleaning & calibration
│   ├── feature_extractor.py        # Biomechanical feature computation
│   ├── baseline_engine.py          # Personalized median & MAD baselines
│   ├── anomaly_detector.py         # Robust Z-score anomaly detector
│   ├── daily_log_service.py        # Food diary, steps, and symptom logs
│   ├── trend_analyzer.py           # Weekly trends & Spearman correlations
│   ├── ai_report_generator.py      # Gemini 2.5 Flash report generator
│   ├── patient_monitoring.py       # Central monitoring coordinator facade
│   ├── firebase_service.py         # Firestore Admin SDK service
│   ├── session_manager.py          # Walking trial lifecycle manager
│   └── dashboard.py                # Redesigned 9-module Streamlit UI
├── firmware/
│   ├── WIRING_GUIDE.md             # Detailed schematic & breadboard guide
│   └── esp32_gait_acquisition/
│       └── esp32_gait_acquisition.ino
├── ml/
│   ├── train_model.py              # Random Forest training pipeline
│   ├── predict.py                  # Model inference with metadata
│   ├── model_utils.py              # Medical disclaimer & feature schemas
│   └── personalized_anomaly_model.py # Personalized Isolation Forest
├── data/
│   ├── raw/                        # Raw trial CSVs (ignored by git)
│   ├── processed/                  # Cleaned CSVs and extracted features
│   ├── demo/                       # Synthetic demo recordings
│   ├── patient_profiles/           # Clinical profiles
│   └── medical_reports/            # Diagnostic scans and reports
└── tests/                          # 7 comprehensive test suites
```

---

## 🔒 Security & Privacy Guardrails

- **Zero Credential Leaks:** Private service account keys (`secrets/firebase-key.json`) and API keys (`.env`) are strictly excluded in `.gitignore`.
- **Pseudonymous Identifiers:** All trial recordings and clinical documents use research identifiers (`patient_001`, `patient_002`) rather than personally identifiable information.
- **Fail-Safe Fallbacks:** When offline or when the Gemini API is unconfigured, the application automatically uses a deterministic rule-based medical communication engine.
- **Strict Clinical Scope:** System documentation, UI banners, and model score cards prominently feature the mandatory research disclaimer on all screens.
