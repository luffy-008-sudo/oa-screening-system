/**
 * firebase_rtdb_listener.js
 * =========================
 * Firebase Realtime Database listener for the OA Gait Monitoring page.
 *
 * What this does (simple explanation)
 * -------------------------------------
 * When the Python bridge is recording and sends data to Firebase,
 * this JavaScript code listens for those updates and shows them
 * live on the website — without refreshing the page.
 *
 * How the data flows:
 *   ESP32 → USB → Python bridge → Firebase RTDB → This listener → Your website
 *
 * Data path in Firebase Realtime Database:
 *   live_sessions/{patientId}/{sessionId}/latest   ← updated every 200ms
 *   live_sessions/{patientId}/{sessionId}/status   ← "recording", "completed"
 *   live_sessions/{patientId}/{sessionId}/chunks/  ← batches of ~50 samples
 *
 * IMPORTANT SECURITY NOTE:
 * -------------------------
 * This file uses the FRONTEND (client) Firebase SDK — NOT the Admin SDK.
 * The Admin SDK with the service account private key MUST NEVER be in browser code.
 * The client SDK uses public configuration only (no secrets).
 *
 * Usage — include in your HTML:
 * --------------------------------
 *   <script type="module" src="firebase_rtdb_listener.js"></script>
 *   <div id="gait-monitor-container"></div>
 *
 * Or import the GaitMonitor class in your framework component.
 */

// ---------------------------------------------------------------------------
// Firebase Client SDK Configuration
// Replace these values with your actual Firebase project config.
// Find them at: Firebase Console → Project Settings → Your Apps → Web App
// These are PUBLIC values — safe to include in browser code.
// ---------------------------------------------------------------------------
const FIREBASE_CONFIG = {
  apiKey:            "AIzaSy_YOUR_ACTUAL_KEY_HERE",
  authDomain:        "oa-screening-system.firebaseapp.com",
  databaseURL:       "https://oa-screening-system-default-rtdb.firebaseio.com",
  projectId:         "oa-screening-system",
  storageBucket:     "oa-screening-system.firebasestorage.app",
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
  appId:             "YOUR_APP_ID",
};

// ---------------------------------------------------------------------------
// GaitMonitor — the main class you use in your website
// ---------------------------------------------------------------------------

export class GaitMonitor {
  /**
   * @param {string} patientId   - e.g. "patient_001"
   * @param {string} sessionId   - e.g. "patient_001_20261111_143022"
   * @param {object} callbacks   - { onLatest, onStatus, onError, onChunk }
   */
  constructor(patientId, sessionId, callbacks = {}) {
    this.patientId   = patientId;
    this.sessionId   = sessionId;
    this.callbacks   = callbacks;

    this._db               = null;
    this._latestRef        = null;
    this._statusRef        = null;
    this._latestUnsubscribe = null;
    this._statusUnsubscribe = null;

    // Chart data buffers (last 200 points per channel)
    this.chartData = {
      labels:   [],
      heel:     [], midfoot:   [], forefoot:  [],
      acc_x:    [], acc_y:     [], acc_z:     [],
      gyro_x:   [], gyro_y:    [], gyro_z:    [],
    };
    this.MAX_CHART_POINTS = 200;

    this._sampleCount     = 0;
    this._startTime       = null;
    this._lastHeartbeat   = null;
    this._connectionState = "disconnected"; // disconnected | loading | connected | error
  }

  // =========================================================================
  // Public API
  // =========================================================================

  /**
   * Start listening to Firebase for this patient + session.
   * Returns a Promise that resolves when listeners are attached.
   */
  async start() {
    this._setState("loading");

    try {
      // Import Firebase dynamically (CDN or bundled)
      const { initializeApp, getApps } = await import(
        "https://www.gstatic.com/firebasejs/10.14.0/firebase-app.js"
      );
      const { getDatabase, ref, onValue, off } = await import(
        "https://www.gstatic.com/firebasejs/10.14.0/firebase-database.js"
      );

      // Initialise Firebase app (once)
      const app = getApps().length === 0
        ? initializeApp(FIREBASE_CONFIG)
        : getApps()[0];

      this._db  = getDatabase(app);
      this._off = off;

      // Subscribe to "latest" reading (most recent sensor values)
      this._latestRef = ref(
        this._db,
        `live_sessions/${this.patientId}/${this.sessionId}/latest`
      );

      this._latestUnsubscribe = onValue(
        this._latestRef,
        (snapshot) => this._onLatestSnapshot(snapshot),
        (error)    => this._onError(error),
      );

      // Subscribe to session status
      this._statusRef = ref(
        this._db,
        `live_sessions/${this.patientId}/${this.sessionId}/status`
      );

      this._statusUnsubscribe = onValue(
        this._statusRef,
        (snapshot) => this._onStatusSnapshot(snapshot),
        (error)    => this._onError(error),
      );

      this._startTime = Date.now();
      console.log(
        `[GaitMonitor] Listening to: live_sessions/${this.patientId}/${this.sessionId}`
      );

    } catch (error) {
      console.error("[GaitMonitor] Failed to start:", error);
      this._setState("error");
      if (this.callbacks.onError) {
        this.callbacks.onError(error.message || "Failed to connect to Firebase");
      }
    }
  }

  /**
   * Stop all Firebase listeners.
   * Always call this when the component unmounts or the user navigates away.
   */
  stop() {
    if (this._off && this._latestRef) {
      this._off(this._latestRef);
    }
    if (this._off && this._statusRef) {
      this._off(this._statusRef);
    }
    this._latestUnsubscribe = null;
    this._statusUnsubscribe = null;
    this._setState("disconnected");
    console.log("[GaitMonitor] Listeners removed.");
  }

  /**
   * Returns the current connection state string.
   * One of: "disconnected" | "loading" | "connected" | "error"
   */
  getState() {
    return this._connectionState;
  }

  /**
   * Returns how many seconds have elapsed since recording started.
   */
  getRecordingDuration() {
    if (!this._startTime) return 0;
    return Math.floor((Date.now() - this._startTime) / 1000);
  }

  // =========================================================================
  // Private handlers
  // =========================================================================

  _onLatestSnapshot(snapshot) {
    if (!snapshot.exists()) {
      // No data yet — still waiting for the first sample
      this._setState("loading");
      return;
    }

    const data = snapshot.val();
    if (!data) return;

    this._setState("connected");
    this._lastHeartbeat = Date.now();
    this._sampleCount++;

    // Build a normalised reading object
    const reading = {
      timestamp_ms:  data.timestamp_ms  || 0,
      server_ts:     data.server_ts     || Date.now(),
      heel:          data.heel          || 0,
      midfoot:       data.midfoot       || 0,
      forefoot:      data.forefoot      || 0,
      acc_x_g:       data.acc_x_g       || 0,
      acc_y_g:       data.acc_y_g       || 0,
      acc_z_g:       data.acc_z_g       || 0,
      gyro_x_dps:    data.gyro_x_dps    || 0,
      gyro_y_dps:    data.gyro_y_dps    || 0,
      gyro_z_dps:    data.gyro_z_dps    || 0,
      patient_id:    data.patient_id    || this.patientId,
      session_id:    data.session_id    || this.sessionId,
      sample_count:  this._sampleCount,
    };

    // Update chart buffers
    this._pushToChart(reading);

    // Fire user callback
    if (this.callbacks.onLatest) {
      this.callbacks.onLatest(reading, this.chartData);
    }
  }

  _onStatusSnapshot(snapshot) {
    if (!snapshot.exists()) return;
    const data = snapshot.val();
    const status = data?.value || "unknown";

    if (this.callbacks.onStatus) {
      this.callbacks.onStatus({
        status,
        updated_at:  data?.updated_at,
        patient_id:  data?.patient_id || this.patientId,
        session_id:  data?.session_id || this.sessionId,
      });
    }

    if (status === "completed" || status === "failed") {
      this.stop();
    }
  }

  _onError(error) {
    console.error("[GaitMonitor] Firebase error:", error);
    this._setState("error");
    if (this.callbacks.onError) {
      this.callbacks.onError(error.message || "Firebase connection error");
    }
  }

  _setState(newState) {
    if (this._connectionState !== newState) {
      this._connectionState = newState;
      console.log("[GaitMonitor] State →", newState);
    }
  }

  _pushToChart(reading) {
    const label = (reading.timestamp_ms / 1000).toFixed(1) + "s";
    this.chartData.labels.push(label);
    this.chartData.heel.push(reading.heel);
    this.chartData.midfoot.push(reading.midfoot);
    this.chartData.forefoot.push(reading.forefoot);
    this.chartData.acc_x.push(reading.acc_x_g);
    this.chartData.acc_y.push(reading.acc_y_g);
    this.chartData.acc_z.push(reading.acc_z_g);
    this.chartData.gyro_x.push(reading.gyro_x_dps);
    this.chartData.gyro_y.push(reading.gyro_y_dps);
    this.chartData.gyro_z.push(reading.gyro_z_dps);

    // Keep rolling window
    if (this.chartData.labels.length > this.MAX_CHART_POINTS) {
      const keys = Object.keys(this.chartData);
      keys.forEach(k => { this.chartData[k].shift(); });
    }
  }
}


// ---------------------------------------------------------------------------
// GaitDashboardUI  — ready-made DOM widget (optional convenience class)
// ---------------------------------------------------------------------------

export class GaitDashboardUI {
  /**
   * Renders a live sensor dashboard in a container element.
   *
   * @param {HTMLElement} container   - The DOM element to render into.
   * @param {string}      patientId   - e.g. "patient_001"
   * @param {string}      sessionId   - e.g. "patient_001_20261111_143022"
   */
  constructor(container, patientId, sessionId) {
    this.container = container;
    this.monitor   = new GaitMonitor(patientId, sessionId, {
      onLatest: (r, chartData) => this._render(r, chartData),
      onStatus: (s) => this._renderStatus(s),
      onError:  (e) => this._renderError(e),
    });

    this._chart = null;
    this._init();
  }

  _init() {
    this.container.innerHTML = `
      <div class="gait-monitor" style="font-family:'Inter',sans-serif;background:#1B1B1B;border-radius:12px;padding:20px;color:#fff;">
        <!-- Status bar -->
        <div id="gm-status-bar" style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
          <div style="display:flex;align-items:center;gap:10px;">
            <span id="gm-led" style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#555;"></span>
            <strong id="gm-status-text" style="color:#A3A3A3;">Connecting…</strong>
          </div>
          <div style="color:#A3A3A3;font-size:13px;">
            Patient: <strong style="color:#FF6B00;" id="gm-patient"></strong>&nbsp;&nbsp;
            Session: <code id="gm-session" style="font-size:11px;color:#ccc;"></code>
          </div>
          <div style="color:#A3A3A3;font-size:13px;">
            Samples: <strong id="gm-count" style="color:#fff;">0</strong>&nbsp;&nbsp;
            Duration: <strong id="gm-duration" style="color:#fff;">0s</strong>
          </div>
        </div>

        <!-- KPI cards -->
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px;">
          ${['Heel FSR','Ball FSR','Toe FSR'].map((label, i) => `
            <div class="gm-kpi-card" style="background:#242424;border-radius:8px;padding:14px;border:1px solid #333;">
              <div style="color:#A3A3A3;font-size:12px;text-transform:uppercase;letter-spacing:0.05em;">${label}</div>
              <div id="gm-fsr-${i}" style="font-size:28px;font-weight:700;color:#FF6B00;margin:4px 0;">—</div>
              <div style="color:#A3A3A3;font-size:11px;">Raw ADC (0–4095)</div>
            </div>
          `).join('')}
        </div>

        <!-- IMU readings -->
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px;">
          ${[
            ['Accel X','gm-ax','g'],['Accel Y','gm-ay','g'],['Accel Z','gm-az','g'],
            ['Gyro X','gm-gx','°/s'],['Gyro Y','gm-gy','°/s'],['Gyro Z','gm-gz','°/s'],
          ].map(([label, id, unit]) => `
            <div style="background:#242424;border-radius:8px;padding:12px;border:1px solid #333;">
              <div style="color:#A3A3A3;font-size:11px;">${label}</div>
              <div id="${id}" style="font-size:20px;font-weight:600;color:#fff;">—</div>
              <div style="color:#555;font-size:10px;">${unit}</div>
            </div>
          `).join('')}
        </div>

        <!-- Chart canvas -->
        <div style="background:#242424;border-radius:8px;padding:14px;border:1px solid #333;">
          <div style="color:#A3A3A3;font-size:12px;margin-bottom:8px;text-transform:uppercase;">FSR Pressure — Live</div>
          <canvas id="gm-chart" height="120"></canvas>
        </div>

        <!-- Disclaimer -->
        <div style="margin-top:12px;padding:10px;background:#1a1a1a;border-radius:6px;border:1px solid #2a2a2a;color:#666;font-size:11px;">
          Research prototype. FSR values are raw ADC units (0–4095), not Newtons.
          Acceleration in g-force. Gyroscope in degrees per second.
          Not a clinical diagnostic device.
        </div>
      </div>
    `;

    this._durationInterval = setInterval(() => {
      const el = document.getElementById("gm-duration");
      if (el) el.textContent = this.monitor.getRecordingDuration() + "s";
    }, 1000);

    this.monitor.start();
  }

  _render(reading, chartData) {
    // Status LED → green
    const led = document.getElementById("gm-led");
    if (led) { led.style.background = "#22c55e"; }
    const statusText = document.getElementById("gm-status-text");
    if (statusText) { statusText.textContent = "● Live"; statusText.style.color = "#22c55e"; }

    // IDs
    const setEl = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };

    setEl("gm-patient",  reading.patient_id);
    setEl("gm-session",  reading.session_id.slice(-15));
    setEl("gm-count",    reading.sample_count);

    // FSR cards
    setEl("gm-fsr-0", reading.heel);
    setEl("gm-fsr-1", reading.midfoot);
    setEl("gm-fsr-2", reading.forefoot);

    // IMU
    setEl("gm-ax", reading.acc_x_g.toFixed(3));
    setEl("gm-ay", reading.acc_y_g.toFixed(3));
    setEl("gm-az", reading.acc_z_g.toFixed(3));
    setEl("gm-gx", reading.gyro_x_dps.toFixed(2));
    setEl("gm-gy", reading.gyro_y_dps.toFixed(2));
    setEl("gm-gz", reading.gyro_z_dps.toFixed(2));

    // Chart (requires Chart.js loaded on the page)
    this._updateChart(chartData);
  }

  _renderStatus(status) {
    const led = document.getElementById("gm-led");
    const text = document.getElementById("gm-status-text");
    const colors = {
      recording:    "#22c55e",
      connected:    "#22c55e",
      completed:    "#FF6B00",
      disconnected: "#ef4444",
      failed:       "#ef4444",
    };
    if (led)  led.style.background  = colors[status.status] || "#555";
    if (text) text.textContent = status.status.toUpperCase();
  }

  _renderError(errorMsg) {
    const led  = document.getElementById("gm-led");
    const text = document.getElementById("gm-status-text");
    if (led)  led.style.background = "#ef4444";
    if (text) { text.textContent = "Error: " + errorMsg; text.style.color = "#ef4444"; }
  }

  _updateChart(chartData) {
    const canvas = document.getElementById("gm-chart");
    if (!canvas || typeof Chart === "undefined") return;

    if (!this._chart) {
      this._chart = new Chart(canvas, {
        type: "line",
        data: {
          labels:   chartData.labels,
          datasets: [
            {
              label: "Heel",
              data:  chartData.heel,
              borderColor: "#FF6B00",
              backgroundColor: "rgba(255,107,0,0.07)",
              borderWidth: 1.5, pointRadius: 0, tension: 0.3,
            },
            {
              label: "Ball",
              data:  chartData.midfoot,
              borderColor: "#FF8A24",
              backgroundColor: "rgba(255,138,36,0.07)",
              borderWidth: 1.5, pointRadius: 0, tension: 0.3,
            },
            {
              label: "Toe",
              data:  chartData.forefoot,
              borderColor: "#fbbf24",
              backgroundColor: "rgba(251,191,36,0.07)",
              borderWidth: 1.5, pointRadius: 0, tension: 0.3,
            },
          ],
        },
        options: {
          responsive: true,
          animation:  { duration: 0 },
          plugins:    { legend: { labels: { color: "#A3A3A3", font: { size: 11 } } } },
          scales: {
            x: {
              ticks:  { color: "#555", maxTicksLimit: 10 },
              grid:   { color: "#2a2a2a" },
            },
            y: {
              min:    0,
              max:    4095,
              title:  { display: true, text: "ADC (0–4095)", color: "#555", font: { size: 10 } },
              ticks:  { color: "#555" },
              grid:   { color: "#2a2a2a" },
            },
          },
        },
      });
    } else {
      // Update existing chart
      this._chart.data.labels   = chartData.labels;
      this._chart.data.datasets[0].data = chartData.heel;
      this._chart.data.datasets[1].data = chartData.midfoot;
      this._chart.data.datasets[2].data = chartData.forefoot;
      this._chart.update("none");
    }
  }

  destroy() {
    clearInterval(this._durationInterval);
    if (this._chart) this._chart.destroy();
    this.monitor.stop();
  }
}

