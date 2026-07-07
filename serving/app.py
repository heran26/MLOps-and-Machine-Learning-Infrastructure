"""
FastAPI inference server for the current production model.

Adds three things beyond plain inference, needed for the drift/bias monitor:
- Every /predict call is logged in memory (features + output + timestamp).
- GET /monitoring/raw-data exposes that log to the drift-check job (API-key protected).
- POST /monitoring/report lets the drift-check job publish its latest findings.
- GET /dashboard renders those findings as a simple live page.

Note: the prediction log is IN-MEMORY, so it resets whenever the service
restarts/redeploys. That's fine for demonstrating the concept; a real
production system would write this to a database (e.g. Postgres) instead.

Run locally:   uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import json
import threading
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

MODEL_PATH = os.environ.get("MODEL_PATH", "models/production_model.pkl")
MONITOR_API_KEY = os.environ.get("MONITOR_API_KEY", "")  # shared secret, see README
MAX_LOG_SIZE = 5000  # cap memory usage; oldest entries drop off

app = FastAPI(title="Income Prediction API")
model = joblib.load(MODEL_PATH)

_lock = threading.Lock()
prediction_log = []          # list of dicts: {"timestamp", "features": {...}, "prediction", "probability"}
latest_drift_report = None   # most recent report POSTed by the drift-check job


class PredictionRequest(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


def check_api_key(x_api_key: str | None):
    if not MONITOR_API_KEY:
        # No key configured -- monitoring endpoints are open. Fine for local
        # testing, NOT recommended once this is public. See README.
        return
    if x_api_key != MONITOR_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    features = request.dict()
    df = pd.DataFrame([features])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "prediction": prediction,
        "probability": probability,
    }
    with _lock:
        prediction_log.append(entry)
        if len(prediction_log) > MAX_LOG_SIZE:
            del prediction_log[: len(prediction_log) - MAX_LOG_SIZE]

    return {"prediction": prediction, "probability_over_50k": probability}


@app.get("/monitoring/raw-data")
def monitoring_raw_data(x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)
    with _lock:
        return {"count": len(prediction_log), "records": list(prediction_log)}


@app.post("/monitoring/report")
def monitoring_report(report: dict, x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)
    global latest_drift_report
    latest_drift_report = report
    return {"status": "stored"}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    report = latest_drift_report
    report_json = json.dumps(report) if report else "null"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
      <title>Model Monitoring Dashboard</title>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
      <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }}
        h1 {{ font-size: 1.4rem; }}
        .alert {{ padding: 12px 16px; border-radius: 8px; margin: 16px 0; font-weight: 600; }}
        .alert-ok {{ background: #e6f4ea; color: #1e7e34; }}
        .alert-bad {{ background: #fce8e6; color: #c5221f; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
        th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #eee; font-size: 0.9rem; }}
        canvas {{ max-height: 320px; }}
        .muted {{ color: #666; font-size: 0.85rem; }}
      </style>
    </head>
    <body>
      <h1>Model Monitoring Dashboard</h1>
      <div id="content">Loading...</div>
      <script>
        const report = {report_json};
        const el = document.getElementById('content');

        if (!report) {{
          el.innerHTML = '<p class="muted">No drift report yet. The scheduled monitoring job hasn\\'t run, or hasn\\'t posted a report to this service since it last restarted.</p>';
        }} else {{
          let html = '<p class="muted">Last checked: ' + report.timestamp + ' | samples analyzed: ' + report.n_samples + '</p>';

          html += '<div class="alert ' + (report.alert ? 'alert-bad' : 'alert-ok') + '">' +
                  (report.alert ? 'ALERT: drift or bias threshold exceeded -- retraining triggered' : 'All checks within normal range') +
                  '</div>';

          html += '<h2>Feature drift (Population Stability Index)</h2>';
          html += '<canvas id="psiChart"></canvas>';

          html += '<h2>Fairness across groups (selection rate)</h2><table><tr><th>Attribute</th><th>Group</th><th>Selection rate</th></tr>';
          for (const [attr, groups] of Object.entries(report.bias.selection_rates || {{}})) {{
            for (const [group, rate] of Object.entries(groups)) {{
              html += '<tr><td>' + attr + '</td><td>' + group + '</td><td>' + (rate * 100).toFixed(1) + '%</td></tr>';
            }}
          }}
          html += '</table>';
          html += '<p class="muted">Four-fifths rule: flagged if any group\\'s selection rate falls below 80% of the highest group\\'s rate. Bias flag: ' + report.bias.bias_flag + '</p>';

          el.innerHTML = html;

          const labels = Object.keys(report.feature_psi);
          const values = Object.values(report.feature_psi);
          new Chart(document.getElementById('psiChart'), {{
            type: 'bar',
            data: {{
              labels: labels,
              datasets: [{{
                label: 'PSI',
                data: values,
                backgroundColor: values.map(v => v > 0.2 ? '#c5221f' : (v > 0.1 ? '#f9a825' : '#1e7e34')),
              }}]
            }},
            options: {{
              scales: {{ y: {{ beginAtZero: true }} }},
              plugins: {{ legend: {{ display: false }} }}
            }}
          }});
        }}
      </script>
    </body>
    </html>
    """
