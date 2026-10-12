from pathlib import Path
import threading
import time

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.results import (
    BASE_DIR, FEATURES, IMPROVED_DIR, build_payload, clean, feature_ranges, read_csv,
)

try:
    from src.live_capture import (
        DEMO_SCENARIOS, CaptureError, LiveMonitor, capture_status, list_interfaces,
    )
    from src.live_predictor import LivePredictor, ModelCompatibilityError
    LIVE_IMPORT_ERROR = None
except Exception as error:  # live capture is optional
    LIVE_IMPORT_ERROR = str(error)


MODELS_DIR = BASE_DIR / "models"
LIGHTWEIGHT_MODEL = MODELS_DIR / "random_forest_10_trees.joblib"
EDGE_MODEL = MODELS_DIR / "edge_ids_model.joblib"
EDGE_MODEL_NAME = "Random Forest (10 trees, depth ≤ 12)"
DIST_DIR = BASE_DIR / "frontend" / "dist"

app = FastAPI(title="5G IDS Cross-Domain Dashboard API")

# allow the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL CACHE
# ============================================================

_models = {}
_model_lock = threading.Lock()


def load_model(path):
    with _model_lock:
        if path not in _models:
            if not path.exists():
                return None
            model = joblib.load(path)
            # single thread for one-row predictions
            if "n_jobs" in model.get_params():
                model.set_params(n_jobs=1)
            _models[path] = model
        return _models[path]


def malicious_probability(model, X):
    column = list(model.classes_).index(1)
    return model.predict_proba(X)[:, column]


# ============================================================
# RESULTS AND STATUS
# ============================================================

@app.get("/api/results")
def results():
    return build_payload()


@app.get("/api/status")
def status():
    live = {"imports_ok": LIVE_IMPORT_ERROR is None, "error": LIVE_IMPORT_ERROR}
    if LIVE_IMPORT_ERROR is None:
        available, message = capture_status()
        live.update({"capture_available": available, "capture_message": message,
                     "scenarios": DEMO_SCENARIOS})
    return {
        "models": {
            "rf10": LIGHTWEIGHT_MODEL.exists(),
            "edge": EDGE_MODEL.exists(),
        },
        "stream_samples": (IMPROVED_DIR / "stream_samples.csv").exists(),
        "live": live,
    }


# ============================================================
# PREDICTION DEMO (10-tree Random Forest)
# ============================================================

class PredictRequest(BaseModel):
    Rate: float = Field(ge=0)
    Packet_Count: float = Field(ge=0)
    Mean_Packet_Size: float = Field(ge=0)
    TTL: float = Field(ge=0, le=255)
    protocol: str = "UDP"


@app.get("/api/predict/ranges")
def predict_ranges():
    return feature_ranges()


@app.post("/api/predict")
def predict(request: PredictRequest):
    model = load_model(LIGHTWEIGHT_MODEL)
    if model is None:
        raise HTTPException(404, "models/random_forest_10_trees.joblib not found. "
                                 "Run `python src/test_lightweight_models.py` locally.")
    protocol = request.protocol.upper()
    values = {
        "Rate": request.Rate,
        "Packet_Count": request.Packet_Count,
        "Mean_Packet_Size": request.Mean_Packet_Size,
        "TTL": request.TTL,
        "TCP": 1.0 if protocol == "TCP" else 0.0,
        "UDP": 1.0 if protocol == "UDP" else 0.0,
        "ICMP": 1.0 if protocol == "ICMP" else 0.0,
    }
    X = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    ranges = feature_ranges()
    outside = [f for f, r in ranges.items() if not (r["min"] <= values[f] <= r["max"])]
    return {
        "prediction": int(model.predict(X)[0]),
        "probability": float(malicious_probability(model, X)[0]),
        "input": values,
        "outside_training_range": outside,
    }


# ============================================================
# EDGE IDS MONITOR (replay of recorded real flows)
# ============================================================

@app.get("/api/stream")
def stream(dataset: str = "5G-NIDD", n: int = 150):
    samples = read_csv(IMPROVED_DIR / "stream_samples.csv")
    if samples is None:
        raise HTTPException(404, "results/improved/stream_samples.csv not found. "
                                 "Run `python src/run_improved_experiments.py` locally.")

    path, label = (EDGE_MODEL, EDGE_MODEL_NAME + " (balanced, deduplicated training)") \
        if EDGE_MODEL.exists() else (LIGHTWEIGHT_MODEL, "Random Forest (10 trees), original")
    model = load_model(path)
    if model is None:
        raise HTTPException(404, "No trained model found in models/. "
                                 "Run `python src/run_improved_experiments.py` locally.")

    feed = samples[samples["Dataset"] == dataset].head(max(1, min(n, 1000)))
    flows = []
    for _, row in feed.iterrows():
        X = row[FEATURES].to_frame().T.astype(float)
        start = time.perf_counter()
        prediction = int(model.predict(X)[0])
        latency = (time.perf_counter() - start) * 1000
        flows.append({
            **{f: float(row[f]) for f in FEATURES},
            "target": int(row["Target"]),
            "attack_type": str(row["Attack_Type"]),
            "prediction": prediction,
            "latency_ms": latency,
        })
    return clean({"model": label, "dataset": dataset, "flows": flows})


# ============================================================
# LIVE IDS MONITORING (wraps src/live_capture.py)
# ============================================================

_monitor = None
_monitor_lock = threading.Lock()


class LiveStartRequest(BaseModel):
    mode: str = "demo"                 # "demo" or "capture"
    interface: str | None = None
    host_filter: str | None = None
    scenario: str = "auto"
    window_seconds: float = Field(default=5.0, ge=1, le=30)


def require_live():
    if LIVE_IMPORT_ERROR is not None:
        raise HTTPException(503, f"Live IDS modules could not be loaded: {LIVE_IMPORT_ERROR}")


@app.get("/api/live/interfaces")
def live_interfaces():
    require_live()
    available, message = capture_status()
    if not available:
        return {"available": False, "message": message, "interfaces": []}
    try:
        return {"available": True, "message": message, "interfaces": list_interfaces()}
    except Exception as error:
        return {"available": False, "message": f"Interfaces could not be listed: {error}",
                "interfaces": []}


@app.post("/api/live/start")
def live_start(request: LiveStartRequest):
    global _monitor
    require_live()
    if request.mode not in ("demo", "capture"):
        raise HTTPException(400, "mode must be 'demo' or 'capture'")
    with _monitor_lock:
        if _monitor is not None and _monitor.running:
            raise HTTPException(409, "Monitoring is already running.")
        try:
            model = load_model(LIGHTWEIGHT_MODEL)
            predictor = LivePredictor(LIGHTWEIGHT_MODEL, model=model)
        except ModelCompatibilityError as error:
            raise HTTPException(404, f"The live IDS needs the trained 10-tree Random Forest. {error}")
        monitor = LiveMonitor(
            predictor,
            mode=request.mode,
            window_seconds=request.window_seconds,
            interface=request.interface,
            host_filter=request.host_filter,
            demo_scenario=request.scenario,
        )
        try:
            monitor.start()
        except CaptureError as error:
            raise HTTPException(400, str(error))
        _monitor = monitor
    return {"started": True, "model": predictor.describe()}


@app.post("/api/live/stop")
def live_stop():
    require_live()
    if _monitor is not None:
        _monitor.stop()
    return {"stopped": True}


@app.post("/api/live/clear")
def live_clear():
    require_live()
    if _monitor is not None:
        _monitor.clear_history()
    return {"cleared": True}


@app.get("/api/live/state")
def live_state(flows: int = 200):
    require_live()
    if _monitor is None:
        return {"active": False}
    state = _monitor.snapshot()      # also keeps the monitor alive
    # newest flows first
    state["flows"] = list(reversed(state["flows"]))[:max(0, flows)]
    state["active"] = True
    return clean(state)


# ============================================================
# REACT APP
# ============================================================

if (DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    if path.startswith("api/"):
        raise HTTPException(404, "Unknown API route")
    target = DIST_DIR / path
    if path and target.is_file() and DIST_DIR in target.resolve().parents:
        return FileResponse(target)
    index = DIST_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {
        "message": "The React app has not been built yet. Run `npm install` and "
                   "`npm run build` inside frontend/, or use `npm run dev` for development."
    }
