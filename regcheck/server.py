"""Flask web server for the reg-check inspection bay.

Thin layer over the pipeline: it serves the single-page UI, stores DVSA
credentials on disk (never sending the secrets back to the browser), starts one
inspection run at a time on a worker thread, and streams every event to the page
over Server-Sent Events.
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading

from flask import Flask, Response, request, send_from_directory, stream_with_context

from .engine import Pipeline
from .mot import MOTClient
from .review import PROVIDERS, is_local, list_models, run_tournament

# Config + results live next to the executable (a PyInstaller build) or at the
# project root (running from source), so they are easy to find and writable.
if getattr(sys, "frozen", False):
    ROOT = os.path.dirname(sys.executable)
else:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "reg_check_config.json")
OUTPUT_FILE = os.path.join(ROOT, "extraction_results.txt")
CRED_FIELDS = ("MOT_CLIENT_ID", "MOT_CLIENT_SECRET", "MOT_API_KEY", "MOT_TOKEN_URL")

app = Flask(__name__, static_folder="static", static_url_path="")


# --- credential storage -----------------------------------------------------

def load_config() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_config(data: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def effective(field: str) -> str:
    """Saved value, falling back to the matching environment variable."""
    return (load_config().get(field) or os.environ.get(field, "")).strip()


# --- event broker (SSE) -----------------------------------------------------

class Broker:
    def __init__(self):
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.subs: list[queue.Queue] = []
        self.counter = 0

    def reset(self):
        with self.lock:
            self.events = []
            self.counter = 0

    def publish(self, ev: dict):
        # _n lets the browser ignore events it already rendered if the SSE
        # connection drops and replays the run's history on reconnect.
        with self.lock:
            ev = {**ev, "_n": self.counter}
            self.counter += 1
            self.events.append(ev)
            subs = list(self.subs)
        for q in subs:
            q.put(ev)

    def subscribe(self):
        q: queue.Queue = queue.Queue()
        with self.lock:
            history = list(self.events)
            self.subs.append(q)
        return q, history

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)


broker = Broker()
run_lock = threading.Lock()
state = {"running": False, "stop_event": None}

# Verified vehicles from the most recent run, kept so the AI review can use them.
last_results: list = []

review_broker = Broker()
review_lock = threading.Lock()
review_state = {"running": False}


def _capture_emit(ev):
    """Publish a pipeline event, keeping verified results for the AI review."""
    if ev.get("type") == "result":
        last_results.append(ev["result"])
    broker.publish(ev)


# --- routes -----------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/config")
def get_config():
    """Report which credentials are set - never returns the secret values."""
    fields = {f: bool(effective(f)) for f in CRED_FIELDS}
    return {"fields": fields, "configured": all(fields.values())}


@app.post("/api/config")
def post_config():
    """Save the non-empty credential fields supplied (blank = keep existing)."""
    body = request.get_json(silent=True) or {}
    cfg = load_config()
    for f in CRED_FIELDS:
        val = (body.get(f) or "").strip()
        if val:
            cfg[f] = val
    try:
        save_config(cfg)
    except Exception as exc:
        return {"error": f"could not save credentials: {exc}"}, 500
    fields = {f: bool(effective(f)) for f in CRED_FIELDS}
    return {"ok": True, "fields": fields, "configured": all(fields.values())}


@app.get("/api/state")
def get_state():
    return {"running": state["running"]}


@app.post("/api/run")
def post_run():
    with run_lock:
        if state["running"]:
            return {"error": "An inspection is already running."}, 409
        body = request.get_json(silent=True) or {}
        urls = re.findall(r"https?://\S+", body.get("urls", ""))
        urls = [u.rstrip(",") for u in urls]
        if not urls:
            return {"error": "Paste at least one Auto Trader or Cazoo link."}, 400
        options = _clean_options(body.get("options") or {})
        mot = MOTClient(effective("MOT_CLIENT_ID"), effective("MOT_CLIENT_SECRET"),
                        effective("MOT_API_KEY"), effective("MOT_TOKEN_URL"))
        stop_event = threading.Event()
        broker.reset()
        last_results.clear()
        state.update(running=True, stop_event=stop_event)
        pipeline = Pipeline(urls, mot, _capture_emit, stop_event, OUTPUT_FILE,
                            options)
        threading.Thread(target=_worker, args=(pipeline,), daemon=True).start()
    return {"ok": True, "count": len(urls), "verified": mot.configured}


@app.post("/api/lookup")
def post_lookup():
    """Verify a hand-typed registration (for a manual-review listing) against DVSA
    and, if real, return it as a result and add it to the AI-review pool."""
    import requests as _rq
    from .mot import shape_vehicle
    body = request.get_json(silent=True) or {}
    plate = (body.get("plate") or "").strip().upper()
    if not re.sub(r"[^A-Z0-9]", "", plate):
        return {"error": "Enter a registration."}, 400
    mot = MOTClient(effective("MOT_CLIENT_ID"), effective("MOT_CLIENT_SECRET"),
                    effective("MOT_API_KEY"), effective("MOT_TOKEN_URL"))
    if not mot.configured:
        return {"error": "MOT API credentials aren't set."}, 400
    try:
        vehicle = mot.lookup(plate, _rq)
    except Exception as exc:
        return {"error": f"MOT lookup failed: {exc}"}, 502
    if not vehicle:
        return {"error": f"{plate} isn't a DVSA-registered vehicle."}, 404
    from .mot import vehicle_tier
    l_make, l_model = body.get("make") or "", body.get("model") or ""
    warning = None
    if (l_make or l_model) and vehicle_tier(l_make, l_model, vehicle) < 2:
        warning = (f"DVSA says {plate} is a {vehicle.get('make', '')} "
                   f"{vehicle.get('model', '')}, but the listing is a {l_make} "
                   f"{l_model} - double-check the reg.")
    result = {"plate": plate, "verified": True, "corrected": False, "tier": 3,
              "manual": True, "votes": None, "warning": warning,
              "price": body.get("price") or None,
              "location": body.get("location") or None, "distanceMiles": None,
              "url": body.get("url") or None, "site": body.get("site") or "manual",
              "make": vehicle.get("make", ""), "model": vehicle.get("model", ""),
              **shape_vehicle(vehicle)}
    last_results.append(result)
    return {"ok": True, "result": result}


@app.post("/api/stop")
def post_stop():
    ev = state.get("stop_event")
    if state["running"] and ev:
        ev.set()
        return {"ok": True}
    return {"ok": False, "error": "Nothing is running."}


@app.get("/api/stream")
def stream():
    @stream_with_context
    def gen():
        q, history = broker.subscribe()
        try:
            for ev in history:
                yield _sse(ev)
            while True:
                try:
                    yield _sse(q.get(timeout=15))
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            broker.unsubscribe(q)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


def _worker(pipeline: Pipeline):
    try:
        pipeline.run()
    finally:
        with run_lock:
            state["running"] = False


# --- AI review (local LLM tournament) ---------------------------------------

def _llm_settings(provider=None, body=None):
    """(provider, base_url, api_key, model) for a review: request body first, then
    saved settings, then defaults. Each provider remembers its own model."""
    body = body or {}
    cfg = load_config()
    provider = provider or body.get("provider") or cfg.get("LLM_PROVIDER") or "local"
    if provider not in PROVIDERS:
        provider = "local"
    spec = PROVIDERS[provider]
    if provider == "local":
        base_url = (body.get("base_url") or cfg.get("LLM_BASE_URL")
                    or spec["base_url"]).strip()
    else:
        base_url = spec["base_url"]
    api_key = effective(spec["key"]) if spec["key"] else None
    model = (body.get("model") or cfg.get(f"LLM_MODEL_{provider}")
             or (cfg.get("LLM_MODEL") if provider == "local" else None)
             or spec["default_model"]).strip()
    return provider, base_url, api_key, model


def _llm_state(provider=None):
    """What the page needs to show for a provider - never the API key itself."""
    provider, base_url, api_key, model = _llm_settings(provider)
    spec = PROVIDERS[provider]
    installed, reach_error = [], None
    if spec["key"] and not api_key:
        reach_error = f"Add your {spec['label']} API key to use it."
    else:
        try:
            installed = list_models(base_url, api_key)
        except Exception as exc:
            reach_error = f"Can't reach {spec['label']} ({exc})"
    suggested = (spec["default_model"] if spec["default_model"] in installed
                 else (installed[0] if installed else None))
    return {"provider": provider,
            "providers": [{"id": k, "label": v["label"], "needs_key": bool(v["key"])}
                          for k, v in PROVIDERS.items()],
            "base_url": base_url, "model": model, "installed": installed,
            "model_ok": model in installed, "suggested": suggested,
            "key_set": bool(api_key) if spec["key"] else None,
            "cloud": not is_local(base_url), "reach_error": reach_error,
            "results": len(last_results), "running": review_state["running"]}


@app.get("/api/llm-config")
def get_llm_config():
    """Review settings for a provider plus the models it actually offers, so the
    page can list them and warn if the saved model has gone."""
    return _llm_state(request.args.get("provider"))


@app.post("/api/llm-config")
def post_llm_config():
    """Save the chosen provider and, if given, a provider API key (write-only)."""
    body = request.get_json(silent=True) or {}
    cfg = load_config()
    provider = body.get("provider") if body.get("provider") in PROVIDERS else None
    if provider:
        cfg["LLM_PROVIDER"] = provider
    for spec in PROVIDERS.values():
        if spec["key"] and (body.get(spec["key"]) or "").strip():
            cfg[spec["key"]] = body[spec["key"]].strip()
    try:
        save_config(cfg)
    except Exception as exc:
        return {"error": f"could not save settings: {exc}"}, 500
    return _llm_state(provider)


@app.post("/api/review")
def post_review():
    with review_lock:
        if review_state["running"]:
            return {"error": "A review is already running."}, 409
        if not last_results:
            return {"error": "Run an inspection first - no verified vehicles yet."}, 400
        body = request.get_json(silent=True) or {}
        provider, base_url, api_key, model = _llm_settings(body=body)
        spec = PROVIDERS[provider]
        if spec["key"] and not api_key:
            return {"error": f"Add your {spec['label']} API key first."}, 400
        try:
            shortlist = max(1, min(25, int(body.get("shortlist", 10))))
        except (TypeError, ValueError):
            shortlist = 10
        brief = (body.get("brief") or "").strip()[:500]
        cfg = load_config()
        cfg["LLM_PROVIDER"], cfg[f"LLM_MODEL_{provider}"] = provider, model
        if provider == "local":
            cfg["LLM_BASE_URL"] = base_url
        try:
            save_config(cfg)
        except Exception:
            pass
        vehicles = list(last_results)
        review_state["running"] = True
        review_broker.reset()
        threading.Thread(target=_review_worker,
                         args=(vehicles, base_url, model, shortlist, brief, api_key,
                               spec["label"]),
                         daemon=True).start()
    return {"ok": True, "count": len(vehicles), "model": model, "provider": provider}


def _review_worker(vehicles, base_url, model, shortlist=10, brief="", api_key=None,
                   label=""):
    def log(text):
        review_broker.publish({"type": "log", "text": text})
    try:
        note = f' - priorities: "{brief}"' if brief else ""
        cloud = ("" if is_local(base_url)
                 else " (cloud: vehicle details are sent to this service)")
        log(f"[Review] Reviewing {len(vehicles)} verified vehicle(s) with {model} "
            f"on {label}{cloud} - Swiss-system, every vehicle compared, none "
            f"dropped{note}...")
        result = run_tournament(vehicles, base_url, model, log=log, api_key=api_key,
                                shortlist=shortlist, brief=brief)
        review_broker.publish({"type": "shortlist", "model": model, "provider": label,
                               **result})
    except Exception as exc:
        review_broker.publish({"type": "log", "text": f"[Review] Failed: {exc}"})
        review_broker.publish({"type": "shortlist", "shortlist": [],
                               "leaderboard": [], "error": str(exc)})
    finally:
        review_broker.publish({"type": "done"})
        with review_lock:
            review_state["running"] = False


@app.get("/api/review/stream")
def review_stream():
    @stream_with_context
    def gen():
        q, history = review_broker.subscribe()
        try:
            for ev in history:
                yield _sse(ev)
            while True:
                try:
                    yield _sse(q.get(timeout=15))
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            review_broker.unsubscribe(q)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _clean_options(opts: dict) -> dict:
    out = {}
    for key, lo, hi in (("max_images", 1, 60), ("early_stop_votes", 0, 20),
                        ("min_images", 1, 60)):
        if key in opts:
            try:
                out[key] = max(lo, min(hi, int(opts[key])))
            except (TypeError, ValueError):
                pass
    if "near_miss" in opts:
        out["near_miss"] = bool(opts["near_miss"])
    return out


def _sse(ev: dict) -> str:
    return f"data: {json.dumps(ev)}\n\n"
