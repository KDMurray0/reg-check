"""Tournament MOT review with a local LLM.

Once the inspection has verified a set of vehicles, a local model (via any
OpenAI-compatible endpoint - Ollama, LM Studio, llama.cpp, vLLM) reviews them and
shortlists the best buys, weighing year, mileage, price, distance and the full
MOT history (failures, dangerous defects, recurring advisories, mileage
consistency).

It runs as a Swiss-system tournament (as chess and LLM-arena leaderboards do) so
no vehicle is ever eliminated: every vehicle is compared in small peer groups over
several rounds and earns an Elo rating from the head-to-head results, with the
group boundaries shifting each round so a borderline vehicle meets the rivals it
just missed. Because an Elo win over a weak group counts for little, a vehicle
can't top the table just by beating weak peers. Every vehicle ends up on a full
leaderboard; the top few get written-up pros and cons. If the model returns
unparseable output for a group, that group keeps its current order, so a flaky
reply never drops or blindly reshuffles a vehicle.
"""

from __future__ import annotations

import datetime
import json
import random
import re
import time

import requests

DEFAULT_BASE_URL = "http://localhost:11434/v1"   # Ollama's OpenAI-compatible API
DEFAULT_MODEL = "qwen2.5:32b"    # non-"thinking" and proven to rank correctly here

# Where the review can run. Both speak the OpenAI chat API; Groq is a hosted
# cloud service (vehicle data is sent to it) and needs an API key.
PROVIDERS = {
    "local": {"label": "Local (Ollama)", "base_url": DEFAULT_BASE_URL,
              "default_model": DEFAULT_MODEL, "key": None},
    "groq": {"label": "Groq (cloud)", "base_url": "https://api.groq.com/openai/v1",
             "default_model": "openai/gpt-oss-120b", "key": "GROQ_API_KEY"},
}

# Model ids that aren't chat models (speech, moderation, embeddings).
_NON_CHAT = re.compile(r"whisper|tts|orpheus|guard|embed|playai", re.I)
BATCH_SIZE = 6          # vehicles compared per peer group (small = reliable ranking)
ROUNDS = 4              # Swiss rounds - every vehicle is ranked in each
_NOW_YEAR = datetime.date.today().year

# Recurring-issue themes worth surfacing to the model / buyer.
_THEMES = {
    "corrosion": ("corro", "rust", "weld"),
    "oil leak": ("oil leak", "leaking oil", "engine oil"),
    "brakes": ("brake", "disc", "pad"),
    "tyres": ("tyre", "tread"),
    "suspension": ("suspension", "shock", "spring", "bush", "ball joint", "arm"),
    "emissions": ("emission", "exhaust", "smoke", "lambda"),
    "steering": ("steering", "track rod", "rack"),
}


# --- derived signals --------------------------------------------------------

def _int_miles(s):
    if not s:
        return None
    m = re.search(r"[\d,]+", str(s))
    return int(m.group(0).replace(",", "")) if m else None


def _year(v):
    fu = (v.get("firstUsed") or "")[:4]
    return int(fu) if fu.isdigit() else None


def _date(s):
    try:
        return datetime.date.fromisoformat((s or "")[:10])
    except ValueError:
        return None


def derive(v: dict) -> dict:
    """Buyer-relevant signals computed from a verified vehicle record.

    Separates the things that matter from the noise: a failure fixed on a retest
    within weeks is routine, a vehicle whose LATEST test failed is not; a
    dangerous defect last year matters more than one a decade ago."""
    tests = v.get("tests") or []
    today = datetime.date.today()
    year = _year(v)
    miles = _int_miles(v.get("latestMileage"))
    age = (_NOW_YEAR - year) if year else None
    per_year = int(miles / age) if (miles and age and age > 0) else None

    # Odometer going backwards between tests = possible clock / anomaly.
    seq = [_int_miles(t.get("mileage")) for t in tests]
    seq = [m for m in seq if m is not None]
    clocking = any(b < a - 1000 for a, b in zip(seq, seq[1:]))

    dated = [(t, _date(t.get("date"))) for t in tests]
    recent = [(t, dt) for t, dt in dated if dt and (today - dt).days <= 3 * 365]
    fails_3y = retested = 0
    for k, (t, dt) in enumerate(dated):
        if t.get("result") == "FAILED" and dt and (today - dt).days <= 3 * 365:
            fails_3y += 1
            nxt = dated[k + 1] if k + 1 < len(dated) else None
            if (nxt and nxt[0].get("result") == "PASSED" and nxt[1]
                    and (nxt[1] - dt).days <= 60):
                retested += 1
    last = tests[-1] if tests else {}
    unresolved = last.get("result") == "FAILED"          # latest test is a fail
    recent_fail = fails_3y > 0
    dangerous = sum(1 for t in tests for d in (t.get("defects") or []) if d.get("dangerous"))
    dangerous_recent = sum(1 for t, _ in recent for d in (t.get("defects") or [])
                           if d.get("dangerous"))
    last_pass = next((t for t in reversed(tests) if t.get("result") == "PASSED"), None)
    last_advisories = (sum(1 for d in (last_pass.get("defects") or [])
                           if (d.get("type") or "").upper() in ("ADVISORY", "MINOR"))
                       if last_pass else None)
    exp = _date(v.get("motExpiry"))
    mot_months = round((exp - today).days / 30.4) if exp else None

    theme_counts = {}
    for t in tests:
        for d in (t.get("defects") or []):
            text = (d.get("text") or "").lower()
            for name, keys in _THEMES.items():
                if any(k in text for k in keys):
                    theme_counts[name] = theme_counts.get(name, 0) + 1
    recurring = sorted((n for n, c in theme_counts.items() if c >= 3),
                       key=lambda n: -theme_counts[n])

    return {"year": year, "miles": miles, "age": age, "per_year": per_year,
            "clocking": clocking, "recent_fail": recent_fail, "fails_3y": fails_3y,
            "retested": retested, "unresolved": unresolved,
            "dangerous": dangerous, "dangerous_recent": dangerous_recent,
            "last_result": last.get("result"), "last_date": last.get("date"),
            "last_advisories": last_advisories, "mot_months": mot_months,
            "recurring": recurring}


def _price_int(v):
    return _int_miles(v.get("price"))


def compact_dossier(i: int, v: dict) -> str:
    d = derive(v)
    bits = [f"#{i} {v.get('plate','?')}"]
    bits.append(f"{d['year'] or '?'} {v.get('make','')} {v.get('model','')}".strip())
    bits.append(v.get("price") or "price n/a")
    miles = f"{d['miles']:,} mi" if d["miles"] else "mileage n/a"
    if d["per_year"]:
        miles += f" (~{d['per_year']:,}/yr)"
    bits.append(miles)
    bits.append(v.get("location") or "location n/a")
    if d["last_result"]:
        last = f"latest MOT {d['last_result']} {(d['last_date'] or '')[:7]}"
        if d["last_advisories"] is not None and d["last_result"] == "PASSED":
            last += f", {d['last_advisories']} advisories"
        if d["mot_months"] is not None:
            last += (f", {d['mot_months']} months left" if d["mot_months"] >= 0
                     else ", MOT EXPIRED")
        bits.append(last)
    if d["unresolved"]:
        bits.append("LATEST TEST FAILED (not yet passed)")
    if d["fails_3y"]:
        bits.append(f"{d['fails_3y']} fail(s) in last 3y, {d['retested']} fixed on retest")
    if d["dangerous"]:
        bits.append(f"dangerous defects: {d['dangerous_recent']} in last 3y, "
                    f"{d['dangerous'] - d['dangerous_recent']} older")
    if d["clocking"]:
        bits.append("MILEAGE WENT BACKWARDS between tests")
    if d["recurring"]:
        bits.append("recurring advisories: " + ", ".join(d["recurring"]))
    return " | ".join(bits)


def full_dossier(i: int, v: dict) -> str:
    """Compact dossier plus the last few notable MOT events, for the final round."""
    lines = [compact_dossier(i, v)]
    tests = v.get("tests") or []
    notable = [t for t in tests if t.get("result") == "FAILED"
               or any(dd.get("dangerous") for dd in (t.get("defects") or []))]
    for t in notable[-3:]:
        defs = "; ".join(d.get("text", "") for d in (t.get("defects") or [])[:3])
        lines.append(f"    {t.get('date','?')} {t.get('result','')}"
                     f" @ {t.get('mileage') or '?'}: {defs[:160]}")
    lines.append(f"    listing: {v.get('url','')}")
    return "\n".join(lines)


# --- LLM plumbing -----------------------------------------------------------

class ReviewError(RuntimeError):
    """The AI review can't produce a real (model-judged) result."""


def _headers(api_key):
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def list_models(base_url, api_key=None, timeout=15):
    """Chat-model ids the endpoint serves (OpenAI-compatible GET /models)."""
    resp = requests.get(base_url.rstrip("/") + "/models", headers=_headers(api_key),
                        timeout=timeout)
    resp.raise_for_status()
    return sorted(m["id"] for m in (resp.json().get("data") or [])
                  if m.get("id") and not _NON_CHAT.search(m["id"]))


def is_local(base_url):
    return bool(re.match(r"https?://(localhost|127\.0\.0\.1|\[::1\])(:|/|$)",
                         base_url or "", re.I))


def unload_model(base_url, model, log=None):
    """Free the model's GPU memory now rather than when Ollama's ~5-minute idle
    timer expires (the GPU is shared with other work). Uses Ollama's native API;
    only applies to a local server."""
    if not is_local(base_url):
        return
    host = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
    try:
        r = requests.post(host + "/api/generate",
                          json={"model": model, "keep_alive": 0}, timeout=30)
        if r.status_code == 200 and log:
            log(f"[Review] Unloaded {model} from GPU memory.")
    except Exception:
        pass


def _retry_wait(resp, attempt):
    """Seconds to wait before retrying a rate-limited or failed call."""
    try:
        if resp.headers.get("retry-after"):
            return min(60.0, float(resp.headers["retry-after"]) + 0.5)
    except ValueError:
        pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)", resp.text or "")
    if m:
        secs = int(m.group(1) or 0) * 60 + float(m.group(2)) / (
            1000 if m.group(3) == "ms" else 1)
        return min(60.0, secs + 0.5)
    return min(60.0, 2.0 * 2 ** attempt)


def llm_chat(base_url, model, messages, api_key=None, temperature=0.2, timeout=900):
    """One completion. Raises on an HTTP error or an empty answer - e.g. a
    "thinking" model that spent its whole token budget reasoning and never
    replied - so a failed call is never mistaken for a judgement.

    Rate limits (HTTP 429, e.g. Groq's tokens-per-minute cap) and server errors
    (5xx, e.g. a local runner that crashed) are retried with a wait - honouring
    the server's retry-after when it gives one."""
    for attempt in range(5):
        resp = requests.post(base_url.rstrip("/") + "/chat/completions",
                             headers=_headers(api_key), timeout=timeout, json={
                                 "model": model, "temperature": temperature,
                                 "stream": False, "max_tokens": 4096,
                                 "messages": messages})
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt < 4:
                time.sleep(_retry_wait(resp, attempt))
                continue
        break
    if resp.status_code != 200:
        raise RuntimeError(f"LLM error {resp.status_code}: {resp.text[:200]}")
    choice = (resp.json().get("choices") or [{}])[0]
    content = ((choice.get("message") or {}).get("content") or "").strip()
    if not content:
        why = ("it used its whole token budget reasoning without answering - a "
               "'thinking' model; pick a non-thinking one such as qwen2.5"
               if choice.get("finish_reason") == "length" else "empty reply")
        raise RuntimeError(f"model gave no answer ({why})")
    return content


def _extract_json(text):
    """Pull the outermost JSON value out of a model reply (tolerates ``` fences,
    <think> blocks and prose around it). Uses whichever of '{' or '[' appears
    first, so an object containing arrays isn't mistaken for its inner array."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    starts = sorted((text.find(c), c, close)
                    for c, close in (("[", "]"), ("{", "}")) if text.find(c) >= 0)
    for start, open_c, close_c in starts:
        depth = 0
        for j in range(start, len(text)):
            if text[j] == open_c:
                depth += 1
            elif text[j] == close_c:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:j + 1])
                    except Exception:
                        break
    return None


# --- Swiss-system tournament (no elimination) -------------------------------

_RANK_SYS = ("You are a shrewd UK used-vehicle buyer's analyst. You judge each "
             "vehicle overall - price, age, mileage, mechanical condition from the "
             "MOT record (failures, dangerous defects, recurring advisories, "
             "mileage consistency) and distance - and you decide for yourself how "
             "much price matters versus condition. If the buyer states priorities, "
             "weight your judgement toward them. Reading the MOT record: a fail "
             "fixed on a retest within weeks is routine; a LATEST test that failed, "
             "dangerous defects in the last few years, recurring corrosion "
             "(structural rust is costly) and mileage going backwards are serious; "
             "the latest pass's advisory count shows current condition. Compare the "
             "vehicles against each other rather than judging each in isolation.")


def _brief_clause(brief):
    brief = (brief or "").strip()
    return f"\n\nThe buyer's priorities (weight these): {brief}\n" if brief else "\n"


def _heuristic_score(v):
    """A transparent value score used only to seed the first round and break ties."""
    d = derive(v)
    s = 60.0
    price, miles, year = _price_int(v), d["miles"], d["year"]
    if price:
        s += max(-15, min(15, (5000 - price) / 400))
    if miles:
        s += max(-20, min(15, (120000 - miles) / 8000))
    if year:
        s += max(-10, min(12, (year - 2012) * 2))
    s -= (6 * d["dangerous_recent"] + 2 * (d["dangerous"] - d["dangerous_recent"])
          + 4 * (d["fails_3y"] - d["retested"]) + (15 if d["unresolved"] else 0)
          + (15 if d["clocking"] else 0))
    s -= 3 * len(d["recurring"])
    dm = v.get("distanceMiles")
    if isinstance(dm, (int, float)):
        s -= min(12, dm / 20)
    return round(max(1, min(100, s)), 1)


def _as_id(x):
    """An id from a model reply: 3, "3", "#3" or "id 3"."""
    if isinstance(x, (int, float)):
        return int(x)
    m = re.search(r"\d+", str(x))
    return int(m.group(0)) if m else None


def _llm_rank(chat, band, log, brief="", rng=random):
    """Have the model order one peer group best->worst.

    Returns the ids in that order, or None if the reply isn't a complete ranking
    of exactly this group - a failed call must not be mistaken for a judgement.
    The group is shown in shuffled order so the model's position bias can't
    simply echo the current standings.
    """
    ids = [i for i, _ in band]
    shown = list(band)
    rng.shuffle(shown)
    dossiers = "\n".join(compact_dossier(i, v) for i, v in shown)
    prompt = (f"Rank these {len(band)} used vehicles from best buy to worst, all "
              f"things considered - weigh price, value, MOT/mechanical condition "
              f"and distance however you judge best.{_brief_clause(brief)}\n"
              f"{dossiers}\n\n"
              f"Reply ONLY with a JSON array of the {len(band)} id numbers (the "
              f"number after #), best first. Include every id exactly once.")
    try:
        data = _extract_json(chat([{"role": "system", "content": _RANK_SYS},
                                   {"role": "user", "content": prompt}]))
    except Exception as exc:
        log(f"[Review] Ranking call failed: {exc}")
        return None
    order = []
    for x in (data if isinstance(data, list) else []):
        xi = _as_id(x)
        if xi in ids and xi not in order:
            order.append(xi)
    if sorted(order) != sorted(ids):
        log(f"[Review] Model's ranking didn't cover the group ({order} vs {ids}); "
            f"ignoring this comparison.")
        return None
    return order


def _bands(order, size, offset):
    """Consecutive peer groups of `size`, shifted by `offset` so band edges move
    between rounds and borderline vehicles meet the rivals they just missed."""
    seq = list(order)
    if offset and len(seq) > offset:
        yield seq[:offset]
        seq = seq[offset:]
    for k in range(0, len(seq), size):
        yield seq[k:k + size]


_ELO_K = 16          # rating step per pairwise comparison


def _swiss(indexed, chat, batch_size, rounds, log, brief=""):
    """Elo ratings from within-group pairwise comparisons - no elimination.

    Each round groups vehicles of similar rating, the model ranks the group, and
    every pair in that ranking updates Elo (earlier beats later). Because an Elo
    win over a low-rated peer barely moves the needle, a vehicle can't reach the
    top just by topping a weak group - it has to out-rank genuinely strong rivals,
    which Swiss pairing keeps feeding it. Ratings seed from the value heuristic in
    round one only (all start equal, tie-broken by heuristic)."""
    by_id = dict(indexed)
    heur = {i: _heuristic_score(v) for i, v in indexed}
    rating = {i: 1000.0 for i, _ in indexed}
    stats = {"ok": 0, "failed": 0}
    rng = random.Random(1234)              # reproducible shuffles
    for r in range(rounds):
        order = sorted(rating, key=lambda i: (rating[i], heur[i]), reverse=True)
        offset = (batch_size // 2) if (r % 2) else 0
        bands = [b for b in _bands(order, batch_size, offset) if len(b) >= 2]
        log(f"[Review] Round {r + 1}/{rounds}: comparing {len(bands)} peer group(s)...")
        for band in bands:
            group = [(i, by_id[i]) for i in band]
            ranked = _llm_rank(chat, group, log, brief, rng)
            if ranked is None:                  # one retry, shown in a new order
                ranked = _llm_rank(chat, group, log, brief, rng)
            if ranked is None:                  # no judgement -> no rating change
                stats["failed"] += 1
                if stats["ok"] == 0 and stats["failed"] >= 2:
                    log("[Review] The model isn't producing usable rankings; "
                        "stopping early.")
                    return sorted(rating, key=lambda i: heur[i], reverse=True), \
                        rating, stats
                continue
            stats["ok"] += 1
            for a in range(len(ranked)):        # best -> worst: earlier beats later
                for b in range(a + 1, len(ranked)):
                    wi, li = ranked[a], ranked[b]
                    exp = 1.0 / (1.0 + 10 ** ((rating[li] - rating[wi]) / 400.0))
                    rating[wi] += _ELO_K * (1 - exp)
                    rating[li] -= _ELO_K * (1 - exp)
    final = sorted(rating, key=lambda i: (rating[i], heur[i]), reverse=True)
    return final, rating, stats


def _review_one(chat, v, log, brief=""):
    """A verdict and pros/cons for one vehicle (small, reliable prompt)."""
    prompt = ("Assess this used vehicle for a buyer in one short line, then give its "
              "pros and cons - each under 12 words, concrete about price, mileage, "
              "year and MOT findings. Use ONLY the facts below - don't assume the "
              f"engine, trim, equipment or anything else not stated."
              f"{_brief_clause(brief)}\n{full_dossier(0, v)}\n\n"
              'Reply ONLY as JSON: {"verdict": "<one sentence>", "pros": ["..."], '
              '"cons": ["..."]} with 2-4 of each.')
    try:
        data = _extract_json(chat([{"role": "system", "content": _RANK_SYS},
                                   {"role": "user", "content": prompt}]))
    except Exception as exc:
        log(f"[Review] Verdict call failed for {v.get('plate')}: {exc}")
        data = None
    if (isinstance(data, dict) and str(data.get("verdict", "")).strip()
            and isinstance(data.get("pros"), list) and isinstance(data.get("cons"), list)):
        return (str(data["verdict"]).strip(), data["pros"], data["cons"], True)
    # No usable AI verdict: show plain facts, clearly labelled as such.
    return ("", _auto_pros(v), _auto_cons(v), False)


def _auto_pros(v):
    d = derive(v)
    p = []
    if d["per_year"] and d["per_year"] < 12000:
        p.append(f"Low use (~{d['per_year']:,}/yr)")
    if v.get("fails", 0) == 0:
        p.append("No MOT failures on record")
    if not d["recurring"]:
        p.append("No recurring advisory theme")
    return p or ["Verified against DVSA"]


def _auto_cons(v):
    d = derive(v)
    c = []
    if d["unresolved"]:
        c.append("Failed its latest MOT")
    if d["fails_3y"] - d["retested"] > 0:
        c.append(f"{d['fails_3y'] - d['retested']} MOT fail(s) in 3 years not fixed "
                 f"on a quick retest")
    if d["dangerous_recent"]:
        c.append(f"{d['dangerous_recent']} dangerous defect(s) in the last 3 years")
    if d["clocking"]:
        c.append("Odometer inconsistency")
    if d["recurring"]:
        c.append("Recurring: " + ", ".join(d["recurring"]))
    return c or ["Check in person"]


def _shortlist_row(rank, v, verdict, pros, cons, ai, best):
    d = derive(v)
    return {"rank": rank, "plate": v.get("plate"), "price": v.get("price"),
            "location": v.get("location"), "url": v.get("url"), "site": v.get("site"),
            "year": d["year"], "mileage": v.get("latestMileage"),
            "make": v.get("make"), "model": v.get("model"), "best": bool(best),
            "ai": bool(ai), "also": v.get("also") or [],
            "verdict": str(verdict)[:300],
            "pros": [str(p)[:160] for p in (pros or [])][:5],
            "cons": [str(c)[:160] for c in (cons or [])][:5]}


def _dedupe(vehicles, log):
    """One entry per registration: the same vehicle listed twice (or on both
    sites) is reviewed once, keeping the cheapest listing and noting the rest."""
    keep, order = {}, []
    for v in vehicles:
        key = re.sub(r"[^A-Z0-9]", "", (v.get("plate") or "").upper())
        if not key:
            order.append(dict(v))
            continue
        cur = keep.get(key)
        if cur is None:
            keep[key] = dict(v, also=[])
            order.append(key)
            continue
        cheaper = (_price_int(v) or 10 ** 9) < (_price_int(cur) or 10 ** 9)
        loser, winner = (cur, dict(v, also=list(cur["also"]))) if cheaper else (v, cur)
        winner["also"].append(loser.get("url"))
        keep[key] = winner
    merged = len(vehicles) - len(order)
    if merged:
        log(f"[Review] Merged {merged} duplicate listing(s) of the same vehicle "
            f"(kept the cheapest).")
    return [keep[k] if isinstance(k, str) else k for k in order]


def _leader_note(v):
    d = derive(v)
    bits = []
    if d["unresolved"]:
        bits.append("LATEST MOT FAILED")
    elif v.get("fails", 0) == 0:
        bits.append("clean MOT")
    if d["fails_3y"]:
        bits.append(f"{d['fails_3y']} fail(s) in 3y ({d['retested']} fixed on retest)")
    if d["dangerous_recent"]:
        bits.append(f"{d['dangerous_recent']} recent dangerous")
    elif d["dangerous"]:
        bits.append(f"{d['dangerous']} old dangerous")
    if d["clocking"]:
        bits.append("mileage anomaly")
    if d["recurring"]:
        bits.append("recurring " + "/".join(d["recurring"][:2]))
    return ", ".join(bits) or "verified"


def run_tournament(vehicles, base_url, model, log, api_key=None,
                   batch_size=BATCH_SIZE, rounds=ROUNDS, shortlist=10, brief=""):
    """Swiss-system review of EVERY vehicle. `brief` is the buyer's free-text
    priorities (empty = the model weighs price and value as it judges best).

    Raises ReviewError rather than return a result the model didn't actually
    judge (model not installed, server down, or no usable rankings).
    Returns {"shortlist": [...], "leaderboard": [...all...], "stats": {...}}.
    """
    where = "Ollama" if is_local(base_url) else base_url
    try:
        installed = list_models(base_url, api_key)
    except Exception as exc:
        raise ReviewError(f"Can't reach the model server at {base_url} ({exc}). "
                          + ("Is Ollama running?" if is_local(base_url)
                             else "Check the API key and your connection."))
    if model not in installed:
        raise ReviewError(f"Model '{model}' isn't available on {where}. "
                          f"Available: {', '.join(installed) or 'none'}.")
    try:
        # One tiny call first, so a model that's blocked, broken or only ever
        # "thinks" fails here with its real reason instead of mid-tournament.
        try:
            llm_chat(base_url, model, [{"role": "user",
                                        "content": "Reply with just the word OK."}],
                     api_key=api_key)
        except Exception as exc:
            raise ReviewError(f"{model} can't be used right now: {exc}")
        return _tournament(vehicles, base_url, model, log, api_key, batch_size,
                           rounds, shortlist, brief)
    finally:
        unload_model(base_url, model, log)     # free the GPU as soon as we're done


def _tournament(vehicles, base_url, model, log, api_key, batch_size, rounds,
                shortlist, brief):
    def chat(messages):
        return llm_chat(base_url, model, messages, api_key=api_key)

    indexed = list(enumerate(_dedupe(vehicles, log), start=1))
    if not indexed:
        return {"shortlist": [], "leaderboard": [], "stats": {}}
    by_id = dict(indexed)
    if len(indexed) == 1:                  # nothing to compare against
        final, rating, stats = [1], {1: 1000.0}, {"ok": 0, "failed": 0}
    else:
        final, rating, stats = _swiss(indexed, chat, batch_size, rounds, log, brief)
        if stats["ok"] == 0:
            raise ReviewError(
                f"{model} never returned a usable ranking ({stats['failed']} "
                f"attempt(s) failed - see the diagnostics feed), so there is no AI "
                f"judgement to show. Try a non-thinking model such as qwen2.5:32b.")
        if stats["failed"]:
            log(f"[Review] Note: {stats['failed']} of {stats['ok'] + stats['failed']} "
                f"comparisons failed and were left out of the ratings.")

    leaderboard = []
    for rank, i in enumerate(final, start=1):
        v = by_id[i]
        d = derive(v)
        leaderboard.append({"rank": rank, "plate": v.get("plate"),
            "price": v.get("price"), "year": d["year"],
            "mileage": v.get("latestMileage"), "location": v.get("location"),
            "url": v.get("url"), "site": v.get("site"),
            "rating": round(rating[i]), "note": _leader_note(v)})

    top = final[:shortlist]
    log(f"[Review] Writing pros & cons for the top {len(top)}...")
    rows = []
    for rank, i in enumerate(top, start=1):
        rows.append(_shortlist_row(rank, by_id[i],
                    *_review_one(chat, by_id[i], log, brief), best=(rank == 1)))
    stats["verdicts_ai"] = sum(1 for r in rows if r["ai"])
    stats["verdicts"] = len(rows)
    log(f"[Review] Shortlist ready ({stats['ok']} AI comparisons, "
        f"{stats['verdicts_ai']}/{len(rows)} AI-written verdicts).")
    return {"shortlist": rows, "leaderboard": leaderboard, "stats": stats}
