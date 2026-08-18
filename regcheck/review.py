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
import re

import requests

DEFAULT_BASE_URL = "http://localhost:11434/v1"   # Ollama's OpenAI-compatible API
DEFAULT_MODEL = "qwen3.6:27b"
BATCH_SIZE = 6          # vehicles compared per peer group (small = reliable ranking)
ROUNDS = 3              # Swiss rounds - every vehicle is ranked in each
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


def derive(v: dict) -> dict:
    """Buyer-relevant signals computed from a verified vehicle record."""
    tests = v.get("tests") or []
    year = _year(v)
    miles = _int_miles(v.get("latestMileage"))
    age = (_NOW_YEAR - year) if year else None
    per_year = int(miles / age) if (miles and age and age > 0) else None

    # Odometer going backwards between tests = possible clock / anomaly.
    seq = [_int_miles(t.get("mileage")) for t in tests]
    seq = [m for m in seq if m is not None]
    clocking = any(b < a - 1000 for a, b in zip(seq, seq[1:]))

    recent_fail = any(t.get("result") == "FAILED"
                      and t.get("date", "")[:4].isdigit()
                      and _NOW_YEAR - int(t["date"][:4]) <= 3 for t in tests)
    dangerous = sum(1 for t in tests for d in (t.get("defects") or []) if d.get("dangerous"))

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
            "clocking": clocking, "recent_fail": recent_fail,
            "dangerous": dangerous, "recurring": recurring}


def _price_int(v):
    return _int_miles(v.get("price"))


def compact_dossier(i: int, v: dict) -> str:
    d = derive(v)
    bits = [f"#{i} {v.get('plate','?')}"]
    bits.append(f"{d['year'] or '?'} {v.get('make','')} {v.get('model','')}".strip())
    bits.append(v.get("price") or "price n/a")
    bits.append(f"{d['miles']:,} mi".replace(",", ",") if d["miles"] else "mileage n/a")
    if d["per_year"]:
        bits.append(f"~{d['per_year']:,}/yr")
    bits.append(v.get("location") or "location n/a")
    mot = f"MOT {v.get('passes',0)}P/{v.get('fails',0)}F"
    if v.get("motExpiry"):
        mot += f", to {v['motExpiry']}"
    bits.append(mot)
    flags = []
    if d["dangerous"]:
        flags.append(f"{d['dangerous']} dangerous")
    if d["recent_fail"]:
        flags.append("recent fail")
    if d["clocking"]:
        flags.append("mileage anomaly")
    if d["recurring"]:
        flags.append("recurring: " + ", ".join(d["recurring"]))
    if flags:
        bits.append("; ".join(flags))
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

def llm_chat(base_url, model, messages, api_key=None, temperature=0.2, timeout=900):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(url, headers=headers, timeout=timeout, json={
        "model": model, "temperature": temperature, "stream": False,
        "max_tokens": 1500, "messages": messages})
    if resp.status_code != 200:
        raise RuntimeError(f"LLM error {resp.status_code}: {resp.text[:200]}")
    return resp.json()["choices"][0]["message"]["content"]


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
             "weight your judgement toward them.")


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
    s -= 8 * d["dangerous"] + (10 if d["recent_fail"] else 0) + (15 if d["clocking"] else 0)
    s -= 3 * len(d["recurring"])
    dm = v.get("distanceMiles")
    if isinstance(dm, (int, float)):
        s -= min(12, dm / 20)
    return round(max(1, min(100, s)), 1)


def _llm_rank(chat, band, log, brief=""):
    """Order one peer group best->worst; return ids in that order. On any parse
    failure the band keeps its incoming order, so a vehicle is never dropped."""
    ids = [i for i, _ in band]
    if len(band) < 2:
        return ids
    dossiers = "\n".join(compact_dossier(i, v) for i, v in band)
    prompt = (f"Rank these {len(band)} used vehicles from best buy to worst, all "
              f"things considered - weigh price, value, MOT/mechanical condition "
              f"and distance however you judge best.{_brief_clause(brief)}\n"
              f"{dossiers}\n\n"
              f"Reply ONLY with a JSON array of the id numbers, best first "
              f"(e.g. [{ids[1]},{ids[0]}]). Include every id exactly once.")
    try:
        data = _extract_json(chat([{"role": "system", "content": _RANK_SYS},
                                   {"role": "user", "content": prompt}]))
    except Exception as exc:
        log(f"[Review] Ranking error: {exc}; keeping current order.")
        data = None
    order = []
    if isinstance(data, list):
        for x in data:
            try:
                xi = int(x)
            except (TypeError, ValueError):
                continue
            if xi in ids and xi not in order:
                order.append(xi)
    for i in ids:                      # append anything the model missed - never drop
        if i not in order:
            order.append(i)
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
    for r in range(rounds):
        order = sorted(rating, key=lambda i: (rating[i], heur[i]), reverse=True)
        offset = (batch_size // 2) if (r % 2) else 0
        bands = [b for b in _bands(order, batch_size, offset) if len(b) >= 2]
        log(f"[Review] Round {r + 1}/{rounds}: comparing {len(bands)} peer group(s)...")
        for band in bands:
            ranked = _llm_rank(chat, [(i, by_id[i]) for i in band], log, brief)
            for a in range(len(ranked)):        # best -> worst: earlier beats later
                for b in range(a + 1, len(ranked)):
                    wi, li = ranked[a], ranked[b]
                    exp = 1.0 / (1.0 + 10 ** ((rating[li] - rating[wi]) / 400.0))
                    rating[wi] += _ELO_K * (1 - exp)
                    rating[li] -= _ELO_K * (1 - exp)
    final = sorted(rating, key=lambda i: (rating[i], heur[i]), reverse=True)
    return final, rating


def _review_one(chat, v, log, brief=""):
    """A verdict and pros/cons for one vehicle (small, reliable prompt)."""
    prompt = ("Assess this used vehicle for a buyer in one short line, then give its "
              "pros and cons - each under 12 words, concrete about price, mileage, "
              f"year and MOT findings.{_brief_clause(brief)}\n{full_dossier(0, v)}\n\n"
              'Reply ONLY as JSON: {"verdict": "<one sentence>", "pros": ["..."], '
              '"cons": ["..."]} with 2-4 of each.')
    try:
        data = _extract_json(chat([{"role": "system", "content": _RANK_SYS},
                                   {"role": "user", "content": prompt}]))
    except Exception as exc:
        log(f"[Review] Verdict error: {exc}; using heuristic notes.")
        data = None
    if isinstance(data, dict):
        return (str(data.get("verdict", "")).strip(),
                data.get("pros") or _auto_pros(v),
                data.get("cons") or _auto_cons(v))
    return ("", _auto_pros(v), _auto_cons(v))


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
    if d["recent_fail"]:
        c.append("Failed an MOT in the last 3 years")
    if d["dangerous"]:
        c.append(f"{d['dangerous']} dangerous defect(s) in history")
    if d["clocking"]:
        c.append("Odometer inconsistency")
    if d["recurring"]:
        c.append("Recurring: " + ", ".join(d["recurring"]))
    return c or ["Check in person"]


def _shortlist_row(rank, v, verdict, pros, cons, best):
    d = derive(v)
    return {"rank": rank, "plate": v.get("plate"), "price": v.get("price"),
            "location": v.get("location"), "url": v.get("url"), "site": v.get("site"),
            "year": d["year"], "mileage": v.get("latestMileage"),
            "make": v.get("make"), "model": v.get("model"), "best": bool(best),
            "verdict": str(verdict)[:300],
            "pros": [str(p)[:160] for p in (pros or [])][:5],
            "cons": [str(c)[:160] for c in (cons or [])][:5]}


def _leader_note(v):
    d = derive(v)
    bits = []
    if v.get("fails", 0) == 0:
        bits.append("clean MOT")
    if d["recent_fail"]:
        bits.append("recent fail")
    if d["dangerous"]:
        bits.append(f"{d['dangerous']} dangerous")
    if d["clocking"]:
        bits.append("mileage anomaly")
    if d["recurring"]:
        bits.append("recurring " + "/".join(d["recurring"][:2]))
    return ", ".join(bits) or "verified"


def run_tournament(vehicles, base_url, model, log, api_key=None,
                   batch_size=BATCH_SIZE, rounds=ROUNDS, shortlist=10, brief=""):
    """Swiss-system review of EVERY vehicle. `brief` is the buyer's free-text
    priorities (empty = the model weighs price and value as it judges best).
    Returns {"shortlist": [...detailed...], "leaderboard": [...all, ranked...]}."""
    def chat(messages):
        return llm_chat(base_url, model, messages, api_key=api_key)

    indexed = list(enumerate(vehicles, start=1))
    if not indexed:
        return {"shortlist": [], "leaderboard": []}
    by_id = dict(indexed)
    if len(indexed) <= batch_size:
        rounds = 1                         # one comparison already sees them all
    final, rating = _swiss(indexed, chat, batch_size, rounds, log, brief)

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
    log("[Review] Shortlist ready.")
    return {"shortlist": rows, "leaderboard": leaderboard}
