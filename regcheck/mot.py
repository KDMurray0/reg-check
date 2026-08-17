"""DVSA MOT History API client and record shaping.

MOT history comes from the official (free) DVSA MOT History API. The public
GOV.UK web checker sits behind Imperva/Incapsula bot protection that blocks
automation, so the sanctioned OAuth2 API is used instead. Register (free) at the
DVSA MOT History portal for a client id / secret / API key / token URL.

`shape_vehicle` turns a raw API record into the compact dict the web UI renders
(summary fields + a chronological, structured MOT test list); `format_mot_history`
renders the same record as the indented plain-text block written to the results
file. `vehicle_tier` scores how confident we are that a looked-up vehicle is the
listed car.
"""

from __future__ import annotations

import os
import re
import time

MOT_HISTORY_URL = "https://history.mot.api.gov.uk/v1/trade/vehicles/registration/{reg}"
MOT_DEFAULT_SCOPE = os.environ.get("MOT_SCOPE", "https://tapi.dvsa.gov.uk/.default")


class MOTClient:
    """Client for the DVSA MOT History API with access-token caching."""

    def __init__(self, client_id, client_secret, api_key, token_url, scope=None):
        self.client_id = (client_id or "").strip()
        self.client_secret = (client_secret or "").strip()
        self.api_key = (api_key or "").strip()
        self.token_url = (token_url or "").strip()
        self.scope = (scope or MOT_DEFAULT_SCOPE).strip()
        self._token = None
        self._token_expiry = 0.0

    @property
    def configured(self) -> bool:
        return all([self.client_id, self.client_secret,
                    self.api_key, self.token_url])

    def _ensure_token(self, requests_mod):
        if self._token and time.time() < self._token_expiry - 60:
            return
        resp = requests_mod.post(
            self.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": self.scope,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"token request failed ({resp.status_code}): "
                               f"{resp.text[:200]}")
        payload = resp.json()
        self._token = payload["access_token"]
        self._token_expiry = time.time() + float(payload.get("expires_in", 3600))

    def lookup(self, registration, requests_mod):
        """Return the vehicle record dict, or None if the plate is unknown (404).

        Raises on auth/other errors so the caller can surface them rather than
        silently treating every plate as 'not found'.
        """
        self._ensure_token(requests_mod)
        reg = registration.replace(" ", "").upper()
        resp = requests_mod.get(
            MOT_HISTORY_URL.format(reg=reg),
            headers={
                "Authorization": f"Bearer {self._token}",
                "X-API-Key": self.api_key,
                "Accept": "application/json",
            },
            timeout=30,
        )
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise RuntimeError(f"MOT API error ({resp.status_code}): "
                               f"{resp.text[:200]}")
        data = resp.json()
        return data[0] if isinstance(data, list) else data


# --- Make / model matching --------------------------------------------------

def make_matches(listing_make: str, api_make: str) -> bool:
    """True if the API make is consistent with the listing make.

    Unknown on either side is not treated as a mismatch (can't disprove it).
    """
    if not listing_make or not api_make:
        return True
    token = re.sub(r"[^A-Z]", "", listing_make.upper().split()[0])
    return bool(token) and token in api_make.upper()


def model_matches(listing_model: str, api_model: str):
    """Tri-state model check: True (overlap), False (no overlap), None (unknown).

    Lenient token overlap so trim/naming differences ('NP300 Navara' vs
    'NAVARA') still match on the shared word.
    """
    if not listing_model or not api_model:
        return None
    lt = set(re.findall(r"[A-Z0-9]{3,}", listing_model.upper()))
    at = set(re.findall(r"[A-Z0-9]{3,}", api_model.upper()))
    return bool(lt & at)


def vehicle_tier(listing_make, listing_model, vehicle) -> int:
    """Confidence that an API vehicle record is the listed car.

    3 = make and model both match   (strongest)
    2 = make matches, model unknown
    1 = make matches, model does NOT match
    0 = make does not match          (reject - almost certainly a coincidence)
    """
    if not make_matches(listing_make, vehicle.get("make", "")):
        return 0
    mm = model_matches(listing_model, vehicle.get("model", ""))
    if mm is True:
        return 3
    if mm is None:
        return 2
    return 1


# --- Record shaping ---------------------------------------------------------

def _tests_sorted(vehicle):
    tests = vehicle.get("motTests") or []
    return sorted(tests, key=lambda t: t.get("completedDate", ""))


def _defects(test):
    out = []
    for d in (test.get("defects") or test.get("rfrAndComments") or []):
        out.append({
            "type": (d.get("type") or "").upper(),
            "text": (d.get("text") or "").strip(),
            "dangerous": bool(d.get("dangerous")),
        })
    return out


def _mileage(test):
    odo = test.get("odometerValue")
    unit = (test.get("odometerUnit") or "mi").lower()
    unit = "mi" if unit.startswith("mi") else unit
    try:
        return f"{int(odo):,} {unit}"
    except (TypeError, ValueError):
        return None


def shape_vehicle(vehicle) -> dict:
    """Compact, JSON-friendly view of an API record for the web UI."""
    if not vehicle:
        return {}
    tests = _tests_sorted(vehicle)
    shaped_tests = []
    for t in tests:
        shaped_tests.append({
            "date": (t.get("completedDate") or "")[:10],
            "result": (t.get("testResult") or "").upper(),
            "mileage": _mileage(t),
            "expiry": (t.get("expiryDate") or "")[:10],
            "defects": _defects(t),
        })
    passes = sum(1 for t in shaped_tests if t["result"] == "PASSED")
    fails = sum(1 for t in shaped_tests if t["result"] == "FAILED")
    latest = shaped_tests[-1] if shaped_tests else None
    return {
        "make": vehicle.get("make", ""),
        "model": vehicle.get("model", ""),
        "colour": vehicle.get("primaryColour", ""),
        "fuel": vehicle.get("fuelType", ""),
        "firstUsed": (vehicle.get("firstUsedDate") or "")[:10],
        "motExpiry": (vehicle.get("motTestExpiryDate")
                      or (latest.get("expiry") if latest else "") or ""),
        "latestMileage": latest.get("mileage") if latest else None,
        "latestResult": latest.get("result") if latest else None,
        "passes": passes,
        "fails": fails,
        "tests": shaped_tests,
    }


def format_mot_history(vehicle) -> str:
    """Indented chronological history block for the results text file."""
    if not vehicle:
        return "   (No MOT record returned.)"
    make = vehicle.get("make", "")
    model = vehicle.get("model", "")
    summary = [f"   Vehicle: {make} {model}".rstrip()]
    extra = []
    for label, key in (("Colour", "primaryColour"), ("Fuel", "fuelType"),
                       ("First used", "firstUsedDate")):
        if vehicle.get(key):
            extra.append(f"{label}: {vehicle[key]}")
    if extra:
        summary.append("   " + " | ".join(extra))

    tests = _tests_sorted(vehicle)
    if not tests:
        return "\n".join(summary + ["   (No MOT tests on record.)"])

    lines = list(summary)
    for t in tests:
        date = (t.get("completedDate") or "?")[:10]
        result = t.get("testResult", "?")
        odo = _mileage(t) or "? mi"
        lines.append(f"   {date} | {result} | {odo}")
        for d in _defects(t):
            danger = " [DANGEROUS]" if d["dangerous"] else ""
            lines.append(f"       - {d['type']}: {d['text']}{danger}")
    return "\n".join(lines)
