"""The inspection pipeline: scrape -> read plate -> verify -> emit result.

`Pipeline` runs off the request thread. It emits structured events through an
`emit(dict)` callback so the web layer can stream them to the browser as a live
log, a progress bar, and result cards. It also appends a plain-text record to the
results file for offline reference.

Heavy libraries (playwright, cv2, numpy, the ANPR/easyocr models) are imported
lazily inside `run()` so the web server starts instantly and reports a clear
error if a dependency is missing.
"""

from __future__ import annotations

import os
import random
import time
from collections import Counter

from . import scrape
from .mot import (MOTClient, format_mot_history, shape_vehicle, vehicle_tier)
from .plates import (_correct_plate, consensus_candidates, contains_dealer_text,
                     decode_image, plate_near_misses, read_plate_reads)

# ANPR: a licence-plate detector localises each plate, then an ensemble of
# plate-specialised recognisers (different architectures make different mistakes)
# reads the crop - far more accurate than general OCR.
ANPR_DETECTOR_MODEL = os.environ.get("REGCHECK_DETECTOR",
                                     "yolo-v9-t-640-license-plate-end2end")
# An ensemble of differently-built recognisers: they make different mistakes, so
# the character-level consensus vote across them lands on the truth more often.
ANPR_OCR_MODELS = ["european-plates-mobile-vit-v2-model", "cct-s-v2-global-model",
                   "global-plates-mobile-vit-v2-model"]

MAX_CANDIDATE_LOOKUPS = 12    # distinct guesses to verify before giving up
MAX_NEAR_MISS_LOOKUPS = 40    # single-char variants tried when nothing verifies
DEALER_MIN_HITS = 2           # images reading dealer text -> classed dealer plate
BACKOFF_MIN, BACKOFF_MAX = 2.0, 5.0   # polite delay between listings (not evasion)

DEFAULT_OPTIONS = {
    "max_images": 30,         # photos read per listing (plate shots first)
    "early_stop_votes": 4,    # stop once a plate is seen in this many photos (0=off)
    "min_images": 10,         # ...but never before reading this many
    "near_miss": True,        # try single-char API variants when nothing verifies
}


# Preferred ONNX Runtime GPU providers, best first. DirectML covers AMD (e.g. a
# 7900 XTX) and Intel GPUs on Windows; CUDA covers NVIDIA (and AMD via ZLUDA);
# ROCm covers AMD on Linux. Install the matching onnxruntime build to enable one
# (e.g. `pip install onnxruntime-directml` for AMD/Windows) - the ANPR detector
# and recognisers then run on the GPU with no code change.
GPU_ONNX_PROVIDERS = ("DmlExecutionProvider", "CUDAExecutionProvider",
                      "ROCMExecutionProvider")


def _onnx_providers(log):
    """Return the ONNX providers to use: the best available GPU one + CPU."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except Exception:
        return None
    for p in GPU_ONNX_PROVIDERS:
        if p in available:
            log(f"[Init] GPU acceleration: ONNX {p}")
            return [p, "CPUExecutionProvider"]
    log("[Init] ONNX on CPU (install onnxruntime-directml / -gpu for GPU).")
    return ["CPUExecutionProvider"]


def load_plate_engines(log):
    """Load the plate-reading engines. ANPR is primary; easyocr is an optional
    full-image recall backup. Raises only if neither is available.

    ONNX models use a GPU execution provider automatically when one is available;
    easyocr uses the GPU when torch reports CUDA (including AMD via ZLUDA)."""
    engines = {"plate_ocrs": []}
    providers = _onnx_providers(log)
    try:
        from open_image_models import LicensePlateDetector
        from fast_plate_ocr import LicensePlateRecognizer
        log("[Init] Loading ANPR plate detector...")
        engines["detector"] = LicensePlateDetector(
            detection_model=ANPR_DETECTOR_MODEL, providers=providers)
        for name in ANPR_OCR_MODELS:
            try:
                engines["plate_ocrs"].append(
                    LicensePlateRecognizer(name, providers=providers))
                log(f"[Init] Loaded ANPR recogniser: {name}")
            except Exception as exc:
                log(f"[Init] Could not load ANPR recogniser {name}: {exc}")
        log(f"[Init] ANPR ready ({len(engines['plate_ocrs'])} recogniser(s)).")
    except Exception as exc:
        log(f"[Init] ANPR unavailable ({exc}); using easyocr only.")

    engines["easyocr"] = None
    try:
        import easyocr
        gpu = False
        try:
            import torch
            gpu = bool(torch.cuda.is_available())
        except Exception:
            gpu = False
        log(f"[Init] Loading easyocr backup reader (gpu={gpu}; "
            f"first run downloads weights)...")
        engines["easyocr"] = easyocr.Reader(["en"], gpu=gpu)
        log("[Init] easyocr ready.")
    except Exception as exc:
        log(f"[Init] easyocr unavailable ({exc}); running on ANPR only.")

    if not engines.get("plate_ocrs") and engines.get("easyocr") is None:
        raise RuntimeError("no plate-reading engine available (ANPR + easyocr "
                           "both failed to load)")
    return engines


class Pipeline:
    def __init__(self, urls, mot_client: MOTClient, emit, stop_event,
                 output_file, options=None):
        self.urls = urls
        self.mot = mot_client
        self.emit = emit
        self.stop_event = stop_event
        self.output_file = output_file
        self.opts = {**DEFAULT_OPTIONS, **(options or {})}
        self.n_success = 0
        self.review = []   # (url, location, note) - manual review
        self.dealer = []   # (url, location) - trade/showroom placeholder plates

    # -- event helpers -----------------------------------------------------
    def log(self, text):
        self.emit({"type": "log", "text": text})

    def status(self, text):
        self.emit({"type": "status", "text": text})

    def run(self):
        try:
            self._run()
        except Exception as exc:
            self.log(f"[FATAL] Worker stopped: {exc!r}")
        finally:
            self.emit({"type": "done", "summary": {
                "verified": self.n_success, "dealer": len(self.dealer),
                "review": len(self.review),
                "file": os.path.abspath(self.output_file),
                "stopped": self.stop_event.is_set()}})

    def _run(self):
        try:
            import requests
            import numpy as np
            import cv2
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            self.log(f"[FATAL] Missing dependency: {exc}. Install requirements.txt "
                     f"and run 'playwright install chromium'.")
            return

        if not self.mot.configured:
            self.log("[Init] MOT API not configured - will record the top-voted "
                     "plate per listing WITHOUT DVSA verification or history.")
        try:
            engines = load_plate_engines(self.log)
        except Exception as exc:
            self.log(f"[FATAL] Could not load plate-reading models: {exc!r}")
            return

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0 Safari/537.36"),
                locale="en-GB", viewport={"width": 1440, "height": 900})
            page = context.new_page()

            tasks = self._build_tasks(page)
            total = len(tasks)
            self.log(f"[Init] {total} listing(s) queued." if total
                     else "[DONE] No listings to process.")

            for idx, task in enumerate(tasks, start=1):
                if self.stop_event.is_set():
                    self.log("[STOPPED] Cancelled.")
                    break
                self.emit({"type": "progress", "current": idx, "total": total})
                self.status(f"Inspecting vehicle {idx} of {total}...")
                self.log("=" * 50)
                self.log(f"[Vehicle {idx}/{total}] {task['url']}")
                if idx > 1:
                    delay = random.uniform(BACKOFF_MIN, BACKOFF_MAX)
                    self.log(f"[Wait] Backing off {delay:.1f}s...")
                    time.sleep(delay)
                try:
                    self._process(page, task, engines, cv2, np, requests)
                except Exception as exc:
                    self.log(f"[ERROR] Failed to process listing: {exc!r}")
                    continue

            context.close()
            browser.close()

        self._write_review_sections()
        if not self.stop_event.is_set():
            self.status("Finished.")
            self.log("=" * 50)
            self.log(f"[DONE] {self.n_success} verified, {len(self.dealer)} dealer "
                     f"plate(s), {len(self.review)} to review. "
                     f"Written to {os.path.abspath(self.output_file)}")

    def _build_tasks(self, page):
        """Flatten input lines into per-listing task dicts (search URLs expanded)."""
        tasks = []
        for line in self.urls:
            if self.stop_event.is_set():
                break
            lo, hi = scrape.price_bounds(line)   # the search's own price filter
            if scrape.is_search_url(line):
                self.log("=" * 50)
                self.log("[Search] Expanding search results...")
                make, model = scrape.search_make_model(line)
                try:
                    for r in scrape.collect_search_results(
                            page, line, self.log, self.stop_event):
                        tasks.append({"url": r["url"], "distance": r["distance"],
                                      "location": r["location"], "price": r["price"],
                                      "make": make, "model": model,
                                      "price_lo": lo, "price_hi": hi})
                except Exception as exc:
                    self.log(f"[Search] Failed to expand search: {exc!r}")
            else:
                tasks.append({"url": line, "distance": None, "location": None,
                              "price": None, "make": "", "model": "",
                              "price_lo": lo, "price_hi": hi})
        return tasks

    def _location_display(self, task, d_location, d_distance_miles):
        """Best 'where/distance' string: miles if known, else dealer town."""
        if task.get("distance"):
            return f"{task['distance']} miles", _to_float(task["distance"])
        if d_distance_miles:
            return f"{d_distance_miles} miles", _to_float(d_distance_miles)
        return task.get("location") or d_location or "N/A", None

    def _process(self, page, task, engines, cv2, np, requests):
        url = task["url"]
        site = "cazoo" if scrape.is_cazoo(url) else "autotrader"
        price, d_make, d_model, d_location, d_dist, image_urls = scrape.scrape_listing(
            page, url, self.opts["max_images"], self.log)
        make = task.get("make") or d_make
        model = task.get("model") or d_model
        price = task.get("price") or price     # card price wins on Cazoo
        location, distance_miles = self._location_display(task, d_location, d_dist)

        # Sponsored/promoted listings ignore the site's price filter; enforce the
        # search URL's own price bounds ourselves before spending time reading it.
        pint = scrape.price_to_int(price)
        lo, hi = task.get("price_lo"), task.get("price_hi")
        if pint is not None and ((hi and pint > hi) or (lo and pint < lo)):
            bound = (f"over £{hi:,}" if hi and pint > hi else f"under £{lo:,}")
            self.log(f"[FILTER] {price} is {bound} (search filter); skipping "
                     f"out-of-filter listing.")
            return

        if not image_urls:
            self.log("[SKIPPED] No images found for listing")
            self.review.append((url, location, "no images found"))
            return

        all_reads = []                           # (chars, probs) from every photo
        image_votes: Counter[str] = Counter()    # distinct photos a plate appears in
        dealer_hits = 0
        images_read = 0

        for i, img_url in enumerate(image_urls, start=1):
            if self.stop_event.is_set():
                return
            self.log(f"[Read] Scanning image {i}/{len(image_urls)}...")
            img_bytes = _fetch_image(requests, img_url)
            if not img_bytes:
                continue
            img = decode_image(cv2, np, img_bytes)
            if img is None:
                continue
            images_read += 1
            reads, texts = read_plate_reads(img, engines, cv2, np, enhanced=True)
            all_reads.extend(reads)
            seen = set()
            for chars, _ps in reads:
                corr = _correct_plate(chars)
                if corr:
                    seen.add(corr)
            for plate in seen:
                if plate not in image_votes:
                    self.log(f"[Read] Candidate plate: {plate}")
                image_votes[plate] += 1
            if contains_dealer_text(texts, exclude=(make, model)):
                dealer_hits += 1
            # Plate shots are read first, so a plate seen in several photos is very
            # likely the real one - stop early to save time (unless disabled).
            esv = self.opts["early_stop_votes"]
            if (esv and images_read >= self.opts["min_images"] and image_votes
                    and image_votes.most_common(1)[0][1] >= esv):
                self.log(f"[Read] Strong consensus after {images_read} photo(s); stopping.")
                break

        if not image_votes:
            if dealer_hits >= DEALER_MIN_HITS:
                self.log(f"[DEALER] No real plate; dealer/trade branding in "
                         f"{dealer_hits} image(s)")
                self.dealer.append((url, location))
                self.emit({"type": "review", "category": "dealer", "url": url,
                           "location": location, "site": site, "note": None})
            else:
                self.log("[FAILED] No registration plate could be read")
                self.review.append((url, location, "no plate could be read"))
                self.emit({"type": "review", "category": "manual", "url": url,
                           "location": location, "site": site,
                           "note": "no plate could be read"})
            return

        # Character-level consensus across every read (per-position confidence
        # voting + a beam over ambiguous positions + UK age prior), best first.
        cands = consensus_candidates(all_reads, MAX_CANDIDATE_LOOKUPS, age_prior=True)
        ordered = [(p, image_votes.get(p, 0)) for p, _ in cands]

        base = {"price": price, "location": location, "distanceMiles": distance_miles,
                "url": url, "site": site, "make": make, "model": model}

        if not self.mot.configured:
            plate, votes = ordered[0]
            self.log(f"[UNVERIFIED] Top-voted plate {plate} ({votes} photo(s)); "
                     f"no MOT API configured")
            self._write_record(plate, price, location, url,
                               "   (MOT lookup skipped: API not configured.)")
            self._emit_result({**base, "plate": plate, "verified": False,
                               "corrected": False, "tier": None, "votes": votes})
            self.n_success += 1
            return

        self.log(f"[Verify] {len(ordered)} candidate(s); checking against the MOT "
                 f"API (listing: {make or '?'} {model or ''})".rstrip() + "...")
        chosen = self._verify(ordered, make, model, requests)

        if chosen is None:
            corrected = (self._near_miss(ordered, make, model, requests)
                         if self.opts["near_miss"] else None)
            if corrected is not None:
                plate, vehicle = corrected
                vdesc = _vdesc(vehicle)
                self.log(f"[SUCCESS] {plate} (OCR-corrected) verified as {vdesc}")
                self._write_record(f"{plate} (OCR-corrected)", price, location, url,
                                   format_mot_history(vehicle))
                self._emit_result({**base, "plate": plate, "verified": True,
                                   "corrected": True, "tier": 3, "votes": None,
                                   **shape_vehicle(vehicle)})
                self.n_success += 1
                return
            # Nothing verified. If the plate region read as dealer/trade branding
            # (the ANPR models turn that into junk regs, so this is the tell), it's
            # a dealer plate - no real reg was shown - not a read failure.
            if dealer_hits >= DEALER_MIN_HITS:
                self.log(f"[DEALER] Dealer/trade branding on the plate "
                         f"({dealer_hits} image(s)); no real registration shown")
                self.dealer.append((url, location))
                self.emit({"type": "review", "category": "dealer", "url": url,
                           "location": location, "site": site, "note": None})
                return
            shortlist = ", ".join(f"{p}({v})" for p, v in ordered[:8])
            self.log(f"[FAILED] Read plate(s) but none verified: {shortlist}")
            self.review.append((url, location, f"read but unverified: {shortlist}"))
            self.emit({"type": "review", "category": "manual", "url": url,
                       "location": location, "site": site,
                       "note": f"read but unverified: {shortlist}"})
            return

        tier, votes, plate, vehicle = chosen
        vdesc = _vdesc(vehicle)
        warning = None
        if tier == 1:
            warning = (f"Make matches but model differs from listing '{model}'. "
                       f"Verify manually.")
            self.log(f"[WARNING] {plate} is a {vdesc} - {warning}")
        self.log(f"[SUCCESS] {plate} verified as {vdesc} ({votes} photo(s), "
                 f"tier {tier}/3)")
        self._write_record(plate, price, location, url, format_mot_history(vehicle))
        self._emit_result({**base, "plate": plate, "verified": True,
                           "corrected": False, "tier": tier, "votes": votes,
                           "warning": warning, **shape_vehicle(vehicle)})
        self.n_success += 1

    def _verify(self, ordered, make, model, requests):
        """Best-matching real vehicle: (tier, votes, plate, vehicle) or None."""
        best = None
        for plate, votes in ordered[:MAX_CANDIDATE_LOOKUPS]:
            if self.stop_event.is_set():
                break
            self.log(f"[Verify] Checking {plate} ({votes} photo(s))...")
            try:
                vehicle = self.mot.lookup(plate, requests)
            except Exception as exc:
                self.log(f"[Verify] MOT API error: {exc}")
                break
            if vehicle is None:
                continue
            tier = vehicle_tier(make, model, vehicle)
            if tier == 0:
                self.log(f"[Verify] {plate} is a real {vehicle.get('make', '')} - "
                         f"wrong make, ignoring")
                continue
            if tier == 3:
                return (tier, votes, plate, vehicle)
            if best is None or tier > best[0]:
                best = (tier, votes, plate, vehicle)
        return best

    def _near_miss(self, ordered, make, model, requests):
        """Single-char API variants of the top guesses; only a full match is kept."""
        self.log("[Correct] No exact match; trying near-miss variants...")
        tried = {p.replace(" ", "") for p, _ in ordered}
        lookups = 0
        for plate, _ in ordered[:3]:
            for variant in plate_near_misses(plate):
                if lookups >= MAX_NEAR_MISS_LOOKUPS or self.stop_event.is_set():
                    return None
                key = variant.replace(" ", "")
                if key in tried:
                    continue
                tried.add(key)
                lookups += 1
                try:
                    vehicle = self.mot.lookup(variant, requests)
                except Exception as exc:
                    self.log(f"[Correct] MOT API error: {exc}")
                    return None
                if vehicle and vehicle_tier(make, model, vehicle) == 3:
                    self.log(f"[Correct] {plate} -> {variant} matches {_vdesc(vehicle)}")
                    return (variant, vehicle)
        return None

    def _emit_result(self, result):
        self.emit({"type": "result", "result": result})

    def _write_record(self, plate, price, location, url, mot_text):
        block = (
            "-" * 50 + "\n"
            f"PLATE: {plate} | PRICE: {price} | WHERE: {location or 'N/A'}\n"
            f"URL: {url}\n" + "-" * 50 + "\n" + mot_text + "\n\n")
        try:
            with open(self.output_file, "a", encoding="utf-8") as fh:
                fh.write(block)
        except Exception as exc:
            self.log(f"[ERROR] Could not write to results file: {exc!r}")

    def _write_review_sections(self):
        if not self.dealer and not self.review:
            return
        lines = ["", "=" * 50]
        if self.dealer:
            lines.append(f"DEALER / TRADE PLATE LISTINGS ({len(self.dealer)})")
            lines.append("-" * 50)
            for url, loc in self.dealer:
                lines.append(f"{url}" + (f"  ({loc})" if loc and loc != "N/A" else ""))
            lines.append("")
        if self.review:
            lines.append("=" * 50)
            lines.append(f"PLEASE REVIEW MANUALLY ({len(self.review)})")
            lines.append("-" * 50)
            for url, loc, note in self.review:
                lines.append(f"{url}" + (f"  ({loc})" if loc and loc != "N/A" else ""))
                if note:
                    lines.append(f"    [{note}]")
            lines.append("")
        try:
            with open(self.output_file, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except Exception as exc:
            self.log(f"[ERROR] Could not write review sections: {exc!r}")


def _fetch_image(requests, img_url):
    try:
        resp = requests.get(img_url, timeout=15)
        if resp.status_code != 200 or not resp.content:
            return None
        ctype = resp.headers.get("Content-Type", "")
        if ctype and not ctype.startswith("image"):
            return None
        return resp.content
    except Exception:
        return None


def _vdesc(vehicle):
    return f"{vehicle.get('make', '')} {vehicle.get('model', '')}".strip()


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
