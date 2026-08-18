"""Number-plate reading and normalisation.

Two halves:

* Pure helpers (no third-party deps) that normalise reads into canonical UK
  registrations ("AB12 CDE"), correct per-position OCR confusions, apply the UK
  age-code prior, and enumerate near-miss variants for API recovery.
* The image pipeline (needs cv2 / numpy) that detects plates, builds enhanced
  crop variants, reads them with an ensemble of plate-specialised recognisers,
  and votes per character (see `consensus_candidates`). easyocr, if present,
  reads the crop for extra recall and for the dealer branding the plate-only
  models can't (so a dealer plate is told apart from a genuine failure).
"""

from __future__ import annotations

import itertools
import re
from collections import defaultdict

# Current-style UK plate: two letters, two digits, three letters ("AB12 CDE").
PLATE_STRICT_RE = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")

# Detector / recogniser tuning.
DETECTION_TILES = 4      # also run detection on a 4x4 grid for small/far plates
DETECTION_CONF = 0.12
OCR_MAG_RATIO = 1.5      # easyocr magnification (reads small/angled text better)


# --- Pure normalisation helpers --------------------------------------------

# Per-position confusion maps. A read character of the "wrong" type for its plate
# position is very likely one of these well-known OCR confusions.
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A",
                    "5": "S", "6": "G", "7": "T", "8": "B"}
_LETTER_TO_DIGIT = {"O": "0", "I": "1", "L": "1", "Z": "2", "A": "4", "S": "5",
                    "G": "6", "T": "7", "B": "8", "J": "3", "Q": "0", "D": "0"}


def _correct_plate(seq: str):
    """Force a 7-char sequence into 'LL DD LLL' via position confusion maps."""
    p = list(seq)
    for i in (0, 1, 4, 5, 6):        # letter positions
        if p[i].isdigit():
            p[i] = _DIGIT_TO_LETTER.get(p[i], p[i])
    for i in (2, 3):                 # digit positions
        if p[i].isalpha():
            p[i] = _LETTER_TO_DIGIT.get(p[i], p[i])
    s = "".join(p)
    return f"{s[:4]} {s[4:]}" if PLATE_STRICT_RE.match(s) else None


# Visually-similar characters, kept within type so variants stay valid plates.
_LETTER_NEIGHBOURS = {
    "B": "RPED", "C": "GOQ", "D": "OQCB", "E": "FBP", "F": "EPR", "G": "CQO",
    "H": "MN", "I": "LTJ", "J": "ILT", "K": "XR", "L": "ITC", "M": "NHW",
    "N": "MH", "O": "DQCG", "P": "FRBE", "Q": "ODG", "R": "BPK", "S": "Z",
    "T": "ILYJ", "U": "VY", "V": "UYW", "W": "MV", "X": "KY", "Y": "VUTX",
    "Z": "S",
}
_DIGIT_NEIGHBOURS = {
    "0": "86", "1": "7", "3": "89", "5": "68", "6": "580", "7": "1",
    "8": "0693", "9": "084",
}

# Valid current-style age identifiers (the two digits): March plates 02-25 and
# September plates 51-75. A real plate's digits are ALWAYS one of these, so a
# read whose digits fall outside is either a misread or an older/personal plate.
VALID_AGE = {f"{n:02d}" for n in list(range(2, 26)) + list(range(51, 76))}


def _age_fix(dd: str):
    """Valid age codes reachable by swapping one digit of a misread (e.g. an
    OCR '13'->'18' confusion), so a wrong-but-close age can still be verified."""
    out = set()
    for i in (0, 1):
        for alt in _DIGIT_NEIGHBOURS.get(dd[i], ""):
            cand = dd[:i] + alt + dd[i + 1:]
            if cand in VALID_AGE:
                out.add(cand)
    return out


def plate_near_misses(plate: str):
    """Yield 1-character 'confusion' variants of a plate (same UK format)."""
    reg = plate.replace(" ", "")
    if not PLATE_STRICT_RE.match(reg):
        return
    seen = set()
    for i, ch in enumerate(reg):
        neighbours = _DIGIT_NEIGHBOURS if i in (2, 3) else _LETTER_NEIGHBOURS
        for alt in neighbours.get(ch, ""):
            variant = reg[:i] + alt + reg[i + 1:]
            if variant != reg and variant not in seen:
                seen.add(variant)
                yield f"{variant[:4]} {variant[4:]}"


# Words that mark a detected "plate" as a dealer / trade placeholder rather than
# a real registration (dealer name, website, trade plate).
DEALER_KEYWORDS = (
    "TRADE", "TRADER", "TRADERS", "SALES", "MOTORS", "MOTORGROUP", "AUTOS",
    "AUTO", "AUTOMOTIVE", "GARAGE", "SPECIALIST", "APPROVED", "WARRANTY",
    "FINANCE", "SHOWROOM", "CENTRE", "GROUP", "MOTORING", "COMMERCIAL",
    "COMMERCIALS", "VEHICLE", "VEHICLES", "RANGE", "PRESTIGE", "SELECT",
    "WWW", "COUK", "COM", "LTD", "FORSALE", "SOLD", "CARS",
)

# A standard reg's longest letter run is three (the LL..LLL groups). A 5+ letter
# word on the plate is a dealer/trade/name plate - the threshold is 5 not 4 so an
# occasional misread digit (CU07 -> "CUOT") doesn't look like branding.
_DEALER_WORD_RE = re.compile(r"[A-Z]{5,}")


def contains_dealer_text(texts, exclude=()) -> bool:
    """True if a plate-region read looks like dealer/trade branding, not a reg.

    Used to tell a dealer plate ("BM RANGE", "TOPGEAR MOTORGROUP") from a genuine
    unreadable-plate failure once the DVSA check has found no real vehicle. Pass
    the listing's make/model as `exclude` so the tailgate badge near the rear
    plate (NISSAN, NAVARA) isn't mistaken for a dealer name.
    """
    joined = " ".join(t.upper() for t in texts)
    if any(kw in joined for kw in DEALER_KEYWORDS):
        return True
    ex = [e.upper() for e in exclude if e]
    for word in _DEALER_WORD_RE.findall(joined):
        if not any(word in e or e in word for e in ex):
            return True
    return False


# --- Image pipeline (cv2 / numpy) ------------------------------------------

def decode_image(cv2, np, img_bytes):
    """Decode image bytes to a BGR array, upscaling very small images."""
    try:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None:
        return None
    w = img.shape[1]
    if w < 1200:
        s = 1200.0 / w
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return img


# --- Confidence-weighted character-level reading (primary) -----------------
#
# The recognisers return a per-character probability for every read. Rather than
# voting on whole-plate strings (where one wrong character throws the whole read
# away), we collect every 7-character read with its per-character confidence and
# vote PER POSITION. The winning character at each position reconstructs the true
# plate even when no single read got all seven right - and a small beam over the
# genuinely ambiguous positions gives the DVSA check a few high-value variants to
# confirm. This is the single biggest accuracy win, especially on Cazoo photos
# with no plate-first ordering.

def read_plate_reads(img, engines, cv2, np, enhanced=True):
    """Read one image, returning (reads, texts).

    reads - list of (chars, probs): every exactly-7-character recogniser output
            with its per-character confidence (probs aligned to chars).
    texts - raw plate-region strings, used to spot dealer/trade branding.
    """
    reads: list[tuple[str, list[float]]] = []
    texts: list[str] = []
    detector = engines.get("detector")
    plate_ocrs = engines.get("plate_ocrs") or []
    reader = engines.get("easyocr")

    for crop, weight in _detect_plate_crops(detector, cv2, np, img):
        for variant in _crop_variants(cv2, np, crop, enhanced):
            for pocr in plate_ocrs:
                raw, probs = _plate_ocr_conf(pocr, variant)
                if not raw:
                    continue
                r = _clean_read(raw, probs)
                if r:                       # scale this read's vote mass by size
                    chars, ps = r
                    reads.append((chars, [p * weight for p in ps]))
        # easyocr reads whatever is written on the plate region - real reg chars
        # (extra recall) AND the dealer/trade branding the plate-only recognisers
        # can't ("TOPGEAR MOTORGROUP", "BM RANGE"), which is how a dealer plate is
        # told apart from a genuine unreadable-plate failure.
        if reader is not None:
            for word in _easyocr_words(reader, cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)):
                texts.append(word)
                alnum = re.sub(r"[^A-Z0-9]", "", word)
                if len(alnum) == 7:
                    reads.append((alnum, [0.5 * weight] * 7))

    return reads, texts


def _clean_read(raw: str, probs):
    """Keep the alphanumeric characters (and their confidences) from one read;
    return (chars, probs) only if exactly seven remain."""
    chars, ps = [], []
    for i, ch in enumerate(raw.upper()):
        if ch.isalnum():
            chars.append(ch)
            ps.append(float(probs[i]) if probs is not None and i < len(probs) else 0.6)
    return ("".join(chars), ps) if len(chars) == 7 else None


def _plate_ocr_conf(plate_ocr, gray):
    """Run one recogniser, returning (raw_string_with_padding, char_probs)."""
    try:
        preds = plate_ocr.run(gray, return_confidence=True, remove_pad_char=False)
        pred = preds[0] if isinstance(preds, list) and preds else preds
        raw = getattr(pred, "plate", "") or ""
        probs = getattr(pred, "char_probs", None)
        return raw, probs
    except Exception:
        return "", None


def _easyocr_words(reader, gray):
    """Every text token easyocr reads on the plate crop, uppercased (words kept,
    so dealer branding like 'RANGE' survives for dealer-plate detection)."""
    try:
        out = reader.readtext(gray, detail=0, paragraph=False,
                              mag_ratio=OCR_MAG_RATIO)
        return [re.sub(r"[^A-Z0-9 ]", "", (t or "").upper()).strip()
                for t in out if t and t.strip()]
    except Exception:
        return []


def consensus_candidates(reads, max_candidates=12, age_prior=True):
    """Rank plate guesses from confidence-weighted per-position voting.

    Every read is first forced to the UK 'LL DD LLL' shape, so per-position tallies
    live in a space where letter positions hold only letters and digit positions
    only digits. The per-position winners give the consensus plate; a beam over the
    up-to-three genuinely ambiguous positions adds a handful of variants. Each
    candidate is scored by the confidence mass backing its exact characters.

    With age_prior on, candidates whose two digits are a real DVLA age code are
    preferred, and age-valid variants of any near-miss age are added for the DVSA
    check to confirm.

    Returns [(plate, score)] best first - the order the DVSA check runs in.
    """
    pos = [defaultdict(float) for _ in range(7)]
    whole: dict[str, float] = defaultdict(float)
    n = 0
    for chars, ps in reads:
        corr = _correct_plate(chars)
        if not corr:
            continue
        reg = corr.replace(" ", "")
        for i in range(7):
            pos[i][reg[i]] += ps[i]
        whole[corr] += sum(ps) / 7.0
        n += 1
    if not n:
        return []

    ranked = [sorted(pos[i].items(), key=lambda kv: kv[1], reverse=True)
              for i in range(7)]
    base = [ranked[i][0][0] for i in range(7)]
    ambiguous = [i for i in range(7) if len(ranked[i]) >= 2
                 and ranked[i][1][1] >= 0.4 * ranked[i][0][1]]
    ambiguous = sorted(ambiguous,
                       key=lambda i: ranked[i][1][1] / (ranked[i][0][1] + 1e-9),
                       reverse=True)[:3]

    pool = set(whole)                       # whole-plate reads that already agreed
    for bits in itertools.product((0, 1), repeat=len(ambiguous)):
        reg = list(base)
        for k, i in enumerate(ambiguous):
            reg[i] = ranked[i][bits[k]][0]
        reg = "".join(reg)
        pool.add(f"{reg[:4]} {reg[4:]}")

    if age_prior:                           # add age-valid variants of near-miss ages
        for cand in list(pool):
            reg = cand.replace(" ", "")
            if reg[2:4] not in VALID_AGE:
                for dd in _age_fix(reg[2:4]):
                    pool.add(f"{reg[:2]}{dd} {reg[4:]}")

    def score(plate: str) -> float:
        reg = plate.replace(" ", "")
        s = sum(pos[i].get(reg[i], 0.0) for i in range(7))
        if age_prior and reg[2:4] in VALID_AGE:
            s *= 1.15                        # prefer real age codes
        return s

    return sorted(((p, score(p)) for p in pool),
                  key=lambda kv: kv[1], reverse=True)[:max_candidates]


def _detect_boxes(detector, img):
    try:
        return [(int(d.bounding_box.x1), int(d.bounding_box.y1),
                 int(d.bounding_box.x2), int(d.bounding_box.y2))
                for d in detector.predict(img)
                if float(d.confidence) >= DETECTION_CONF]
    except Exception:
        return []


def _detect_plate_crops(detector, cv2, np, img) -> list:
    """Return (crop, weight) for every plate the detector finds - the full image
    plus an NxN tile grid for small/distant plates.

    weight favours the FOREGROUND plate (larger, more central), so a background
    vehicle's plate on a dealer's lot can't out-vote the one being sold: it counts
    for a fraction of a full-size, central detection.
    """
    if detector is None:
        return []
    H, W = img.shape[:2]
    boxes = list(_detect_boxes(detector, img))
    n = DETECTION_TILES
    if n and n > 1:
        tw, th = W // n, H // n
        ox, oy = int(tw * 0.2), int(th * 0.2)
        for iy in range(n):
            for ix in range(n):
                x0, y0 = max(0, ix * tw - ox), max(0, iy * th - oy)
                x1, y1 = min(W, (ix + 1) * tw + ox), min(H, (iy + 1) * th + oy)
                tile = img[y0:y1, x0:x1]
                if tile.size == 0:
                    continue
                for (bx1, by1, bx2, by2) in _detect_boxes(detector, tile):
                    boxes.append((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0))

    max_area = max((max(1, (x2 - x1) * (y2 - y1)) for (x1, y1, x2, y2) in boxes),
                   default=1)
    crops, seen = [], set()
    for (x1, y1, x2, y2) in boxes:
        key = (x1 // 30, y1 // 30, x2 // 30, y2 // 30)  # de-dup overlaps
        if key in seen:
            continue
        seen.add(key)
        area = max(1, (x2 - x1) * (y2 - y1))
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        off = ((cx - W / 2) / W) ** 2 + ((cy - H / 2) / H) ** 2   # 0 .. ~0.5
        weight = max(0.08, (area / max_area) ** 0.5 * max(0.5, 1.0 - off))
        padx, pady = int((x2 - x1) * 0.1), int((y2 - y1) * 0.3)
        crop = img[max(0, y1 - pady):y2 + pady, max(0, x1 - padx):x2 + padx]
        if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
            continue
        if crop.shape[1] < 240:  # enlarge tiny plates for the OCR
            s = 240.0 / crop.shape[1]
            crop = cv2.resize(crop, None, fx=s, fy=s,
                              interpolation=cv2.INTER_CUBIC)
        crops.append((crop, weight))
    return crops


def _crop_variants(cv2, np, crop, enhanced=False) -> list:
    """Grayscale variants of a plate crop for the recognisers to read.

    Always: raw, CLAHE-equalised, and a perspective-deskewed frontal warp.
    Enhanced adds Otsu binarisation, an unsharp-mask sharpen, and a small +/-7
    degree rotation sweep - each a fresh shot for the recognisers, which the
    character-level consensus then turns into accuracy.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variants = [gray]
    try:
        variants.append(cv2.createCLAHE(clipLimit=2.0,
                                        tileGridSize=(8, 8)).apply(gray))
    except Exception:
        pass
    try:
        desk = _deskew(cv2, np, crop)
        if desk is not None:
            variants.append(desk)
    except Exception:
        pass
    if enhanced:
        try:
            variants.append(cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        except Exception:
            pass
        try:
            blur = cv2.GaussianBlur(gray, (0, 0), 3)
            variants.append(cv2.addWeighted(gray, 1.6, blur, -0.6, 0))
        except Exception:
            pass
        h, w = gray.shape[:2]
        for ang in (-7, 7):
            try:
                m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
                variants.append(cv2.warpAffine(gray, m, (w, h),
                                borderMode=cv2.BORDER_REPLICATE))
            except Exception:
                pass
    return variants


def _deskew(cv2, np, crop):
    """Perspective-correct an angled plate to a frontal rectangle, if found."""
    g = cv2.bilateralFilter(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 11, 17, 17)
    cnts, _ = cv2.findContours(cv2.Canny(g, 30, 200), cv2.RETR_LIST,
                               cv2.CHAIN_APPROX_SIMPLE)
    area = crop.shape[0] * crop.shape[1]
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(c) > 0.2 * area:
            pts = approx.reshape(4, 2).astype("float32")
            s, d = pts.sum(1), np.diff(pts, axis=1)
            rect = np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                             pts[np.argmax(s)], pts[np.argmax(d)]], "float32")
            (tl, tr, br, bl) = rect
            w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
            if w < 40 or h < 15:
                continue
            dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                           "float32")
            warp = cv2.warpPerspective(crop, cv2.getPerspectiveTransform(rect, dst),
                                       (w, h))
            return cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    return None


