"""Listing scraping for Auto Trader and Cazoo (Playwright page passed in).

Turns a search URL into listing URLs (with distance/location), and a listing URL
into its price, make/model, location and gallery. Two things make the plate read
reliable on Auto Trader:

* Gallery scoping - only THIS listing's own "images":[...] array is used, so the
  "more from this dealer" / similar-vehicle thumbnails don't crowd out the plate.
* Plate-first ordering - the gallery JSON tags each photo (e.g. "Front Left" /
  category "Exterior"); front/rear exterior shots (where the plate lives) are read
  first, so the plate is found sooner and with more agreement.

Cazoo (relaunched as a marketplace) serves galleries from the autoexposure CDN as
{dealer}/{vehicle}_{n}.jpg. The page embeds other vehicles too, so images are
grouped by vehicle id and the largest group (this listing's own gallery) is kept.
It's a national marketplace, so "distance" is the dealer's town rather than miles.
"""

from __future__ import annotations

import random
import re
from urllib.parse import unquote_plus

AUTOTRADER_BASE = "https://www.autotrader.co.uk"
MAX_SEARCH_PAGES = 60

# Auto Trader: m.atcdn.co.uk with a resize token (.../a/media/{resize}/<hash>.jpg);
# w2048 is the maximum native size.
ATCDN_HASH_RE = re.compile(r"a/media/[^/\"]*/([a-f0-9]{32})", re.I)
ATCDN_HIRES = "https://m.atcdn.co.uk/a/media/w2048/{hash}.jpg"

# Cazoo: autoexposure CDN as {dealer}/{vehicle}_{n}.jpg.
AUTOEXP_RE = re.compile(
    r"https?://cdn\.images\.autoexposure\.co\.uk/[A-Za-z0-9]+/"
    r"([A-Za-z0-9]+)_(\d+)\.jpg", re.I)


def is_cazoo(url: str) -> bool:
    return "cazoo.co.uk" in url.lower()


def is_search_url(url: str) -> bool:
    """True if the URL is a results page to expand (not a single listing)."""
    u = url.lower()
    if is_cazoo(u):
        return not re.search(r"-for-sale/\d+", u)   # any non-listing Cazoo page
    return "/car-search" in u


def search_make_model(url: str):
    """Extract (make, model) from a search URL - query params or Cazoo slugs."""
    m1 = re.search(r"[?&]make=([^&]+)", url)
    m2 = re.search(r"[?&]model=([^&]+)", url)
    make = unquote_plus(m1.group(1)) if m1 else ""
    model = unquote_plus(m2.group(1)) if m2 else ""
    if make or model:
        return make, model
    m = re.search(r"/(?:cars|vans)/([^/?]+)(?:/([^/?]+))?", url)  # Cazoo path form
    if m:
        return m.group(1) or "", m.group(2) or ""
    return "", ""


def _with_page_param(url: str, n: int) -> str:
    if re.search(r"[?&]page=\d+", url):
        return re.sub(r"([?&]page=)\d+", lambda m: m.group(1) + str(n), url)
    return url + ("&" if "?" in url else "?") + "page=" + str(n)


def _site_base(url: str) -> str:
    m = re.match(r"https?://[^/]+", url)
    return m.group(0) if m else AUTOTRADER_BASE


# Auto Trader search cards: listing link + distance from the location chip (falls
# back to scanning card text for "(N miles)" if the testid ever changes).
_AT_SEARCH_JS = r"""
() => {
  const results = [];
  const seen = new Set();
  document.querySelectorAll('a[href*="/car-details/"]').forEach(a => {
    const path = a.getAttribute('href').split('?')[0];
    if (!/\/car-details\/\d+/.test(path) || seen.has(path)) return;
    seen.add(path);
    let card = a;
    for (let i = 0; i < 8 && card.parentElement; i++) {
      card = card.parentElement;
      if (card.querySelector('[data-testid="search-listing-location"]')) break;
    }
    const loc = card.querySelector('[data-testid="search-listing-location"]');
    const text = (loc ? loc.innerText : card.innerText) || '';
    const m = text.match(/\(([\d.]+)\s*mile/i);
    results.push({path: path, distance: m ? m[1] : null, location: null, price: null});
  });
  return results;
}
"""

# Cazoo search cards: /vans-for-sale/<id>/ links, with the card's price and dealer
# town (no per-listing mileage distance on a national marketplace).
_CAZOO_SEARCH_JS = r"""
() => {
  const results = [];
  const seen = new Set();
  const cards = document.querySelectorAll('[data-testid="search-result"]');
  const scope = cards.length ? cards : [document];
  scope.forEach(card => {
    const a = card.querySelector ? card.querySelector('a[href*="-for-sale/"]') : null;
    if (!a) return;
    const path = a.getAttribute('href').split('?')[0];
    if (!/-for-sale\/\d+/.test(path) || seen.has(path)) return;
    seen.add(path);
    let price = null, location = null;
    card.querySelectorAll('*').forEach(e => {
      if (e.children.length) return;
      const t = e.textContent.trim();
      if (!price && /^£[\d,]{3,}$/.test(t)) price = t;
      if (!location && /^[A-Za-z][A-Za-z.\s'-]+,\s*[A-Za-z][A-Za-z.\s'-]+$/.test(t)
          && t.length < 40) location = t;
    });
    results.push({path: path, distance: null, location: location, price: price});
  });
  return results;
}
"""


def collect_search_results(page, url, log, stop_event) -> list[dict]:
    """Page through a search URL, returning per-listing dicts.

    Each dict: {url, distance (miles str|None), location (town str|None),
    price (str|None)} - the hints a card exposes; the rest come from the listing.
    """
    cazoo = is_cazoo(url)
    harvest_js = _CAZOO_SEARCH_JS if cazoo else _AT_SEARCH_JS
    link_sel = 'a[href*="-for-sale/"]' if cazoo else 'a[href*="/car-details/"]'
    base = _site_base(url)
    results, seen = [], set()
    for n in range(1, MAX_SEARCH_PAGES + 1):
        if stop_event.is_set():
            break
        try:
            page.goto(_with_page_param(url, n),
                      wait_until="domcontentloaded", timeout=45000)
        except Exception as exc:
            log(f"[Search] Page {n} load error: {exc!r}")
            break
        try:
            page.wait_for_selector(link_sel, timeout=12000)
        except Exception:
            break  # no listings rendered -> reached the last page
        page.wait_for_timeout(1000)
        for _ in range(3):
            page.mouse.wheel(0, 3000)
            page.wait_for_timeout(400)
        try:
            rows = page.evaluate(harvest_js)
        except Exception:
            rows = []
        new = 0
        for r in rows:
            path = r.get("path")
            if not path or path in seen:
                continue
            seen.add(path)
            listing_url = path if path.startswith("http") else base + path
            results.append({"url": listing_url, "distance": r.get("distance"),
                            "location": r.get("location"), "price": r.get("price")})
            new += 1
        log(f"[Search] Page {n}: +{new} new listing(s) (total {len(results)})")
        if new == 0:
            break
    return results


# --- Gallery ---------------------------------------------------------------

def _label_score(label: str, category: str) -> int:
    """Rank a photo by how likely it shows the number plate (higher = sooner)."""
    lab, cat = label.upper(), category.upper()
    exterior = "EXTERIOR" in cat or (not cat and "INTERIOR" not in lab)
    facing = any(k in lab for k in ("FRONT", "REAR", "BACK"))
    if exterior and facing:
        return 3            # front/rear exterior - the plate shots
    if exterior:
        return 2            # other exterior - plate often visible at an angle
    if "INTERIOR" in cat or "INTERIOR" in lab or "DASH" in lab or "ENGINE" in lab:
        return 0            # inside/engine bay - no plate
    return 1                # untagged / unknown


def _autotrader_gallery(html: str) -> list[str]:
    """This listing's own gallery URLs, plate-bearing shots first.

    The listing's own photos are the FIRST "images":[...] array in the page-state
    JSON, each {"url": "...{resize}/<hash>.jpg", "classificationTags": [{"label",
    "category"}]}. Grabbing every atcdn hash would pull in other vehicles.
    """
    u = html.replace('\\"', '"').replace("\\u002F", "/").replace("\\/", "/")
    start = u.find('"images":[')
    if start < 0:
        return []
    i = u.find("[", start)
    depth, end = 0, -1
    for j in range(i, min(len(u), i + 400000)):
        if u[j] == "[":
            depth += 1
        elif u[j] == "]":
            depth -= 1
            if depth == 0:
                end = j
                break
    if end < 0:
        return []
    segment = u[i:end + 1]

    scored, seen, order = [], set(), 0
    for m in ATCDN_HASH_RE.finditer(segment):
        h = m.group(1)
        if h in seen:
            continue
        seen.add(h)
        window = segment[m.end():m.end() + 400]   # this photo's own object tail
        lab = re.search(r'"label"\s*:\s*"([^"]*)"', window)
        cat = re.search(r'"category"\s*:\s*"([^"]*)"', window)
        score = _label_score(lab.group(1) if lab else "",
                             cat.group(1) if cat else "")
        scored.append((score, order, ATCDN_HIRES.format(hash=h)))
        order += 1
    scored.sort(key=lambda t: (-t[0], t[1]))       # best score first, else order
    return [url for _, _, url in scored]


def _cazoo_gallery(html: str) -> list[str]:
    """Cazoo: group autoexposure images by vehicle id, keep the largest gallery."""
    groups: dict[str, list] = {}
    for m in AUTOEXP_RE.finditer(html):
        groups.setdefault(m.group(1), []).append((int(m.group(2)), m.group(0)))
    if not groups:
        return []
    best = max(groups.values(), key=len)
    best.sort()
    return [u for _, u in best]


def harvest_images(page, cazoo: bool) -> list[str]:
    """Return the listing's gallery image URLs (high resolution)."""
    try:
        html = page.content()
    except Exception:
        html = ""
    gallery = _cazoo_gallery(html) if cazoo else _autotrader_gallery(html)
    if gallery:
        return gallery
    # Fallback: any atcdn hashes on the page (unscoped) if the array isn't found.
    seen, hires = set(), []
    for h in ATCDN_HASH_RE.findall(html):
        if h not in seen:
            seen.add(h)
            hires.append(ATCDN_HIRES.format(hash=h))
    return hires


# --- Listing detail fields --------------------------------------------------

def extract_price(page) -> str:
    """Headline price - the dedicated advert-price element, else a text scan."""
    try:
        el = page.query_selector('[data-testid="advert-price"]')
        if el:
            m = re.search(r"£\s?[\d,]{3,}", (el.inner_text() or ""))
            if m:
                return m.group(0).replace(" ", "")
    except Exception:
        pass
    try:
        text = page.inner_text("body")
    except Exception:
        text = ""
    amounts = []
    for m in re.findall(r"£\s?([\d,]{3,})", text):
        try:
            value = int(m.replace(",", ""))
        except ValueError:
            continue
        if 300 <= value <= 500000:
            amounts.append(value)
    if not amounts:
        return "N/A"
    big = [v for v in amounts if v >= 1000]
    return f"£{(big[0] if big else amounts[0]):,}"


def _title_words(page):
    """(make, model) from the page title, both site title styles."""
    try:
        title = page.title() or ""
    except Exception:
        title = ""
    # Cazoo: "Used Nissan Navara 2015 for sale in ..." (year sits after the model).
    m = re.search(r"\bUsed\s+([A-Za-z-]+)\s+([A-Za-z0-9-]+)", title)
    if m:
        return (m.group(1), m.group(2))
    # Auto Trader: "2007 Nissan Navara for sale ..." / "2013 Grey Nissan Navara ...".
    m = re.search(r"\b(?:19|20)\d{2}\s+(?:\w+\s+)?([A-Za-z-]+)\s+([A-Za-z0-9-]+)"
                  r"\s+for sale", title)
    if m:
        return (m.group(1), m.group(2))
    return ("", "")


def extract_make(page, url) -> str:
    m = re.search(r"[?&]make=([^&]+)", url)
    return unquote_plus(m.group(1)) if m else _title_words(page)[0]


def extract_model(page, url) -> str:
    m = re.search(r"[?&]model=([^&]+)", url)
    return unquote_plus(m.group(1)) if m else _title_words(page)[1]


def extract_location(page, cazoo: bool):
    """Display 'distance/where' for a listing.

    Auto Trader gives miles from the postcode ('7 miles'); Cazoo is national, so
    the dealer town from the page title ('... for sale in Royton, Oldham at Cazoo')
    is used instead.
    """
    if cazoo:
        try:
            title = page.title() or ""
        except Exception:
            title = ""
        m = re.search(r"for sale in (.+?) at Cazoo", title, re.I)
        if m:
            return m.group(1).strip(), None
        return None, None
    try:
        text = page.inner_text("body")
    except Exception:
        text = ""
    m = (re.search(r"\(([\d.]+)\s*mile", text)
         or re.search(r"([\d.]+)\s*miles?\s*away", text, re.I))
    if m:
        return f"{m.group(1)} miles", m.group(1)
    return None, None


def scrape_listing(page, url, max_images, log):
    """Return (price, make, model, location, distance_miles, [image_urls])."""
    cazoo = is_cazoo(url)
    page.goto(url, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(random.randint(1500, 3000))
    for _ in range(6):                       # trigger lazy-loaded gallery images
        page.mouse.wheel(0, 2500)
        page.wait_for_timeout(400)

    price = extract_price(page)
    make = extract_make(page, url)
    model = extract_model(page, url)
    location, distance_miles = extract_location(page, cazoo)
    images = harvest_images(page, cazoo)
    order = "" if cazoo else ", plate shots first"
    log(f"[Scraping] Price {price}, {make or '?'} {model or ''}".rstrip()
        + f", {len(images)} image(s) (reading up to {max_images}{order})")
    return price, make, model, location, distance_miles, images[:max_images]
