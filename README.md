# Reg Check — inspection bay

A local web app that hunts a used-vehicle search (Auto Trader or Cazoo), reads the
number plate off each listing's photos, and verifies it against the **official
DVSA MOT History API** — keeping the plate that is a real registered vehicle whose
make matches the listing. Every verified vehicle streams into the page as a card
headed by its rendered plate (the header links straight to the listing), with the
price, distance/location and full MOT history. Results are also written to
`extraction_results.txt`.

It runs entirely on your machine: a thin Flask server drives Playwright + a
dedicated ANPR plate-reading pipeline, and a single designed page shows progress
live over Server-Sent Events.

## Setup (Windows)

1. **Use Python 3.9–3.12.**

2. **Create a virtual environment and install dependencies:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Download the Playwright browser engine (one-time):**
   ```powershell
   playwright install chromium
   ```
   > The ANPR detector/recogniser weights download on first run (small). If you
   > also install the optional `easyocr` backup, it downloads ~100 MB the first
   > time.

4. **Run it:**
   ```powershell
   python -m regcheck
   ```
   The inspection bay opens at <http://127.0.0.1:5000/>. Close the console window
   (or Ctrl+C) to stop. Override the port with `REGCHECK_PORT` if 5000 is taken.

## Using it

Paste one or more **Auto Trader** or **Cazoo** links into the console, one per line.
Each can be:

- A **search-results URL** (recommended) — the app pages through every result:
  ```
  https://www.autotrader.co.uk/car-search?make=Nissan&model=Navara&postcode=M40 1QB&price-to=5000&body-type=Pickup
  https://www.cazoo.co.uk/vans/nissan/navara/
  ```
- An **individual listing URL**:
  ```
  https://www.autotrader.co.uk/car-details/202607274539745
  https://www.cazoo.co.uk/vans-for-sale/79533185/
  ```

From an Auto Trader search the make, model and **distance away** come straight from
the results; Cazoo is a national marketplace, so its cards carry the dealer's town
instead of a mileage.

**Thoroughness** trades speed for recall: *Fast* reads ~12 photos and stops at a
3-photo match; *Balanced* reads up to 24 (plate shots first); *Thorough* reads
every photo with no early stop — use it when a plate is being missed.

Verified vehicles appear as cards; listings with a dealer/trade placeholder plate,
or where no plate could be verified, drop into the **Dealer / trade plates** and
**Manual review** shelves — open those and check the photos yourself.

## DVSA MOT History API — free, official

MOT history uses the **official DVSA MOT History API** (free, one-time
registration). The public GOV.UK web checker sits behind bot protection that
blocks automation, so the sanctioned API is used instead.

Register for the DVSA MOT History API to get a **Client ID**, **Client Secret**,
**API Key** and **Token URL** (an Azure AD `.../oauth2/v2.0/token` URL). Enter the
four values once under **DVSA MOT API → Enter / update credentials**, or preset
them as environment variables:

```powershell
$env:MOT_CLIENT_ID     = "..."
$env:MOT_CLIENT_SECRET = "..."
$env:MOT_API_KEY       = "..."
$env:MOT_TOKEN_URL     = "https://login.microsoftonline.com/<tenant>/oauth2/v2.0/token"
```

The scope defaults to `https://tapi.dvsa.gov.uk/.default` (override with `MOT_SCOPE`).

Saved credentials are written to `reg_check_config.json` next to the app and are
**never sent back to the browser** — the page only shows whether each field is set.
That file holds your secret and API key in plain text: it is git-ignored and stays
on this PC. **Do not commit or share it.**

Without credentials the app still reads plates, but records the top guess
**unverified** with no MOT history.

## How the plate read works

Tuned for accuracy over speed:

1. **Gallery scoping.** Only *this* listing's own photos are used (Auto Trader
   embeds other cars and dealer banners). Auto Trader is pulled high-res
   (`w2048`); Cazoo images are grouped by vehicle id, keeping the largest gallery.
2. **Plate-first order.** On Auto Trader each photo is tagged (e.g. "Front Left" /
   "Exterior"); front and rear exterior shots — where the plate lives — are read
   first, so it is found sooner and with more agreement.
3. **Detect + tile.** A licence-plate detector (`open-image-models`, YOLO-v9) runs
   on the full image *and* a 4×4 tile grid, so small/distant plates survive
   downscaling.
4. **Ensemble read.** Each crop becomes several variants (raw, CLAHE, deskewed)
   read by an ensemble of plate recognisers (`fast-plate-ocr`). Only exactly-seven-
   character reads are kept and corrected to the UK `LL DD LLL` format.
5. **Verify against DVSA.** Guesses are checked most-consensus first; the first
   that is a **real vehicle matching the listing's make (and model)** wins.
   Wrong-make hits are discarded, so a misread that happens to be a real
   *other* car never slips through. If nothing matches, single-character variants
   of the best guesses are tried (recovers off-by-one misreads, flagged
   "OCR-corrected").

`easyocr` is an optional full-image recall backup (needs `torch`); the ANPR
ensemble does the heavy lifting, so the app runs fine without it.

## Notes & caveats

- **Selectors are best-effort.** Auto Trader and Cazoo change their markup and
  image hosts often; the harvest logic in `regcheck/scrape.py` may need updating
  over time.
- Some dealer photos blur or omit the plate. API verification removes wrong
  guesses, so an obscured plate lands in **Manual review** rather than producing a
  false result — try *Thorough* mode, or read it by eye from the listing.
- The randomised delay between listings is polite rate-limiting. Respect each
  site's Terms of Service and `robots.txt`, and only run this against listings you
  are entitled to access.
