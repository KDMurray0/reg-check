# PyInstaller spec for reg-check (onedir).
#
# ANPR-only frozen build: torch + easyocr are excluded to keep the bundle small
# and reliable (they add ~2 GB and are fragile to freeze). The app already runs
# on the ONNX ANPR ensemble alone and degrades gracefully without easyocr, so the
# only things the frozen build loses are the easyocr recall backup and the
# easyocr-based dealer-plate text read (dealer plates fall to manual review).
# Run from source (python -m regcheck) if you want those.
#
# Build:   pyinstaller reg-check.spec --noconfirm
# Run:     dist/reg-check/reg-check.exe   (needs `playwright install chromium` once;
#          the ANPR model weights download on first run)

from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = [("regcheck/static", "regcheck/static")]
binaries = []
hiddenimports = ["regcheck", "regcheck.server", "regcheck.engine", "regcheck.scrape",
                 "regcheck.plates", "regcheck.mot", "regcheck.review",
                 "cv2", "requests", "playwright.sync_api"]

for pkg in ("onnxruntime", "fast_plate_ocr", "open_image_models", "playwright", "flask"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

datas += collect_data_files("cv2")

a = Analysis(
    ["run.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["torch", "torchvision", "torchaudio", "easyocr", "matplotlib",
              "tkinter", "PyQt5", "PySide2", "notebook", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="reg-check",
          console=True, disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, name="reg-check")
