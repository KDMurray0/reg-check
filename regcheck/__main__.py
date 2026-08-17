"""Launch the inspection bay: start the web server and open the browser.

    python -m regcheck        (or: python run.py)

Host/port can be overridden with REGCHECK_HOST / REGCHECK_PORT.
"""

from __future__ import annotations

import os
import threading
import webbrowser

from .server import app


def main():
    host = os.environ.get("REGCHECK_HOST", "127.0.0.1")
    port = int(os.environ.get("REGCHECK_PORT", "5000"))
    url = f"http://{host}:{port}/"
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"\n  Reg-check inspection bay running at {url}")
    print("  Close this window (or Ctrl+C) to stop.\n")
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
