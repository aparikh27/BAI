"""
run_backend_webots.py — Launch the FastAPI backend as a Webots extern controller
────────────────────────────────────────────────────────────────────────────────
The backend must run *inside* a Webots extern-controller environment for
``Robot()`` to attach to the simulator. The supported way to arrange that is
Webots' own launcher::

    webots-controller --robot-name="Khepera III" <python> scripts/run_backend_webots.py

Two things make that launcher awkward to use directly, both handled here:

* it sets the working directory to the *interpreter's* directory, which would
  break every relative path the app relies on (``yolo11n.pt``,
  ``backend/models/...``, ``robot_memory.db``), so we chdir back to the repo; and
* ``--reload`` must never be used, because uvicorn's reloader runs the app in a
  child process that would not inherit the controller connection.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Restore the working directory the application's relative paths assume.
os.chdir(ROOT)

# Make the project importable regardless of how the interpreter was invoked.
# ``backend.webots_runtime`` re-resolves the genuine Webots ``controller``
# package ahead of the repo-root stub, so adding ROOT here is safe.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import uvicorn

    host = os.environ.get("BAI_HOST", "127.0.0.1")
    port = int(os.environ.get("BAI_PORT", "8000"))

    print(f"[BAI] Launching backend from {ROOT} on {host}:{port}", flush=True)
    print(f"[BAI] WEBOTS_CONTROLLER_URL={os.environ.get('WEBOTS_CONTROLLER_URL')}", flush=True)

    # Single process, no reloader: the extern-controller handle cannot be
    # inherited by a reload child.
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
