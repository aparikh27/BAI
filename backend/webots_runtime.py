"""
webots_runtime.py — Real Webots controller module resolution
────────────────────────────────────────────────────────────
The repository root ships a ``controller.py`` stub so editors and static
analysis can resolve ``from controller import Robot`` outside the simulator.
That stub sits on ``sys.path[0]`` whenever the app is launched from the project
root, so it *shadows* the genuine Webots Python API even when Webots is
installed and running.  The result is a coordinator that never initialises and
an API surface that answers 503 to every request.

This module resolves the real ``controller`` package from ``WEBOTS_HOME`` and
registers it in ``sys.modules`` before anything imports it, falling back to the
stub-driven "no hardware" path when Webots genuinely is not installed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Default install locations, in probe order.  ``WEBOTS_HOME`` always wins.
_DEFAULT_WEBOTS_HOMES = (
    Path.home() / "AppData" / "Local" / "Programs" / "Webots",
    Path("C:/Program Files/Webots"),
    Path("/usr/local/webots"),
    Path("/Applications/Webots.app"),
)


def _candidate_homes() -> list[Path]:
    env_home = os.environ.get("WEBOTS_HOME")
    candidates = [Path(env_home)] if env_home else []
    candidates.extend(_DEFAULT_WEBOTS_HOMES)
    return [c for c in candidates if c.is_dir()]


def _register_dll_directories(home: Path) -> None:
    """Windows needs the Webots native libs discoverable before import."""
    if not hasattr(os, "add_dll_directory"):
        return

    for relative in (("lib", "controller"), ("msys64", "mingw64", "bin")):
        dll_dir = home.joinpath(*relative)
        if dll_dir.is_dir():
            try:
                os.add_dll_directory(str(dll_dir))
            except OSError:  # pragma: no cover - depends on the host OS
                pass


def load_robot_class():
    """Return the genuine Webots ``Robot`` class, or ``None`` when unavailable.

    ``None`` signals the caller to run the API without robot hardware — the
    same contract ``main.py`` previously expected from a failed import.
    """
    for home in _candidate_homes():
        python_api = home / "lib" / "controller" / "python"
        if not (python_api / "controller").is_dir():
            continue

        os.environ.setdefault("WEBOTS_HOME", str(home))
        _register_dll_directories(home)

        # Drop the shadowing stub so the real package wins the import.
        sys.modules.pop("controller", None)
        sys.path.insert(0, str(python_api))

        try:
            from controller import Robot  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - depends on the Webots build
            print(f"[BAI] Found Webots at '{home}' but could not import its API: {exc}")
            sys.path.remove(str(python_api))
            continue

        print(f"[BAI] Webots Python API loaded from '{python_api}'")
        return Robot

    print("[BAI] No Webots installation found; running without robot hardware.")
    return None
