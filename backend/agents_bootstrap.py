"""
agents_bootstrap.py
────────────────────
Adds the `agents` Git submodule root to ``sys.path`` so that all
submodule-internal imports (``from messaging import …``,
``from agent.base_agent import Agent``, ``from master_planner.coordinator
import Coordinator``, etc.) resolve correctly when called from the BAI
project.

Import this module **once** at the very top of your entry-point (e.g.
``main.py``) before any ``agents``-package imports::

    import backend.agents_bootstrap  # noqa: F401  — path side-effect only
    from master_planner.coordinator import Coordinator
"""

import sys
from pathlib import Path

# Resolve <project_root>/agents — one level up from backend/, then into agents/
_AGENTS_ROOT = Path(__file__).resolve().parents[1] / "agents"

if str(_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTS_ROOT))

# Also ensure MemoryEngine inside the agents submodule is importable
# (needed by MemoryEngineAgent → agent.memory_agent.MemoryEngine.memory)
_MEMORY_ENGINE_ROOT = _AGENTS_ROOT / "agent" / "memory_agent" / "MemoryEngine"
if str(_MEMORY_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEMORY_ENGINE_ROOT))
