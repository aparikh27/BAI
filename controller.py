"""Minimal stub for the Webots controller module used by static analysis.

This module is only available at runtime inside a Webots simulator process.
The stub keeps editor analysis and import resolution working in the local
workspace when the real Webots Python API is not installed.
"""


class Robot:
    """Placeholder Robot implementation for non-Webots environments."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Webots controller module is not available in this environment")
