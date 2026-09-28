from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Log to stderr; stdout is reserved for the stdout sink's JSON lines."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
