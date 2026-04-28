# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Conftest: register scripts/ for coverage tracing when using importlib loading."""

from __future__ import annotations

import sys
from pathlib import Path

# Add the scripts directory to sys.path so that coverage.py can track
# importlib-loaded scripts by their canonical file path.
_SCRIPTS_DIR = str(Path(__file__).parent.parent.resolve())
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
