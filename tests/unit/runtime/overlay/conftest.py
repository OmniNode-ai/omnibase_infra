# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Conftest for overlay unit tests."""

from __future__ import annotations

import os
import sys
from importlib.util import find_spec
from pathlib import Path

if find_spec("omnibase_core") is None:
    core_src = os.environ.get("OMNIBASE_CORE_PATH")
    if core_src:
        core_path = Path(core_src)
        if core_path.exists() and str(core_path) not in sys.path:
            sys.path.insert(0, str(core_path))
