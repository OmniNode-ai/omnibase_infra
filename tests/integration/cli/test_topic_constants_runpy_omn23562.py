# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-23562: the topic drift script's runpy load of the suffix module must work.

``scripts/check_topic_drift.py`` executes ``platform_topic_suffixes.py`` with
``runpy`` in a fresh interpreter. With the topics package lazy, nothing preloads
that module, so the snapshot model it builds a spec from must not import it back.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SUFFIXES = _REPO / "src" / "omnibase_infra" / "topics" / "platform_topic_suffixes.py"


@pytest.mark.integration
def test_suffix_module_runs_standalone_in_fresh_interpreter() -> None:
    code = (
        "import runpy, sys;"
        f"ns = runpy.run_path({str(_SUFFIXES)!r});"
        "sys.exit(0 if ns['SUFFIX_REGISTRATION_SNAPSHOTS'] else 1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
