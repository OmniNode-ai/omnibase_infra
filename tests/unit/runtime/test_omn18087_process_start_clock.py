# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18087: the boot bound counts from process start, not from module import."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from omnibase_infra.handlers.boot_readiness_checks import (
    seconds_since_process_start,
)

# Fields 3..21 of /proc/<pid>/stat between ``state`` and ``starttime``.
_STAT_PREFIX = "42 (python 3) S " + " ".join(["0"] * 18)


@pytest.mark.unit
def test_seconds_since_process_start_reads_proc_clocks() -> None:
    stat = f"{_STAT_PREFIX} 25000 0 0"
    assert seconds_since_process_start(stat, "1000.50 2000.00", 100) == pytest.approx(
        750.5
    )


@pytest.mark.unit
def test_seconds_since_process_start_survives_spaces_in_comm() -> None:
    stat = f"{_STAT_PREFIX} 100 0 0".replace("python 3", "a) b (c")
    assert seconds_since_process_start(stat, "10.0 0.0", 100) == pytest.approx(9.0)


@pytest.mark.unit
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_clock_precedes_module_import() -> None:
    """Time spent before the module is imported counts against the bound."""
    script = textwrap.dedent(
        """
        import time
        time.sleep(1.5)
        before_import = time.monotonic()
        from omnibase_infra.handlers import boot_readiness_checks as m
        print(before_import - m.PROCESS_START_MONOTONIC)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert float(out.stdout.strip().splitlines()[-1]) >= 1.5
