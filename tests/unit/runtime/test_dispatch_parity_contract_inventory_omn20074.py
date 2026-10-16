# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run the committed AC11 inventory probe against its committed oracle blobs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]


def test_committed_ac11_inventory_probe_matches_regenerated_header() -> None:
    # The evidence verifier overlays contracts as well as tests on its control
    # tree. Read HEAD so that overlay cannot mask a stale committed inventory.
    contract = yaml.safe_load(
        subprocess.check_output(
            ["git", "show", "HEAD:contracts/OMN-20074.yaml"],
            cwd=ROOT,
            env=scrub_git_location_env(),
            text=True,
        )
    )
    evidence = next(
        item
        for item in contract["dod_evidence"]
        if item["id"] == "dod-dispatch-parity-declared-closure-ac11"
    )
    probes = [
        check["check_value"]
        for check in evidence["checks"]
        if "uv run python - <<'PY'\n" in check["check_value"]
    ]
    assert len(probes) == 1, "AC11 must retain its tracked inventory probe"
    _, _, body = probes[0].partition("uv run python - <<'PY'\n")
    script, delimiter, _ = body.rpartition("\nPY")
    assert delimiter, "AC11 inventory probe has no closing heredoc delimiter"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=scrub_git_location_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Tracked dispatch parity closure and inventories passed" in result.stdout
