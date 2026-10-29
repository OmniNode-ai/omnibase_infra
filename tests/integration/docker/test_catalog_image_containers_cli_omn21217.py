# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the catalog CLI's ``image-containers`` command (OMN-21217).

The lab proof shells out to the real CLI, so this runs the real entry point in a
subprocess against the real catalog rather than a stand-in.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_IMAGE = "omnibase-infra-local-omnimarket-runtime:latest"


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "omnibase_infra.docker.catalog.cli", *args],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.mark.integration
def test_the_real_cli_names_the_containers_that_run_the_composed_image() -> None:
    result = _cli("image-containers", _IMAGE, "local-market")
    assert result.returncode == 0, result.stderr
    names = result.stdout.split()
    assert "omnibase-infra-local-omninode-runtime" in names
    assert "omnibase-infra-local-postgres" not in names


@pytest.mark.integration
def test_the_real_cli_refuses_an_unknown_bundle_without_a_traceback() -> None:
    result = _cli("image-containers", _IMAGE, "no-such-bundle")
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "no-such-bundle" in result.stderr
