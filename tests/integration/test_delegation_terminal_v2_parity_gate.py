# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for delegation v2 terminal topic provisioning."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validation.validator_topic_suffix import validate_topic_suffix
from omnibase_infra.topics import (
    ALL_PROVISIONED_SUFFIXES,
    SUFFIX_DELEGATION_COMPLETED_V2,
    SUFFIX_DELEGATION_FAILED_ROUTED_V2,
    SUFFIX_DELEGATION_FAILED_UNROUTED_V2,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_SCRIPT = REPO_ROOT / "scripts" / "check_contract_topic_parity.py"

V2_TERMINAL_SUFFIXES: tuple[str, ...] = (
    SUFFIX_DELEGATION_COMPLETED_V2,
    SUFFIX_DELEGATION_FAILED_ROUTED_V2,
    SUFFIX_DELEGATION_FAILED_UNROUTED_V2,
)


def test_contract_topic_parity_gate_passes_with_delegation_v2_allowlist() -> None:
    completed = subprocess.run(
        [sys.executable, str(PARITY_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    combined = completed.stdout + completed.stderr
    assert completed.returncode == 0, combined
    assert "python-only gaps:   0" in completed.stdout, combined


@pytest.mark.parametrize("suffix", V2_TERMINAL_SUFFIXES)
def test_delegation_v2_terminal_suffix_is_valid_and_provisioned(suffix: str) -> None:
    result = validate_topic_suffix(suffix)

    assert result.is_valid, result.error
    assert suffix in ALL_PROVISIONED_SUFFIXES
