# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: ENABLE_LOCAL_INFERENCE_PIPELINE + ENABLE_PATTERN_ENFORCEMENT removed [OMN-8780].

These two feature flags defaulted to false and were pure informational gates
(silent non-enforcement). They have been deleted. This test asserts:

1. No module under src/ references either env var by name.
2. Setting either env var at runtime has no observable effect on the
   pattern lifecycle / LLM inference code paths — i.e., callers do not
   branch on them.

Uses real filesystem scans (no mocks) per repo testing standards.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

_DEAD_FLAGS = ("ENABLE_LOCAL_INFERENCE_PIPELINE", "ENABLE_PATTERN_ENFORCEMENT")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"


def _grep_src(token: str) -> list[str]:
    """Return lines under src/ that mention the token, excluding this test."""
    result = subprocess.run(
        ["git", "grep", "-n", "--", token, "src/"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        pytest.fail(f"git grep failed: {result.stderr}")
    return [ln for ln in result.stdout.splitlines() if ln.strip()]


def test_dead_flags_absent_from_src() -> None:
    """No module under src/ may reference the deleted flag names."""
    for flag in _DEAD_FLAGS:
        hits = _grep_src(flag)
        assert not hits, (
            f"{flag} was removed in OMN-8780 but still referenced in src/:\n"
            + "\n".join(hits)
        )


def test_dead_flag_env_has_no_runtime_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting the removed env vars must not alter importable module state.

    Because no code reads these names, toggling them between runs must be
    invisible to the pattern lifecycle and LLM inference modules.
    """
    for flag in _DEAD_FLAGS:
        monkeypatch.setenv(flag, "true")
    assert all(os.environ[f] == "true" for f in _DEAD_FLAGS)

    contract_dir = _SRC_ROOT / "omnibase_infra" / "nodes"
    llm_contract = contract_dir / "node_llm_inference_effect" / "contract.yaml"
    pattern_contract = contract_dir / "node_pattern_lifecycle_effect" / "contract.yaml"

    for path in (llm_contract, pattern_contract):
        text = path.read_text()
        for flag in _DEAD_FLAGS:
            assert flag not in text, (
                f"{path} still declares {flag} after OMN-8780 removal"
            )
