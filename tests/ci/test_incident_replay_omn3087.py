# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay the verified-grant projection/recorder bypass (omnibase_infra#3087)."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _ROOT / "tests/fixtures/omn3087/issue-3087.gh-api.json.captured"
_VALIDATOR = runpy.run_path(
    str(_ROOT / "scripts/validation/validate_verified_grant_ingress_boundary.py")
)["_violations"]


@pytest.mark.unit
def test_captured_projection_recorder_bypass_is_rejected_by_the_real_boundary_guard(
    tmp_path: Path,
) -> None:
    """Use the GitHub-captured incident record, never a retyped bypass claim."""
    captured = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert captured["number"] == 3087
    assert captured["state"] == "open"
    body = captured["body"]
    assert isinstance(body, str)

    observed = (
        "ModelFirstEffectVerifiedGrantProjection",
        "PostgresVerifiedFirstEffectRecorder",
        "record_verified_grant",
    )
    assert all(f"`{identifier}`" in body for identifier in observed)

    reproduced_source = tmp_path / "removed_verified_grant_capability.py"
    reproduced_source.write_text(
        "\n".join(f"capability = {identifier!r}" for identifier in observed),
        encoding="utf-8",
    )

    violations = _VALIDATOR(reproduced_source)
    assert len(violations) == len(observed)
    assert all(
        "removed verified-row capability" in violation for violation in violations
    )
