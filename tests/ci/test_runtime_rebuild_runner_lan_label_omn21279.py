# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-rebuild publishers require a label h203-light lacks (OMN-21279)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
H203_LIGHT_LABELS = {"self-hosted", "omnibase-ci", "h203-light", "linux", "x64"}
FALLBACK = re.compile(r"vars\.OMNI_RUNTIME_REBUILD_RUNS_ON_JSON\s*\|\|\s*'([^']+)'")


def _fallback_labels(runs_on: str) -> list[str]:
    match = FALLBACK.search(runs_on)
    assert match is not None, "publisher must declare its dedicated runner fallback"
    return json.loads(match.group(1))


def _can_land_on_h203_light(labels: list[str]) -> bool:
    return {label.lower() for label in labels}.issubset(
        label.lower() for label in H203_LIGHT_LABELS
    )


@pytest.mark.parametrize(
    ("workflow_file", "job_id"),
    [
        ("runtime-rebuild-trigger.yml", "trigger-rebuild"),
        ("runtime-rebuild-trigger-reusable.yml", "trigger-rebuild"),
        ("onex-api-lab-delivery-reusable.yml", "trigger-delivery"),
    ],
)
def test_publisher_fallback_requires_a_lan_label(
    workflow_file: str, job_id: str
) -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / workflow_file).read_text(
            encoding="utf-8"
        )
    )
    runs_on = workflow["jobs"][job_id]["runs-on"]
    assert isinstance(runs_on, str)
    labels = _fallback_labels(runs_on)
    assert "arch-amd64" in labels
    assert not _can_land_on_h203_light(labels), (
        "OMN-21279: a fallback whose labels are a subset of h203-light's labels "
        "can land off-LAN on a runner with no uv"
    )


def test_h203_light_landability_positive_control() -> None:
    old = _fallback_labels(
        """${{ fromJSON(vars.OMNI_RUNTIME_REBUILD_RUNS_ON_JSON || '["self-hosted","omnibase-ci"]') }}"""
    )
    pinned = _fallback_labels(
        """${{ fromJSON(vars.OMNI_RUNTIME_REBUILD_RUNS_ON_JSON || '["self-hosted","omnibase-ci","arch-amd64"]') }}"""
    )
    assert _can_land_on_h203_light(old)
    assert _can_land_on_h203_light([label.upper() for label in old])
    assert not _can_land_on_h203_light(pinned)
