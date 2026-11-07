# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The Tests (Split) job must not fall back to a floating hosted image (OMN-22812).

``ubuntu-latest`` became Ubuntu 26.04 on 2026-11-06. That image carries no
PostgreSQL 16 binaries, so the ephemeral-PostgreSQL tests skip there, and Skip
Count Ratchet (OMN-18776) reads the new skips as growth of the skipped set. The
``test-parallel`` job resolves its runner through the OMNI_RUNNER_SELECTOR_V1
expression, whose last fallback is the label used while the variables stay
unset. That fallback is what this module pins.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PINNED_IMAGE = "ubuntu-24.04"
FALLBACK_LITERAL = re.compile(r"""'\[\s*"([^"]+)"\s*\]'""")


def fallback_labels(runs_on: str) -> list[str]:
    """Every JSON-literal fallback label named in a selector expression."""
    return FALLBACK_LITERAL.findall(runs_on)


def floating_or_unpinned(runs_on: str) -> list[str]:
    """Fallback labels that are not the pinned image."""
    return [label for label in fallback_labels(runs_on) if label != PINNED_IMAGE]


def _test_parallel_runs_on() -> str:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    runs_on = workflow["jobs"]["test-parallel"]["runs-on"]
    assert isinstance(runs_on, str), runs_on
    return runs_on


def test_the_probe_names_a_floating_fallback() -> None:
    """Control: the selector as it stood on dev is named by the probe."""
    on_dev = (
        "${{ (github.event_name == 'pull_request') "
        "&& fromJSON(vars.OMNI_PUBLIC_PR_RUNS_ON_JSON || '[\"ubuntu-latest\"]') "
        "|| fromJSON(vars.OMNI_TRUSTED_CI_RUNS_ON_JSON || '[\"ubuntu-latest\"]') }}"
    )
    assert floating_or_unpinned(on_dev) == ["ubuntu-latest", "ubuntu-latest"]


def test_the_probe_reads_both_fallbacks_of_the_real_selector() -> None:
    """The probe finds the two fallbacks of the real expression, so a pass is not an empty read."""
    assert len(fallback_labels(_test_parallel_runs_on())) == 2


def test_test_parallel_falls_back_to_the_pinned_image() -> None:
    """Neither fallback of the Tests (Split) selector floats with ubuntu-latest."""
    assert floating_or_unpinned(_test_parallel_runs_on()) == []
