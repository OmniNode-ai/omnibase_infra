# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for OMN-18362's superseded-criterion live set.

The unit tests prove the parser edges. This integration test keeps the PR under
the repository's feature-code coverage gate by exercising the committed scrubbed
Linear history capture through the same live-set and coverage-gap functions the
autoclose sweep uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    _ac_coverage_gap,
    _canonical_ac_label,
    _live_acceptance_criteria_items,
)

pytestmark = pytest.mark.integration

_FIXTURE = (
    Path(__file__).parents[2]
    / "fixtures"
    / "autoclose"
    / "omn_18362_omn_18035_document_content_history.json"
)


def _capture() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_superseded_duplicate_is_not_a_live_criterion_for_coverage_gap() -> None:
    description = _capture()["issue"]["description"]

    live_items = _live_acceptance_criteria_items(description)

    assert [_canonical_ac_label(item) for item in live_items] == [
        "AC1",
        "AC2",
        "AC3",
        "AC4",
    ]

    reason, uncovered = _ac_coverage_gap(
        description,
        verified_count=4,
        coverage_verified_count=4,
        coverage_non_probative_count=0,
    )

    assert reason == ""
    assert uncovered == ()
