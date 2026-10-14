# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The omniintelligence pin carries the larger retry budget (OMN-17427).

The review voter asks for 4096 tokens. When a reply does not parse as review
JSON it retries once, and before omniintelligence#1072 that retry got the same
4096-token cap, so a reply cut off at the cap failed the vote again and the
verdict degraded to ``degraded_quorum``. omniintelligence#1072 gives the retry
8192 tokens. The gate runs the omniintelligence this repository's workflows
clone, so the fix reaches this repository's Hostile Review Gate only when the
sibling pin names a revision that carries it.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_SIBLING_PINS = Path(__file__).resolve().parents[2] / ".github" / "sibling-pins.yaml"
# The merge commit of omniintelligence#1072 on its dev branch.
RETRY_BUDGET_PIN = "3fcbf795087a79081c0ceeb676216bf64c5d49ae"
# The pin before this change, which retries at the same 4096-token cap.
PRE_RETRY_BUDGET_PIN = "64b950973a5686d33e0b439fb80749d66362d06d"


def omniintelligence_pin(sibling_pins_text: str) -> str:
    pins = yaml.safe_load(sibling_pins_text)["pins"]
    return cast("str", pins["omniintelligence"])


def test_omniintelligence_is_pinned_to_the_retry_budget_revision() -> None:
    text = _SIBLING_PINS.read_text(encoding="utf-8")
    assert omniintelligence_pin(text) == RETRY_BUDGET_PIN


def test_the_pin_before_the_retry_budget_is_caught() -> None:
    """Positive control: the sibling pins as they were before this change."""
    before = _SIBLING_PINS.read_text(encoding="utf-8").replace(
        RETRY_BUDGET_PIN, PRE_RETRY_BUDGET_PIN
    )
    assert omniintelligence_pin(before) == PRE_RETRY_BUDGET_PIN
    assert omniintelligence_pin(before) != RETRY_BUDGET_PIN


@pytest.mark.live_contact(
    "tests/ci/fixtures/hostile_reviewer_overlay_schema_pin_omn20947.json"
)
def test_pinned_omniintelligence_retries_with_the_larger_token_budget(
    recorded_response: dict[str, object],
) -> None:
    """The recording is the pinned revision's own retry budget, read on a lab host."""
    provenance = cast("dict[str, str]", recorded_response["_provenance"])
    previous = provenance["retry_budget_previous_revision"]
    budgets = cast(
        "dict[str, dict[str, int | None]]", recorded_response["retry_budget"]
    )
    pin = omniintelligence_pin(_SIBLING_PINS.read_text(encoding="utf-8"))
    assert budgets[pin]["default_max_tokens"] == 4096
    assert budgets[pin]["retry_max_tokens"] == 8192
    # Control: the revision before it retries with no budget of its own.
    assert previous == PRE_RETRY_BUDGET_PIN
    assert budgets[previous]["retry_max_tokens"] is None
