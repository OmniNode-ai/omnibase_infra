# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18490 — the ticked-open decision survives to the receipt JSON.

Drives the handler, serialises the result as receipt mode does, and parses it
back: a started ticket with every box ticked is named in the receipt, counted
in ``tickets_ticked_open`` and never flipped to Done.
"""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_result import (
    ModelEvidenceAutocloseSweepResult,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect.test_handler_evidence_autoclose_sweep import (
    FakeLinearClient,
    _issue,
    _request,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect.test_omn_18490_ticked_open import (
    _CHECKED,
    _TICKET,
    _handler,
)

pytestmark = pytest.mark.integration


async def test_ticked_open_ticket_is_named_in_the_round_tripped_receipt() -> None:
    linear = FakeLinearClient(issues={_TICKET: _issue(description=_CHECKED)})
    result = await _handler(linear).handle(_request(apply=True))

    receipt = result.model_dump(mode="json")
    parsed = ModelEvidenceAutocloseSweepResult.model_validate(receipt)

    assert receipt["tickets_ticked_open"] == 1
    assert parsed.tickets_flipped == 0
    assert [(o.ticket_id, o.decision.value) for o in parsed.outcomes] == [
        (_TICKET, "gap_ticked_open")
    ]
    assert linear.state_updates == []
