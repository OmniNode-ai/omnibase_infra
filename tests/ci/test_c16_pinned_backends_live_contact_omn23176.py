# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A recorded pinned delegation through the C16 pinned-backend reader (OMN-23176).

The artifact is a real receipt of ``onex delegate --backend-id`` run on the lab's
deployed dev lane (provenance in the file), not a dict written to match the
reader.
"""

from __future__ import annotations

from typing import Any

import pytest

from omnibase_infra.handlers import handler_c16_pinned_backend as probe


@pytest.mark.unit
@pytest.mark.live_contact("tests/fixtures/omn23176/pinned_local_receipt.json")
def test_a_recorded_pinned_receipt_reduces_to_the_accepted_attempt_the_row_is_compared_with(
    recorded_response: dict[str, Any],
) -> None:
    run = probe.pinned_run_from_receipt(
        "local", recorded_response["requested_backend_id"], recorded_response, None
    )
    assert run.accepted is not None
    assert run.accepted["backend_id"] == recorded_response["requested_backend_id"]
    assert run.accepted["host"]
    assert run.terminal_cost_usd == recorded_response["terminal_cost_usd"]
    assert run.correlation_id == recorded_response["correlation_id"]
