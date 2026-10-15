# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20772: the post-merge event is built from a recorded read of a live lane.

The recording is the output of ``lab_pass_receipt.py lane-identity`` taken
against the running dev-lane containers and their migration ledger, not a dict
written to look like what the reader returns.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from omnibase_infra.models.lab_pass import ModelPostMergeLabPassReceiptEvent
from scripts.ci import lab_pass_receipt as receipts

pytestmark = pytest.mark.unit


@pytest.mark.live_contact("tests/fixtures/omn20772/lane_identity_recorded.json")
def test_post_merge_receipt_event_binds_a_recorded_live_lane_identity(
    recorded_response: dict[str, object],
) -> None:
    identity = receipts.parse_post_merge_identity(recorded_response["identity"])
    receipt = receipts.build_receipt(
        sha=identity.generation.revision,
        lane=receipts.EnumLabLane.COMPOSE_DEV,
        started_at=datetime(2026, 10, 15, 10, 0, tzinfo=UTC),
        finished_at=datetime(2026, 10, 15, 10, 1, tzinfo=UTC),
        checks=[receipts.ModelLabPassCheck(name="main_ready", ok=True, evidence="200")],
        agent_command_id=None,
    )

    event = ModelPostMergeLabPassReceiptEvent.model_validate(
        receipts.build_post_merge_event(receipt, identity)
    )

    assert event.verdict == "PASS"
    assert event.binding_valid is True
    assert event.generation.revision == identity.generation.revision
    assert event.identity.migration_fingerprint.startswith("sha256:")
    assert set(event.identity.image_ids) == set(dict(identity.image_ids))
    assert event.agent_command_id is None
