# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The flip proofs of the three reconciliation nodes passed the seam gate (OMN-21053).

The recording is the output of omnibase_core's ``verify_flip_bundle.py`` run over
the committed receipts, one run per flipped node, with the base ref the parity
tests were held red against.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

_RECEIPTS = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "adequacy_receipts"


@pytest.mark.live_contact("tests/ci/fixtures/omn21053_flip_bundle_live.json")
def test_every_flipped_reconciliation_node_passed_the_flip_bundle_gate(
    recorded_response: dict[str, object],
) -> None:
    runs = cast("list[dict[str, object]]", recorded_response["runs"])
    assert len(runs) == 3
    for run in runs:
        node_id = str(run["node_id"])
        assert run["exit_code"] == 0, node_id
        verdicts = cast("list[str]", run["verdict_lines"])
        assert any("all seven flip-bundle assertions hold" in v for v in verdicts), (
            node_id
        )
        # The recorded node is one whose proof is committed in this repository.
        assert (_RECEIPTS / f"{node_id}.handflip.json").is_file(), node_id
        assert (_RECEIPTS / f"{node_id}.json").is_file(), node_id
