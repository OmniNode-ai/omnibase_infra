# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14186: the pinned onex_change_control schema admits ``binds_ac``.

``validate-contract@main`` read onex_change_control ``main``, whose schema
predates ``binds_ac`` and refused every contract that repo-evidence /
dod-verify requires, so no PR with declared acceptance criteria could pass
both. The recording is the real validate-yaml output of the sibling-pin commit
and of ``main`` against this ticket's own contract. The workflow no longer
reads onex_change_control at all (OMN-20074: it validates with omnibase_core's
ModelTicketContract), so the recording stays as the history of that fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/contract-validation.yml"
PINS = cast(
    "dict[str, str]",
    yaml.safe_load((ROOT / ".github/sibling-pins.yaml").read_text())["pins"],
)


@pytest.mark.live_contact(
    "tests/ci/fixtures/contract_validation_pinned_validator_omn14186.json"
)
def test_the_pinned_validator_admits_binds_ac_that_main_refuses(
    recorded_response: dict[str, object],
) -> None:
    runs = cast("dict[str, dict[str, object]]", recorded_response["response"])
    pinned, main = runs["pinned"], runs["main"]
    assert pinned["onex_change_control_sha"] == PINS["onex_change_control"]
    assert pinned["exit_code"] == 0
    assert main["exit_code"] == 1
    assert any(
        "binds_ac: Extra inputs are not permitted" in line
        for line in cast("list[str]", main["output"])
    )


def test_the_workflow_no_longer_calls_the_floating_composite_action() -> None:
    assert "validate-contract@" not in WORKFLOW.read_text(encoding="utf-8")
