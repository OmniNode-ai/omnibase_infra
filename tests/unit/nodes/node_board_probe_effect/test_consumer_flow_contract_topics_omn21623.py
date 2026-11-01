# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The board probe's exposure topic and seam DLQ come from its contract (OMN-21623)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.nodes.node_board_probe_effect.handlers import (
    _consumer_flow_constants as constants,
)

_CONTRACT = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_board_probe_effect"
    / "contract.yaml"
)


def _declared() -> dict[str, str]:
    return yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))["consumer_flow_topics"]


@pytest.mark.unit
class TestConsumerFlowContractTopics:
    def test_exposure_topic_is_the_declared_snapshot_topic(self) -> None:
        assert _declared()["exposure"] == constants.EXPOSURE_TOPIC
        assert constants.EXPOSURE_TOPIC.startswith("onex.snapshot.")

    def test_seam_dlq_is_the_declared_dlq_topic(self) -> None:
        assert _declared()["seam_dlq"] == constants.SEAM_DLQ
        assert constants.SEAM_DLQ.startswith("onex.dlq.")

    def test_a_name_declared_under_another_family_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no dlq topic for exposure"):
            constants._contract_topic("exposure", "dlq")

    def test_an_undeclared_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no snapshot topic for missing"):
            constants._contract_topic("missing", "snapshot")
