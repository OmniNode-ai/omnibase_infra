# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime DLQ topic provisioning coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor

pytestmark = pytest.mark.integration


def test_canonical_dlq_topics_are_in_service_manifest() -> None:
    """Contract-first provisioning must include non-contract runtime DLQ topics."""
    repo_root = Path(__file__).resolve().parents[3]
    extractor = ContractTopicExtractor()

    entries = extractor.extract_all(
        contracts_root=repo_root / "src" / "omnibase_infra" / "nodes",
        skill_manifests_roots=[repo_root / "src" / "omnibase_infra" / "services"],
    )
    topics = {entry.topic for entry in entries}

    assert "onex.dlq.intents.v1" in topics
    assert "onex.dlq.events.v1" in topics
    assert "onex.dlq.commands.v1" in topics
