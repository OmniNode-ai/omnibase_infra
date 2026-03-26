# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
#
# Tests for topic constants vs generated enum coverage (OMN-3254).
#
# Verifies that every topic string in topic_constants.py is represented
# in the generated enum files, ensuring the CONTRACT_DRIFT gap is closed.

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor

# Paths are relative to repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOPIC_CONSTANTS = (
    _REPO_ROOT / "src" / "omnibase_infra" / "event_bus" / "topic_constants.py"
)
_CONTRACTS_ROOT = _REPO_ROOT / "src" / "omnibase_infra" / "nodes"
_GENERATED_DIR = _REPO_ROOT / "src" / "omnibase_infra" / "enums" / "generated"


@pytest.mark.unit
def test_all_topic_constants_covered_by_generated_enums() -> None:
    """Every topic in topic_constants.py appears in the generated enum set.

    This test is the runtime equivalent of the --check CI invariant.
    It ensures that the CONTRACT_DRIFT gap (OMN-3254) does not regress.
    """
    if not _TOPIC_CONSTANTS.exists():
        pytest.skip("topic_constants.py not found")
    if not _CONTRACTS_ROOT.exists():
        pytest.skip("contracts root not found")

    extractor = ContractTopicExtractor()

    # Get topics from topic_constants.py
    constant_entries = extractor.extract_from_python_sources([_TOPIC_CONSTANTS])
    constant_topics = {e.topic for e in constant_entries}

    # Get topics from the full pipeline (contracts + supplementary)
    all_entries = extractor.extract_all(
        _CONTRACTS_ROOT, supplementary_sources=[_TOPIC_CONSTANTS]
    )
    all_topics = {e.topic for e in all_entries}

    # Every constant topic must be in the full set
    missing = constant_topics - all_topics
    assert not missing, (
        f"Topic constants NOT covered by generated enums: {missing}\n"
        f"Run: uv run python scripts/generate_topic_enums.py --generate"
    )


@pytest.mark.unit
def test_generated_enum_files_importable() -> None:
    """Generated enum files can be imported without errors.

    Note: EnumOmniclaudeTopic and EnumOmniintelligenceTopic were consolidated
    into ProtocolTopicRegistry in OMN-5839. Only remaining generated enums are
    tested here.
    """
    from omnibase_infra.enums.generated import (
        EnumOmnibaseInfraTopic,
        EnumPlatformTopic,
    )

    # Verify known topic constants are members of their producer enums
    assert (
        EnumOmnibaseInfraTopic.EVT_CONSUMER_HEALTH_V1
        == "onex.evt.omnibase-infra.consumer-health.v1"
    )
    assert (
        EnumPlatformTopic.EVT_CONTRACT_REGISTERED_V1
        == "onex.evt.platform.contract-registered.v1"
    )
