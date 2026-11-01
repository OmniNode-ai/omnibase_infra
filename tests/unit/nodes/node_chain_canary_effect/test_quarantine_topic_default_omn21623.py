# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The canary's quarantine topic default is the contract-derived registry member (OMN-21623)."""

from __future__ import annotations

import pytest

from omnibase_infra.enums.generated.enum_omnibase_infra_topic import (
    EnumOmnibaseInfraTopic,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_request import (
    ModelChainCanaryRequest,
)


@pytest.mark.unit
def test_quarantine_topic_default_is_the_registry_member() -> None:
    field = ModelChainCanaryRequest.model_fields["quarantine_topic"]
    assert field.default == EnumOmnibaseInfraTopic.DLQ_QUARANTINE_V1.value
    assert field.default.startswith("onex.dlq.")
