# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for standalone consumer config session timeout fields (OMN-5445).

Verifies that all 6 standalone consumer configs have session_timeout_ms and
heartbeat_interval_ms fields with correct defaults (45000/15000).

Updated in OMN-6066..OMN-6072: defaults raised from 30000/10000 to 45000/15000
to prevent rebalance storms during brief processing delays.
"""

from __future__ import annotations

import pytest

from omnibase_infra.services.observability.agent_actions.config import (
    ConfigAgentActionsConsumer,
)
from omnibase_infra.services.observability.context_audit.config import (
    ConfigContextAuditConsumer,
)
from omnibase_infra.services.observability.injection_effectiveness.config import (
    ConfigInjectionEffectivenessConsumer,
)
from omnibase_infra.services.observability.llm_cost_aggregation.config import (
    ConfigLlmCostAggregation,
)
from omnibase_infra.services.observability.skill_lifecycle.config import (
    ConfigSkillLifecycleConsumer,
)
from omnibase_infra.services.session.config_consumer import ConfigSessionConsumer

# All standalone consumer config classes with their names for parametrized tests
_STANDALONE_CONFIGS = [
    ConfigSessionConsumer,
    ConfigAgentActionsConsumer,
    ConfigInjectionEffectivenessConsumer,
    ConfigLlmCostAggregation,
    ConfigSkillLifecycleConsumer,
    ConfigContextAuditConsumer,
]


class TestStandaloneConsumerConfigDefaults:
    """Test that all standalone consumer configs have session timeout fields."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "config_cls",
        _STANDALONE_CONFIGS,
        ids=[c.__name__ for c in _STANDALONE_CONFIGS],
    )
    def test_has_session_timeout_ms_field(self, config_cls: type) -> None:
        """Config class must have a session_timeout_ms field."""
        fields = config_cls.model_fields
        assert "session_timeout_ms" in fields, (
            f"{config_cls.__name__} missing session_timeout_ms field"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "config_cls",
        _STANDALONE_CONFIGS,
        ids=[c.__name__ for c in _STANDALONE_CONFIGS],
    )
    def test_has_heartbeat_interval_ms_field(self, config_cls: type) -> None:
        """Config class must have a heartbeat_interval_ms field."""
        fields = config_cls.model_fields
        assert "heartbeat_interval_ms" in fields, (
            f"{config_cls.__name__} missing heartbeat_interval_ms field"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "config_cls",
        _STANDALONE_CONFIGS,
        ids=[c.__name__ for c in _STANDALONE_CONFIGS],
    )
    def test_session_timeout_ms_default_is_45000(self, config_cls: type) -> None:
        """session_timeout_ms default must be 45000 (raised from 30000 in OMN-6066..OMN-6072)."""
        field_info = config_cls.model_fields["session_timeout_ms"]
        assert field_info.default == 45000, (
            f"{config_cls.__name__}.session_timeout_ms default is "
            f"{field_info.default}, expected 45000"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "config_cls",
        _STANDALONE_CONFIGS,
        ids=[c.__name__ for c in _STANDALONE_CONFIGS],
    )
    def test_heartbeat_interval_ms_default_is_15000(self, config_cls: type) -> None:
        """heartbeat_interval_ms default must be 15000 (raised from 10000 in OMN-6066..OMN-6072)."""
        field_info = config_cls.model_fields["heartbeat_interval_ms"]
        assert field_info.default == 15000, (
            f"{config_cls.__name__}.heartbeat_interval_ms default is "
            f"{field_info.default}, expected 15000"
        )
