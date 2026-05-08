# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for shadow mode in bifrost delegation config — OMN-10638.

DoD evidence (test_pass type):
    - test_shadow_disabled_by_default: shadow_mode.enabled=False by default
    - test_shadow_does_not_alter_live_routing_output: shadow recommendation
      does NOT change the backend_selected in the live routing response
    - test_shadow_mode_toggled_via_config: shadow_mode.enabled can be set True
    - test_delegation_config_carries_shadow_config: ModelBifrostDelegationConfig
      exposes a shadow_mode field
    - test_canonical_config_shadow_disabled: canonical bifrost_delegation.yaml
      has shadow_mode.enabled=False (safe default for demo)
    - test_shadow_label_in_comparison_event: comparison events carry label=SHADOW

Related:
    - OMN-10638: Shadow mode for delegation A/B testing
    - OMN-10604: Delegation Routing Production Feature Plan
"""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import yaml

from omnibase_infra.enums import EnumLlmFinishReason, EnumLlmOperationType
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport
from omnibase_infra.models.llm.model_llm_inference_response import (
    ModelLlmInferenceResponse,
)
from omnibase_infra.models.llm.model_llm_usage import ModelLlmUsage
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost import (
    HandlerBifrostGateway,
    ModelBifrostConfig,
    ModelBifrostRequest,
    ModelBifrostShadowConfig,
    ModelShadowDecisionLog,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.config_loader_bifrost_delegation import (
    load_bifrost_delegation_config,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_config import (
    ModelBifrostBackendConfig,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_delegation_config import (
    ModelBifrostDelegationConfig,
    ModelDelegationShadowConfig,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)

pytestmark = pytest.mark.unit

_TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")
_CONFIGS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent.parent.parent
    / "src"
    / "omnibase_infra"
    / "configs"
)
_CANONICAL_CONFIG = _CONFIGS_DIR / "bifrost_delegation.yaml"


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_inference_response() -> ModelLlmInferenceResponse:
    from datetime import UTC, datetime

    return ModelLlmInferenceResponse(
        generated_text="Hello from bifrost",
        model_used="test-model",
        operation_type=EnumLlmOperationType.CHAT_COMPLETION,
        finish_reason=EnumLlmFinishReason.STOP,
        usage=ModelLlmUsage(),
        latency_ms=50.0,
        backend_result=ModelBackendResult(success=True, duration_ms=50.0),
        correlation_id=uuid4(),
        execution_id=uuid4(),
        timestamp=datetime.now(UTC),
    )


def _make_gateway_with_shadow(
    shadow_config: ModelBifrostShadowConfig,
    shadow_policy: object | None = None,
    shadow_callback: object | None = None,
) -> HandlerBifrostGateway:
    transport = MagicMock(spec=MixinLlmHttpTransport)
    inference_handler = HandlerLlmOpenaiCompatible(transport)
    inference_handler.handle = AsyncMock(return_value=_make_inference_response())

    config = ModelBifrostConfig(
        backends={
            "backend-a": ModelBifrostBackendConfig(
                backend_id="backend-a",
                base_url="http://backend-a:8000",
            ),
            "backend-b": ModelBifrostBackendConfig(
                backend_id="backend-b",
                base_url="http://backend-b:8000",
            ),
        },
        default_backends=("backend-a",),
    )
    return HandlerBifrostGateway(
        config=config,
        inference_handler=inference_handler,
        shadow_config=shadow_config,
        shadow_policy=shadow_policy,
        shadow_decision_callback=shadow_callback,
    )


class MockShadowPolicy:
    """Mock shadow policy that recommends a configurable backend."""

    def __init__(self, recommended: str = "backend-b") -> None:
        self.recommended = recommended
        self.call_count = 0

    async def recommend(
        self,
        request: ModelBifrostRequest,
        available_backends: tuple[str, ...],
    ) -> tuple[str, float, dict[str, float]]:
        self.call_count += 1
        return self.recommended, 0.9, {self.recommended: 0.9, "backend-a": 0.1}


# ── ModelDelegationShadowConfig tests ────────────────────────────────────


class TestModelDelegationShadowConfig:
    """ModelDelegationShadowConfig is the delegation-specific shadow config model."""

    def test_default_disabled(self) -> None:
        """Shadow mode must be disabled by default — safe for demo."""
        config = ModelDelegationShadowConfig()
        assert config.enabled is False

    def test_can_enable(self) -> None:
        """Shadow mode can be enabled via config."""
        config = ModelDelegationShadowConfig(enabled=True, policy_version="v1.0.0")
        assert config.enabled is True
        assert config.policy_version == "v1.0.0"

    def test_comparison_logging_enabled_by_default(self) -> None:
        """comparison_logging_enabled defaults to True when shadow is configured."""
        config = ModelDelegationShadowConfig()
        assert config.comparison_logging_enabled is True

    def test_shadow_label_constant(self) -> None:
        """shadow_label is 'SHADOW' — required for dashboard/eval labeling."""
        config = ModelDelegationShadowConfig()
        assert config.shadow_label == "SHADOW"

    def test_shadow_label_immutable(self) -> None:
        """shadow_label cannot be changed — it is the canonical SHADOW label."""
        with pytest.raises(Exception):
            ModelDelegationShadowConfig(shadow_label="NOT_SHADOW")  # type: ignore[call-arg]


# ── ModelBifrostDelegationConfig shadow_mode field ──────────────────────


class TestDelegationConfigCarriesShadowConfig:
    """ModelBifrostDelegationConfig must expose a shadow_mode field."""

    def test_delegation_config_has_shadow_mode(self, tmp_path: Path) -> None:
        """ModelBifrostDelegationConfig carries a shadow_mode field."""
        minimal = {
            "config_version": "0.1.0",
            "schema_version": "bifrost_delegation.v1",
            "backends": [
                {
                    "backend_id": "test-backend",
                    "model_name": "test-model",
                    "tier": "local",
                }
            ],
            "routing_rules": [
                {
                    "rule_id": "aaaaaaaa-0001-4000-8000-000000000001",
                    "task_class": "code_generation",
                    "task_class_contract_version": "1.0.0",
                    "backend_policy_version": "1.0.0",
                    "backend_ids": ["test-backend"],
                    "fallback_policy": {"action": "return_error"},
                    "shadow_policy_id": "bbbbbbbb-0001-4000-8000-000000000001",
                }
            ],
        }
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text(yaml.dump(minimal), encoding="utf-8")
        config = load_bifrost_delegation_config(config_file)
        assert hasattr(config, "shadow_mode")
        assert isinstance(config.shadow_mode, ModelDelegationShadowConfig)

    def test_shadow_mode_disabled_by_default_in_loaded_config(
        self, tmp_path: Path
    ) -> None:
        """Loaded config defaults shadow_mode.enabled=False when not specified in YAML."""
        minimal = {
            "config_version": "0.1.0",
            "schema_version": "bifrost_delegation.v1",
            "backends": [
                {
                    "backend_id": "test-backend",
                    "model_name": "test-model",
                    "tier": "local",
                }
            ],
            "routing_rules": [
                {
                    "rule_id": "aaaaaaaa-0001-4000-8000-000000000001",
                    "task_class": "code_generation",
                    "task_class_contract_version": "1.0.0",
                    "backend_policy_version": "1.0.0",
                    "backend_ids": ["test-backend"],
                    "fallback_policy": {"action": "return_error"},
                    "shadow_policy_id": "bbbbbbbb-0001-4000-8000-000000000001",
                }
            ],
        }
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text(yaml.dump(minimal), encoding="utf-8")
        config = load_bifrost_delegation_config(config_file)
        assert config.shadow_mode.enabled is False

    def test_shadow_mode_can_be_set_in_yaml(self, tmp_path: Path) -> None:
        """shadow_mode section in YAML enables shadow mode via config."""
        config_data = {
            "config_version": "0.1.0",
            "schema_version": "bifrost_delegation.v1",
            "backends": [
                {
                    "backend_id": "test-backend",
                    "model_name": "test-model",
                    "tier": "local",
                }
            ],
            "routing_rules": [
                {
                    "rule_id": "aaaaaaaa-0001-4000-8000-000000000001",
                    "task_class": "code_generation",
                    "task_class_contract_version": "1.0.0",
                    "backend_policy_version": "1.0.0",
                    "backend_ids": ["test-backend"],
                    "fallback_policy": {"action": "return_error"},
                    "shadow_policy_id": "bbbbbbbb-0001-4000-8000-000000000001",
                }
            ],
            "shadow_mode": {
                "enabled": True,
                "policy_version": "v1.0.0-test",
                "log_sample_rate": 1.0,
            },
        }
        config_file = tmp_path / "shadow_enabled.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")
        config = load_bifrost_delegation_config(config_file)
        assert config.shadow_mode.enabled is True
        assert config.shadow_mode.policy_version == "v1.0.0-test"


# ── Canonical config tests ────────────────────────────────────────────────


class TestCanonicalConfigShadowDefault:
    """The canonical bifrost_delegation.yaml must have shadow_mode disabled by default."""

    def test_canonical_config_shadow_disabled(self) -> None:
        """Canonical config has shadow_mode.enabled=False (safe default for demo)."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        assert config.shadow_mode.enabled is False, (
            "bifrost_delegation.yaml must have shadow_mode.enabled=False by default "
            "to prevent accidental learned-policy activation during demo (OMN-10638)"
        )

    def test_canonical_config_shadow_label_is_shadow(self) -> None:
        """Canonical config shadow_label is 'SHADOW'."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        assert config.shadow_mode.shadow_label == "SHADOW"


# ── DoD: shadow does not alter live routing ───────────────────────────────


class TestShadowDoesNotAlterLiveRouting:
    """Key DoD: shadow decisions must NEVER affect live routing output."""

    @pytest.mark.asyncio
    async def test_shadow_does_not_alter_live_routing_output(self) -> None:
        """Shadow recommending a different backend does NOT change backend_selected."""
        # Policy recommends backend-b; static config only has backend-a as default
        policy = MockShadowPolicy(recommended="backend-b")
        callback = AsyncMock()
        shadow_config = ModelBifrostShadowConfig(enabled=True, policy_version="v1-test")

        gateway = _make_gateway_with_shadow(
            shadow_config=shadow_config,
            shadow_policy=policy,
            shadow_callback=callback,
        )
        request = ModelBifrostRequest(
            operation_type=EnumLlmOperationType.CHAT_COMPLETION,
            tenant_id=_TENANT_ID,
            messages=({"role": "user", "content": "test"},),
        )

        response = await gateway.handle(request)

        # Live routing must use backend-a (the only configured default)
        assert response.success is True
        assert response.backend_selected == "backend-a", (
            f"Shadow mode altered live routing: expected 'backend-a' "
            f"but got '{response.backend_selected}'. "
            "Shadow decisions MUST NOT affect live routing output (OMN-10638)."
        )

        # Wait for fire-and-forget shadow task
        await asyncio.sleep(0.05)
        # Shadow policy was invoked
        assert policy.call_count == 1

        # Shadow callback received a comparison event
        callback.assert_called_once()
        log_entry: ModelShadowDecisionLog = callback.call_args[0][0]

        # Live routing is unchanged
        assert log_entry.static_backend_selected == "backend-a"
        # Shadow recommended a different backend
        assert log_entry.shadow_backend_recommended == "backend-b"
        assert log_entry.agreed is False

    @pytest.mark.asyncio
    async def test_shadow_disabled_does_not_run_policy(self) -> None:
        """With shadow_mode.enabled=False, policy is never invoked."""
        policy = MockShadowPolicy(recommended="backend-b")
        shadow_config = ModelBifrostShadowConfig(enabled=False)

        gateway = _make_gateway_with_shadow(
            shadow_config=shadow_config,
            shadow_policy=policy,
        )
        request = ModelBifrostRequest(
            operation_type=EnumLlmOperationType.CHAT_COMPLETION,
            tenant_id=_TENANT_ID,
            messages=({"role": "user", "content": "test"},),
        )

        response = await gateway.handle(request)
        assert response.success is True
        await asyncio.sleep(0.05)
        assert policy.call_count == 0

    @pytest.mark.asyncio
    async def test_shadow_label_appears_in_comparison_event(self) -> None:
        """Comparison events carry the SHADOW label from delegation shadow config."""
        policy = MockShadowPolicy(recommended="backend-a")
        callback = AsyncMock()
        shadow_config = ModelBifrostShadowConfig(enabled=True, policy_version="v1-test")

        gateway = _make_gateway_with_shadow(
            shadow_config=shadow_config,
            shadow_policy=policy,
            shadow_callback=callback,
        )
        request = ModelBifrostRequest(
            operation_type=EnumLlmOperationType.CHAT_COMPLETION,
            tenant_id=_TENANT_ID,
            messages=({"role": "user", "content": "test"},),
        )

        await gateway.handle(request)
        await asyncio.sleep(0.05)

        callback.assert_called_once()
        log_entry: ModelShadowDecisionLog = callback.call_args[0][0]
        assert log_entry.policy_version == "v1-test"
        assert log_entry.shadow_label == ModelDelegationShadowConfig().shadow_label

    @pytest.mark.asyncio
    async def test_multiple_requests_shadow_never_bleeds_into_live(self) -> None:
        """Across multiple requests, shadow never modifies live backend selection."""
        policy = MockShadowPolicy(recommended="backend-b")
        callback = AsyncMock()
        shadow_config = ModelBifrostShadowConfig(enabled=True, policy_version="v1")

        gateway = _make_gateway_with_shadow(
            shadow_config=shadow_config,
            shadow_policy=policy,
            shadow_callback=callback,
        )
        request = ModelBifrostRequest(
            operation_type=EnumLlmOperationType.CHAT_COMPLETION,
            tenant_id=_TENANT_ID,
            messages=({"role": "user", "content": "test"},),
        )

        responses = [await gateway.handle(request) for _ in range(5)]
        await asyncio.sleep(0.1)

        for resp in responses:
            assert resp.backend_selected == "backend-a", (
                "Shadow mode leaked into live routing on repeated calls."
            )

        assert policy.call_count == 5


# ── YAML shadow_mode section loading ─────────────────────────────────────


class TestShadowModeYamlSection:
    """shadow_mode YAML section controls delegation shadow mode."""

    def test_shadow_mode_fields_validated(self, tmp_path: Path) -> None:
        """Invalid log_sample_rate is rejected by Pydantic."""
        config_data = {
            "config_version": "0.1.0",
            "schema_version": "bifrost_delegation.v1",
            "backends": [
                {
                    "backend_id": "test-backend",
                    "model_name": "test-model",
                    "tier": "local",
                }
            ],
            "routing_rules": [
                {
                    "rule_id": "aaaaaaaa-0001-4000-8000-000000000001",
                    "task_class": "code_generation",
                    "task_class_contract_version": "1.0.0",
                    "backend_policy_version": "1.0.0",
                    "backend_ids": ["test-backend"],
                    "fallback_policy": {"action": "return_error"},
                    "shadow_policy_id": "bbbbbbbb-0001-4000-8000-000000000001",
                }
            ],
            "shadow_mode": {
                "enabled": True,
                "log_sample_rate": 2.0,  # invalid — must be <= 1.0
            },
        }
        config_file = tmp_path / "bad_sample_rate.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")
        with pytest.raises(ValueError):
            load_bifrost_delegation_config(config_file)

    def test_shadow_mode_omitted_gives_safe_default(self, tmp_path: Path) -> None:
        """Config without shadow_mode section is safe (disabled by default)."""
        config_data = textwrap.dedent("""\
            config_version: "0.1.0"
            schema_version: "bifrost_delegation.v1"
            backends:
              - backend_id: test-backend
                model_name: test-model
                tier: local
            routing_rules:
              - rule_id: "aaaaaaaa-0001-4000-8000-000000000001"
                task_class: code_generation
                task_class_contract_version: "1.0.0"
                backend_policy_version: "1.0.0"
                backend_ids: [test-backend]
                fallback_policy:
                  action: return_error
                shadow_policy_id: "bbbbbbbb-0001-4000-8000-000000000001"
        """)
        config_file = tmp_path / "no_shadow.yaml"
        config_file.write_text(config_data, encoding="utf-8")
        config = load_bifrost_delegation_config(config_file)
        assert config.shadow_mode.enabled is False
