# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every service that fell back to a host, address or port literal reads the declared settings (OMN-21623).

The consumers are imported as the runtime imports them, so a circular import
between the settings model and a consumer fails here, and each fallback is
compared with the value in ``network_binding_settings.yaml``.
"""

from __future__ import annotations

import importlib

import pytest

from omnibase_infra.models.network_binding import load_network_binding_settings

pytestmark = pytest.mark.integration

_LOOPBACK_BOUND_CONFIGS = (
    (
        "omnibase_infra.services.observability.agent_actions.config",
        "ConfigAgentActionsConsumer",
    ),
    (
        "omnibase_infra.services.observability.consumer_health.config",
        "ConfigConsumerHealthProjection",
    ),
    (
        "omnibase_infra.services.observability.context_audit.config",
        "ConfigContextAuditConsumer",
    ),
    (
        "omnibase_infra.services.observability.infra_routing_decisions.config",
        "ConfigInfraRoutingDecisionsConsumer",
    ),
    (
        "omnibase_infra.services.observability.llm_cost_aggregation.config",
        "ConfigLlmCostAggregation",
    ),
    (
        "omnibase_infra.services.observability.skill_lifecycle.config",
        "ConfigSkillLifecycleConsumer",
    ),
    ("omnibase_infra.services.post_merge.config", "ConfigPostMergeConsumer"),
)
_BIND_ALL_CONFIGS = (
    (
        "omnibase_infra.services.observability.injection_effectiveness.config",
        "ConfigInjectionEffectivenessConsumer",
        "health_check_host",
    ),
    (
        "omnibase_infra.services.waitlist_signup_notifier.config",
        "ConfigWaitlistSignupNotifier",
        "health_check_host",
    ),
    (
        "omnibase_infra.models.mcp.model_mcp_server_config",
        "ModelMCPServerConfig",
        "http_host",
    ),
    (
        "omnibase_infra.handlers.models.mcp.model_mcp_handler_config",
        "ModelMcpHandlerConfig",
        "host",
    ),
)


def _field_default(module_name: str, class_name: str, field: str) -> object:
    model = getattr(importlib.import_module(module_name), class_name)
    return model.model_fields[field].default


@pytest.mark.parametrize(("module_name", "class_name"), _LOOPBACK_BOUND_CONFIGS)
def test_health_servers_bind_the_declared_loopback_address(
    module_name: str, class_name: str
) -> None:
    default = _field_default(module_name, class_name, "health_check_host")

    assert default == load_network_binding_settings().loopback_address


@pytest.mark.parametrize(("module_name", "class_name", "field"), _BIND_ALL_CONFIGS)
def test_container_facing_servers_bind_the_declared_bind_all_address(
    module_name: str, class_name: str, field: str
) -> None:
    default = _field_default(module_name, class_name, field)

    assert default == load_network_binding_settings().bind_all_address


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [*_LOOPBACK_BOUND_CONFIGS, *(item[:2] for item in _BIND_ALL_CONFIGS[:2])],
)
def test_a_health_host_environment_override_still_wins(
    monkeypatch: pytest.MonkeyPatch, module_name: str, class_name: str
) -> None:
    model = getattr(importlib.import_module(module_name), class_name)

    prefix = model.model_config["env_prefix"]
    monkeypatch.setenv(f"{prefix}KAFKA_BOOTSTRAP_SERVERS", "broker.test:9092")
    monkeypatch.setenv(f"{prefix}POSTGRES_DSN", "postgresql://db.test/observability")
    monkeypatch.delenv(f"{prefix}HEALTH_CHECK_HOST", raising=False)
    assert (
        model(_env_file=None).health_check_host
        == model.model_fields["health_check_host"].default
    )
    monkeypatch.setenv(f"{prefix}HEALTH_CHECK_HOST", "health-probe.test")

    assert model(_env_file=None).health_check_host == "health-probe.test"
    assert (
        model(health_check_host="explicit.test", _env_file=None).health_check_host
        == "explicit.test"
    )


def test_the_registration_storage_port_default_is_the_declared_port() -> None:
    import inspect

    from omnibase_infra.handlers.registration_storage.handler_registration_storage_postgres import (
        HandlerRegistrationStoragePostgres,
    )

    default = (
        inspect.signature(HandlerRegistrationStoragePostgres.__init__)
        .parameters["port"]
        .default
    )

    assert default == load_network_binding_settings().postgres_port


def test_the_declared_port_bounds_gate_the_kernel_and_the_domain_validator() -> None:
    from omnibase_infra.runtime.security_metadata_validator import (
        SecurityMetadataValidator,
    )

    bounds = load_network_binding_settings()
    validator = SecurityMetadataValidator()

    accepted = validator.validate_domains(
        [f"example.com:{bounds.tcp_port_min}", f"example.com:{bounds.tcp_port_max}"]
    )
    refused = validator.validate_domains([f"example.com:{bounds.tcp_port_max + 1}"])

    assert accepted == []
    assert len(refused) == 1
