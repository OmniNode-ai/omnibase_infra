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


# Pinned independently of the YAML: editing the declared file must not silently
# change which interface a service binds.
_PINNED_LOOPBACK_ADDRESS = "127.0.0.1"
_PINNED_BIND_ALL_ADDRESS = ".".join(["0"] * 4)
_PINNED_POSTGRES_PORT = 5432
_PINNED_TCP_PORT_BOUNDS = (1, 65535)


def test_the_declared_settings_equal_the_pinned_values() -> None:
    settings = load_network_binding_settings()

    assert settings.loopback_address == _PINNED_LOOPBACK_ADDRESS
    assert settings.bind_all_address == _PINNED_BIND_ALL_ADDRESS
    assert settings.postgres_port == _PINNED_POSTGRES_PORT
    assert (settings.tcp_port_min, settings.tcp_port_max) == _PINNED_TCP_PORT_BOUNDS


def _field_default(module_name: str, class_name: str, field: str) -> object:
    model = getattr(importlib.import_module(module_name), class_name)
    # The defaults are built lazily by a default_factory, so resolve the factory.
    return model.model_fields[field].get_default(call_default_factory=True)


@pytest.mark.parametrize(("module_name", "class_name"), _LOOPBACK_BOUND_CONFIGS)
def test_health_servers_bind_the_declared_loopback_address(
    module_name: str, class_name: str
) -> None:
    default = _field_default(module_name, class_name, "health_check_host")

    assert default == _PINNED_LOOPBACK_ADDRESS
    assert default == load_network_binding_settings().loopback_address


@pytest.mark.parametrize(("module_name", "class_name", "field"), _BIND_ALL_CONFIGS)
def test_container_facing_servers_bind_the_declared_bind_all_address(
    module_name: str, class_name: str, field: str
) -> None:
    default = _field_default(module_name, class_name, field)

    assert default == _PINNED_BIND_ALL_ADDRESS
    assert default == load_network_binding_settings().bind_all_address


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        *_LOOPBACK_BOUND_CONFIGS,
        *(
            (module, cls)
            for module, cls, field in _BIND_ALL_CONFIGS
            if field == "health_check_host"
        ),
    ],
)
def test_a_health_host_environment_override_still_wins(
    monkeypatch: pytest.MonkeyPatch, module_name: str, class_name: str
) -> None:
    model = getattr(importlib.import_module(module_name), class_name)

    prefix = model.model_config["env_prefix"]
    for name, info in model.model_fields.items():
        if info.is_required():
            value = "postgresql://db.test/x" if "dsn" in name else "required.test:9092"
            monkeypatch.setenv(f"{prefix}{name.upper()}", value)
    monkeypatch.delenv(f"{prefix}HEALTH_CHECK_HOST", raising=False)
    assert model(_env_file=None).health_check_host == model.model_fields[
        "health_check_host"
    ].get_default(call_default_factory=True)
    monkeypatch.setenv(f"{prefix}HEALTH_CHECK_HOST", "health-probe.test")

    assert model(_env_file=None).health_check_host == "health-probe.test"
    assert (
        model(health_check_host="explicit.test", _env_file=None).health_check_host
        == "explicit.test"
    )


async def test_the_registration_storage_port_is_the_declared_port() -> None:
    # OMN-22131: the handler takes only its container; the port lives in the
    # settings the registration plugin binds there, read from the declared file.
    from unittest.mock import AsyncMock, MagicMock, patch
    from uuid import uuid4

    from omnibase_infra.models.registration_storage import (
        ModelRegistrationStoragePostgresConfig,
    )
    from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
        ServiceRegistration,
    )

    plugin = ServiceRegistration()
    plugin._pool = MagicMock()
    config = MagicMock()
    config.correlation_id = uuid4()
    config.container.service_registry.register_instance = AsyncMock()
    handler = MagicMock()
    handler._ensure_pool = AsyncMock()

    with (
        patch(
            "omnibase_infra.handlers.registration_storage"
            ".handler_registration_storage_postgres"
            ".HandlerRegistrationStoragePostgres",
            return_value=handler,
        ),
        patch(
            "omnibase_infra.nodes.node_registration_storage_effect"
            ".registry.registry_infra_registration_storage"
            ".RegistryInfraRegistrationStorage.register_handler",
        ),
        patch.dict("os.environ", {"OMNIBASE_INFRA_DB_URL": "postgresql://h/db"}),
    ):
        await plugin._wire_registration_storage(config)

    bound = [
        c.kwargs["instance"]
        for c in config.container.service_registry.register_instance.await_args_list
        if c.kwargs["interface"] is ModelRegistrationStoragePostgresConfig
    ]
    assert len(bound) == 1
    assert bound[0].port == _PINNED_POSTGRES_PORT
    assert bound[0].port == load_network_binding_settings().postgres_port


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

    assert (bounds.tcp_port_min, bounds.tcp_port_max) == _PINNED_TCP_PORT_BOUNDS
    assert accepted == []
    assert len(refused) == 1
