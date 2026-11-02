# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The network binding burn-down leaves no host, address or port literal in its files (OMN-21623).

Runs the core ``contracts_and_di`` validator over the source files that fell back to a
host, address or port literal before the declared settings replaced them, with a planted
file as the positive control.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.validation.validator_contracts_and_di import (
    ValidatorContractsAndDi,
)

pytestmark = pytest.mark.unit

_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra"
_BINDING_RULES = frozenset({"literal_host", "literal_ip", "literal_port"})
_SLICE_FILES = (
    "handlers/models/mcp/model_mcp_handler_config.py",
    "handlers/registration_storage/handler_registration_storage_postgres.py",
    "models/mcp/model_mcp_server_config.py",
    "nodes/node_setup_local_provision_effect/handlers/handler_local_provision.py",
    "nodes/node_setup_preflight_effect/handlers/handler_preflight_check.py",
    "nodes/node_setup_validate_effect/handlers/handler_service_validate.py",
    "runtime/security_metadata_validator.py",
    "runtime/service_kernel.py",
    "services/observability/agent_actions/config.py",
    "services/observability/consumer_health/config.py",
    "services/observability/context_audit/config.py",
    "services/observability/infra_routing_decisions/config.py",
    "services/observability/injection_effectiveness/config.py",
    "services/observability/llm_cost_aggregation/config.py",
    "services/observability/skill_lifecycle/config.py",
    "services/post_merge/config.py",
    "services/waitlist_signup_notifier/config.py",
    "models/network_binding/model_bind_all_health_server_binding.py",
    "models/network_binding/model_loopback_health_server_binding.py",
    "models/network_binding/model_network_binding_settings.py",
)


def _binding_rule_ids(targets: list[Path]) -> list[str]:
    result = ValidatorContractsAndDi().validate(targets)
    return sorted(
        str(issue.code) for issue in result.issues if issue.code in _BINDING_RULES
    )


def test_the_validator_reports_a_planted_host_address_and_port(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    planted = tmp_path / "src" / "planted.py"
    planted.parent.mkdir()
    planted.write_text(
        'BIND = "0.0.0.0"\n'
        'bootstrap_servers = "localhost:19092"\n'
        "DEFAULT_HTTP_PORT = 8085\n",
        encoding="utf-8",
    )

    assert _binding_rule_ids([planted]) == [
        "literal_host",
        "literal_ip",
        "literal_port",
    ]


@pytest.mark.parametrize("relative", _SLICE_FILES)
def test_the_slice_file_holds_no_host_address_or_port_literal(relative: str) -> None:
    path = _SOURCE_ROOT / relative

    assert path.is_file()
    assert _binding_rule_ids([path]) == []
