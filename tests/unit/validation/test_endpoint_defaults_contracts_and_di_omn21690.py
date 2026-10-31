# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The endpoint-defaults burn-down leaves no host, address or port literal in its files (OMN-21690).

Runs the core ``contracts_and_di`` validator over the source files that fell back to a
host, address or port literal before the declared defaults replaced them, with a planted
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
    "cli/artifact_reconcile.py",
    "event_bus/models/config/model_kafka_event_bus_config.py",
    "handlers/handler_ledger_write_guard.py",
    "handlers/handler_mcp.py",
    "observability/handlers/handler_metrics_prometheus.py",
    "observability/handlers/model_metrics_handler_config.py",
    "runtime/models/model_runtime_scheduler_config.py",
    "services/contract_resolver/launcher.py",
    "services/health_checker.py",
    "models/endpoint_defaults/model_endpoint_defaults.py",
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
