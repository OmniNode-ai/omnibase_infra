# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outbound-call timeouts are read from a declared file, not from literals (OMN-21623)."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from omnibase_infra.adapters.llm import (
    adapter_code_analysis_enrichment,
    adapter_code_review_analysis,
    adapter_documentation_generation,
    adapter_test_boilerplate_generation,
)
from omnibase_infra.adapters.project_tracker import (
    linear_graphql_project_tracker_adapter,
)
from omnibase_infra.adapters.ticket import adapter_ticket_service_linear
from omnibase_infra.diagnostics import bus_audit
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.outbound_timeouts import (
    ModelOutboundTimeouts,
    load_outbound_timeouts,
)

_TIMEOUTS_FILE = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "models"
    / "outbound_timeouts"
    / "outbound_timeouts.yaml"
)

_TIMEOUTS = load_outbound_timeouts()


def _declared_block() -> dict[str, object]:
    return yaml.safe_load(_TIMEOUTS_FILE.read_text(encoding="utf-8"))["timeouts"]


def _write(tmp_path: Path, block: dict[str, object]) -> Path:
    path = tmp_path / "timeouts.yaml"
    path.write_text(yaml.safe_dump({"timeouts": block}), encoding="utf-8")
    return path


@pytest.mark.unit
class TestOutboundTimeoutsContract:
    def test_loaded_timeouts_equal_the_declared_block(self) -> None:
        assert _TIMEOUTS.model_dump() == {
            key: float(value) for key, value in _declared_block().items()
        }

    @pytest.mark.parametrize("key", sorted(_declared_block()))
    def test_block_missing_a_value_fails_to_load(
        self, key: str, tmp_path: Path
    ) -> None:
        block = _declared_block()
        del block[key]
        with pytest.raises(ValueError, match=key):
            ModelOutboundTimeouts.from_file(_write(tmp_path, block))

    def test_file_without_a_timeouts_mapping_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "timeouts.yaml"
        path.write_text("other: 1\n", encoding="utf-8")
        with pytest.raises(ProtocolConfigurationError):
            ModelOutboundTimeouts.from_file(path)

    def test_non_positive_wait_is_refused(self, tmp_path: Path) -> None:
        block = _declared_block()
        block["probe_connect_seconds"] = 0
        with pytest.raises(ValueError, match="probe_connect_seconds"):
            ModelOutboundTimeouts.from_file(_write(tmp_path, block))

    @pytest.mark.parametrize(
        "key", ["llm_request_seconds", "llm_documentation_request_seconds"]
    )
    def test_request_wait_above_the_transport_ceiling_is_refused(
        self, key: str, tmp_path: Path
    ) -> None:
        block = _declared_block()
        block[key] = float(block["llm_long_form_max_seconds"]) + 1
        with pytest.raises(ValueError, match=key):
            ModelOutboundTimeouts.from_file(_write(tmp_path, block))


def _default(callable_: Callable[..., object], parameter: str) -> object:
    return inspect.signature(callable_).parameters[parameter].default


_TRACKER = linear_graphql_project_tracker_adapter.AdapterLinearGraphQLProjectTracker

_SITES: list[tuple[str, object, str]] = [
    (
        "linear graphql request",
        _default(_TRACKER.__init__, "timeout_seconds"),
        "linear_graphql_request_seconds",
    ),
    (
        "linear graphql transport",
        _default(_TRACKER.graphql_transport, "timeout_seconds"),
        "linear_graphql_request_seconds",
    ),
    (
        "linear graphql close",
        _default(_TRACKER.close, "timeout_seconds"),
        "close_seconds",
    ),
    (
        "linear ticket request",
        _default(adapter_ticket_service_linear.AdapterTicketLinear.__init__, "timeout"),
        "linear_ticket_request_seconds",
    ),
    (
        "linear ticket close",
        _default(
            adapter_ticket_service_linear.AdapterTicketLinear.close, "timeout_seconds"
        ),
        "close_seconds",
    ),
    (
        "bus audit poll",
        bus_audit.AuditConfig(broker="broker").poll_timeout,
        "audit_poll_seconds",
    ),
]


@pytest.mark.unit
class TestSitesUseDeclaredTimeouts:
    @pytest.mark.parametrize(
        ("site", "value", "field"), _SITES, ids=[site[0] for site in _SITES]
    )
    def test_site_carries_the_declared_value(
        self, site: str, value: object, field: str
    ) -> None:
        assert value == getattr(_TIMEOUTS, field), site
