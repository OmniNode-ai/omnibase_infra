# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cold wiring uses the fast safe parser without changing refusal semantics."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from omnibase_infra.handlers.handler_contract_subscribe_preflight import (
    load_consumer_group_handover,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _read_completion_bound,
    _read_declared_key_grains,
    _read_dlq_topics,
    _read_state_io,
)
from omnibase_infra.runtime.auto_wiring.models.model_consume_concurrency import (
    load_consume_concurrency,
)
from omnibase_infra.runtime.contract_terminal_events import load_terminal_event_topics
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    load_published_events_map,
)
from omnibase_infra.tools.contract_topic_extractor import read_projection_api_topics

pytestmark = pytest.mark.unit

READERS = (
    _read_completion_bound,
    _read_declared_key_grains,
    _read_dlq_topics,
    _read_state_io,
    load_published_events_map,
    load_consumer_group_handover,
    load_consume_concurrency,
    load_terminal_event_topics,
    read_projection_api_topics,
)


@pytest.mark.parametrize("portable", [False, True])
def test_valid_contract_preserves_wiring_extensions(
    portable: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if portable:
        monkeypatch.delattr(yaml, "CSafeLoader")
    path = tmp_path / "contract.yaml"
    path.write_text(
        "projection_api:\n"
        "  expose: true\n"
        "  key_grain: immutable\n"
        "  bus_backed: true\n"
        "  topic: probe-snapshot\n"
        "event_bus:\n"
        "  dlq_topics: [probe-dlq]\n"
        "state_io:\n"
        "  database: omnibase_infra\n"
        "  table: workflow_state\n"
        "completion_bound:\n"
        "  max_wall_seconds: 900\n"
        "  on_runtime_restart: terminalise_failed\n"
        "  failure_class: runtime_restart_during_delegation\n"
        "published_events:\n"
        "  - event_type: ProbeCompleted\n"
        "    topic: probe-completed\n"
        "terminal_event: probe-completed\n"
        "consume_concurrency:\n"
        "  max_in_flight_records: 3\n"
        "consumer_group_handover:\n"
        "  from_group_id: retired-probe\n"
        "  from_auto_offset_reset: earliest\n"
    )

    assert _read_declared_key_grains(path) == ("immutable",)
    assert _read_dlq_topics(path) == ["probe-dlq"]
    assert _read_state_io(path) == {
        "database": "omnibase_infra",
        "table": "workflow_state",
    }
    bound = _read_completion_bound(path)
    assert bound is not None
    assert bound.max_wall_seconds == 900
    assert bound.failure_class == "runtime_restart_during_delegation"
    assert load_published_events_map(path) == {"ProbeCompleted": "probe-completed"}
    handover = load_consumer_group_handover(path)
    assert handover is not None
    assert handover.from_group_id == "retired-probe"
    assert handover.from_auto_offset_reset == "earliest"
    assert load_consume_concurrency(path).max_in_flight_records == 3
    assert load_terminal_event_topics(path) == frozenset({"probe-completed"})
    assert read_projection_api_topics(path) == ("probe-snapshot",)


@pytest.mark.parametrize("reader", READERS)
def test_cold_wiring_uses_libyaml_when_available(
    reader: Callable[[Path], object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "contract.yaml"
    path.write_text("name: node_probe\n")
    fast_loader = Mock(wraps=yaml.CSafeLoader)
    monkeypatch.setattr(yaml, "CSafeLoader", fast_loader)

    reader(path)

    fast_loader.assert_called_once()


@pytest.mark.parametrize("reader", READERS)
def test_portable_safe_loader_is_used_without_libyaml(
    reader: Callable[[Path], object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "contract.yaml"
    path.write_text("name: node_probe\n")
    portable_loader = Mock(wraps=yaml.SafeLoader)
    monkeypatch.delattr(yaml, "CSafeLoader")
    monkeypatch.setattr(yaml, "SafeLoader", portable_loader)

    reader(path)

    portable_loader.assert_called_once()


@pytest.mark.parametrize("portable", [False, True])
@pytest.mark.parametrize(
    "body", ["name: [unterminated\n", "!!python/object/apply:builtins.str [unsafe]\n"]
)
def test_unreadable_yaml_never_provides_an_exemption_or_wiring_configuration(
    body: str, portable: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if portable:
        monkeypatch.delattr(yaml, "CSafeLoader")
    path = tmp_path / "contract.yaml"
    path.write_text(body)

    assert _read_declared_key_grains(path) == ()
    assert load_published_events_map(path) == {}
    assert load_terminal_event_topics(path) == frozenset()
    assert read_projection_api_topics(path) == ()
    for reader in (
        _read_dlq_topics,
        _read_state_io,
        _read_completion_bound,
        load_consumer_group_handover,
        load_consume_concurrency,
    ):
        with pytest.raises(yaml.YAMLError):
            reader(path)


def test_missing_contract_keeps_each_readers_absent_value(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"

    assert _read_declared_key_grains(path) == ()
    assert _read_dlq_topics(path) == []
    assert _read_state_io(path) == {}
    assert _read_completion_bound(path) is None
    assert load_published_events_map(path) == {}
    assert load_consumer_group_handover(path) is None
    assert load_consume_concurrency(path).is_serial
    assert load_terminal_event_topics(path) == frozenset()
    assert read_projection_api_topics(path) == ()
