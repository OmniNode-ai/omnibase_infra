# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lane overlay answers the contract hot-load gate (OMN-21318).

The shared runtime config file is baked into the image and cannot name one lane,
so the gate reads the lane's own row from the ``runtime_contract_hotload``
overlay (root named by ``ONEX_SKILL_OVERLAY_ROOTS``, lane named by
``ONEX_RUNTIME_LANE``). The seams: a lane with no row resolves the gate from the
shared file exactly as before (the failing control), a row answers for its own
lane and no other, and the resolved mode reaches the start-up topic pass.

Lane names are two registered runtime lanes (the resolver refuses an unregistered
one); environment names are planted in each test's own overlay and config. The
code under test holds none of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.models.model_contract_hotload_lane_overlay import (
    lane_defers_topics_to_first_request,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess

pytestmark = pytest.mark.unit

_LANE = "dogfood"
_OTHER_LANE = "judge"
_ENVIRONMENT = "sandbox"


def _overlay_root(tmp_path: Path, lanes: dict[str, dict[str, object]]) -> Path:
    root = tmp_path / "overlays"
    (root / "runtime_contract_hotload").mkdir(parents=True)
    (root / "runtime_contract_hotload" / "overlay.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "lanes": lanes}), encoding="utf-8"
    )
    return root


def _gate(
    monkeypatch: pytest.MonkeyPatch,
    *,
    root: Path | None,
    lane: str = _LANE,
    file_block: dict[str, object] | None = None,
    environment: str = _ENVIRONMENT,
):
    monkeypatch.setenv("ONEX_ENVIRONMENT", environment)
    monkeypatch.setenv("ONEX_RUNTIME_LANE", lane)
    if root is None:
        monkeypatch.delenv("ONEX_SKILL_OVERLAY_ROOTS", raising=False)
    else:
        monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", str(root))
    host = RuntimeHostProcess(
        event_bus=EventBusInmemory(),
        config={
            "service_name": "hotload_lane_overlay",
            "node_name": "gate",
            "event_bus": {"environment": environment},
            **({"contract_hotload": file_block} if file_block is not None else {}),
        },
    )
    gate = host._resolve_contract_hotload_gate()
    assert lane_defers_topics_to_first_request() is gate.defers_topics_to_first_request
    return gate


_ON_DEMAND_ROW = {
    "enabled": True,
    "mode": "on_demand",
    "allowed_environments": [_ENVIRONMENT],
}


class TestLaneRowAnswersTheGate:
    def test_a_lane_with_no_overlay_resolves_from_the_shared_file_exactly_as_before(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failing control: no overlay root anywhere, the gate is off and defers nothing."""
        gate = _gate(monkeypatch, root=None)
        assert gate.enabled is False
        assert gate.defers_topics_to_first_request is False

    def test_a_lane_without_a_row_ignores_another_lanes_row(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Red if the lookup ignores the lane name: every lane would turn on."""
        root = _overlay_root(tmp_path, {_LANE: _ON_DEMAND_ROW})
        gate = _gate(monkeypatch, root=root, lane=_OTHER_LANE)
        assert gate.enabled is False
        assert gate.defers_topics_to_first_request is False

    def test_a_lane_without_a_row_keeps_the_shared_files_block(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root = _overlay_root(tmp_path, {_LANE: _ON_DEMAND_ROW})
        gate = _gate(
            monkeypatch,
            root=root,
            lane=_OTHER_LANE,
            file_block={
                "enabled": True,
                "mode": "full_load",
                "allowed_environments": [_ENVIRONMENT],
            },
        )
        assert (gate.enabled, gate.mode) == (True, "full_load")
        assert gate.defers_topics_to_first_request is False

    def test_the_lanes_on_demand_row_turns_the_gate_on_and_defers_topics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root = _overlay_root(tmp_path, {_LANE: _ON_DEMAND_ROW})
        gate = _gate(monkeypatch, root=root)
        assert (gate.enabled, gate.mode) == (True, "on_demand")
        assert gate.defers_topics_to_first_request is True

    def test_a_full_load_row_enables_the_gate_and_defers_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root = _overlay_root(tmp_path, {_LANE: {**_ON_DEMAND_ROW, "mode": "full_load"}})
        gate = _gate(monkeypatch, root=root)
        assert (gate.enabled, gate.mode) == (True, "full_load")
        assert gate.defers_topics_to_first_request is False

    def test_the_row_answers_over_the_shared_files_block(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root = _overlay_root(tmp_path, {_LANE: _ON_DEMAND_ROW})
        gate = _gate(
            monkeypatch,
            root=root,
            file_block={
                "enabled": True,
                "mode": "full_load",
                "allowed_environments": [_ENVIRONMENT],
            },
        )
        assert gate.mode == "on_demand"


class TestErrorChain:
    def test_a_row_naming_an_environment_the_lane_is_not_in_ends_the_start(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root = _overlay_root(tmp_path, {_LANE: _ON_DEMAND_ROW})
        with pytest.raises(ProtocolConfigurationError, match="does not name"):
            _gate(monkeypatch, root=root, environment="elsewhere")

    def test_an_unregistered_lane_name_has_no_row_and_boots_as_before(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root = _overlay_root(tmp_path, {"not-a-lane": _ON_DEMAND_ROW})
        gate = _gate(monkeypatch, root=root, lane="not-a-lane")
        assert gate.enabled is False

    def test_a_malformed_overlay_ends_the_start_for_every_lane(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root = _overlay_root(
            tmp_path, {_LANE: {"enabled": True, "allowed_environments": ["x"]}}
        )
        with pytest.raises(ProtocolConfigurationError, match="lane overlay"):
            _gate(monkeypatch, root=root, lane=_OTHER_LANE)
