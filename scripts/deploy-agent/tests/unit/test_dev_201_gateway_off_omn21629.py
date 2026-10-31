# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21588: the lab dev lane declares its gateway-deploy phase off.

The gateway forwarder dialled the cloud dev broker, which the cloud owner has
deleted. Its TLS handshake failed on every deploy, the gateway-deploy phase
raised ``GATEWAY_DEPLOY_FAILED`` and the dev boot guard rolled the lane back for
a fault outside the lab runtime. The switch is data: ``flags: disabled_phases``
on the default instance in ``config/deploy_lane_routing.yaml``. These tests read
the SHIPPED table, so removing the entry fails them.

The phase itself is kept: an instance without the flag still runs it.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args
from unittest.mock import patch

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import (
    DEV_LANE_GATEWAY_SERVICES,
    EnumRuntimeLane,
    Phase,
    PhaseStatus,
    Scope,
    services_for_scope,
)
from deploy_agent.executor import (
    _LANE_CONFIGS,
    DEFAULT_DEV_INSTANCE,
    DEV_INSTANCE_LANE_CONFIGS,
    DeployExecutor,
    EnumInstancePhase,
    GatewayDeployScriptUnavailableError,
    _requested_services_for_up,
    lane_config_for,
    lane_runs_phase,
    lane_with_phases_off,
    select_dev_instance,
)
from deploy_agent.routing import (
    AGENT_CLONE_ROOT,
    ModelLaneFlags,
    PhaseName,
    RoutingTableError,
    lane_flags_for_instance,
    parse_lane_flags,
)

pytestmark = [pytest.mark.unit, pytest.mark.gateway_lane]

SHA = "a" * 40
GATEWAY = EnumInstancePhase.GATEWAY_DEPLOY
#: The composition no flag touches: what the default instance ran before.
HISTORICAL = _LANE_CONFIGS[EnumRuntimeLane.DEV]
#: What the module built from the SHIPPED table at import, before any fixture.
BUILT_AT_IMPORT = DEV_INSTANCE_LANE_CONFIGS[DEFAULT_DEV_INSTANCE]


def _noop(phase: Phase, status: PhaseStatus) -> None:
    return None


def _fail() -> str:
    raise AssertionError("the gateway deploy script was looked up")


@pytest.fixture(autouse=True)
def _shipped_default_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the composition the agent builds from the SHIPPED table in the slot.

    The ``gateway_lane`` marker (conftest) gives the slot the historical
    composition, which carries no flag; these tests are about the flag, so they
    set the shipped one back.
    """
    monkeypatch.setitem(
        DEV_INSTANCE_LANE_CONFIGS, DEFAULT_DEV_INSTANCE, BUILT_AT_IMPORT
    )
    select_dev_instance(DEFAULT_DEV_INSTANCE)


class TestTheShippedTable:
    def test_the_default_instance_declares_gateway_deploy_off(self) -> None:
        flags = lane_flags_for_instance(AGENT_CLONE_ROOT, DEFAULT_DEV_INSTANCE)
        assert flags.disabled_phases == (GATEWAY.value,)

    def test_the_executor_built_its_composition_from_that_declaration(self) -> None:
        assert lane_with_phases_off(HISTORICAL, (GATEWAY.value,)) == BUILT_AT_IMPORT
        assert BUILT_AT_IMPORT != HISTORICAL

    def test_the_declaration_turns_off_that_phase_and_no_other(self) -> None:
        cfg = lane_config_for(EnumRuntimeLane.DEV)
        assert cfg.disabled_phases == frozenset({GATEWAY})
        assert cfg.disabled_services == frozenset(DEV_LANE_GATEWAY_SERVICES)
        for phase in EnumInstancePhase:
            assert lane_runs_phase(EnumRuntimeLane.DEV, phase) is (phase is not GATEWAY)

    def test_the_other_instances_declare_no_flag_phases(self) -> None:
        for name in DEV_INSTANCE_LANE_CONFIGS:
            if name != DEFAULT_DEV_INSTANCE:
                flags = lane_flags_for_instance(AGENT_CLONE_ROOT, name)
                assert flags.disabled_phases == ()

    def test_the_table_accepts_exactly_the_executors_phases(self) -> None:
        assert set(get_args(PhaseName)) == {p.value for p in EnumInstancePhase}


class TestThePlanForTheLabLane:
    """What a rebuild of the default instance would do, read without a deploy."""

    def test_the_plan_has_no_gateway_service(self) -> None:
        # The dev scope is still RESPONSIBLE for the forwarder ...
        assert set(DEV_LANE_GATEWAY_SERVICES) <= set(
            services_for_scope(Scope.FULL, lane=EnumRuntimeLane.DEV)
        )
        # ... and nothing hands it to compose or to the gateway script.
        for scope in (Scope.RUNTIME, Scope.FULL):
            requested = _requested_services_for_up(scope, [], lane=EnumRuntimeLane.DEV)
            assert requested
            assert not set(requested) & set(DEV_LANE_GATEWAY_SERVICES)

    def test_the_gateway_phase_never_reaches_the_script_or_the_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(executor_mod, "deploy_gateway_script", _fail)
        monkeypatch.setattr(executor_mod, "gateway_env_file", _fail)
        with patch("deploy_agent.executor._run") as run:
            DeployExecutor()._deploy_gateway_lane(
                _noop,
                lane=EnumRuntimeLane.DEV,
                build_source="release",
                targets=list(DEV_LANE_GATEWAY_SERVICES),
                git_ref=SHA,
            )
        run.assert_not_called()

    def test_a_runtime_rebuild_restarts_no_gateway_service(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(executor_mod, "deploy_gateway_script", _fail)
        executor = DeployExecutor()
        with (
            patch.object(executor, "_compose_build"),
            patch.object(executor, "_compose_up"),
            patch.object(
                executor_mod, "_reconcile_lock_roots", return_value=[str(tmp_path)]
            ),
        ):
            restarted = executor.rebuild_scope(
                Scope.RUNTIME, [], _noop, git_sha=SHA, lane=EnumRuntimeLane.DEV
            )
        assert restarted
        assert not set(restarted) & set(DEV_LANE_GATEWAY_SERVICES)


class TestThePhaseStaysAvailable:
    def test_an_instance_without_the_flag_runs_the_gateway_phase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(DEV_INSTANCE_LANE_CONFIGS, DEFAULT_DEV_INSTANCE, HISTORICAL)
        monkeypatch.setattr(
            executor_mod, "deploy_gateway_script", lambda: "/nonexistent/deploy.sh"
        )
        assert lane_runs_phase(EnumRuntimeLane.DEV, GATEWAY)
        with pytest.raises(GatewayDeployScriptUnavailableError):
            DeployExecutor()._deploy_gateway_lane(
                _noop,
                lane=EnumRuntimeLane.DEV,
                build_source="release",
                targets=list(DEV_LANE_GATEWAY_SERVICES),
                git_ref=SHA,
            )

    def test_an_empty_flag_changes_nothing(self) -> None:
        assert lane_with_phases_off(HISTORICAL, ()) == HISTORICAL
        assert GATEWAY not in HISTORICAL.disabled_phases
        assert not HISTORICAL.disabled_services & set(DEV_LANE_GATEWAY_SERVICES)


class TestTheFlagParsing:
    def test_the_flag_unions_with_the_phases_a_composition_already_has(self) -> None:
        narrowed = HISTORICAL.model_copy(
            update={"disabled_phases": frozenset({EnumInstancePhase.LAB_OVERLAY})}
        )
        merged = lane_with_phases_off(narrowed, ("onex-api-pin",))
        assert merged.disabled_phases == frozenset(
            {EnumInstancePhase.LAB_OVERLAY, EnumInstancePhase.ONEX_API_PIN}
        )
        assert not merged.disabled_services

    def test_an_unknown_phase_refuses_the_table(self) -> None:
        text = (
            "default_instance: x\n"
            "instances:\n"
            "  x:\n"
            "    flags:\n"
            "      disabled_phases: [not-a-phase]\n"
        )
        with pytest.raises(RoutingTableError, match="not-a-phase"):
            parse_lane_flags(text)

    def test_an_unknown_flag_key_still_refuses_the_table(self) -> None:
        text = (
            "default_instance: x\n"
            "instances:\n"
            "  x:\n"
            "    flags:\n"
            "      disabled_phase: [gateway-deploy]\n"
        )
        with pytest.raises(RoutingTableError):
            parse_lane_flags(text)

    def test_the_flag_is_a_typed_tuple(self) -> None:
        flags = ModelLaneFlags(disabled_phases=("gateway-deploy",))
        assert flags.disabled_phases == ("gateway-deploy",)
        assert ModelLaneFlags().disabled_phases == ()
