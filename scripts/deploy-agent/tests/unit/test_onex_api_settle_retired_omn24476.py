# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24476 -- the settle tail must not build or pin a retired onex-api.

The dev lane's onex-api service is retired by name in
``docker/runtime-policy.env`` (``DEV_RUNTIME_MAIN_RETIRED_SERVICES``). An instance
whose folded composition retires it runs no onex-api, so the settle tail must neither
build its image nor deliver its pin: nothing recreates the service, and the repoint
would advance a tag toward an image the lane no longer starts.

What these tests pin:

1. ``lane_config_for`` turns the ONEX_API_PIN phase off, and disables the service, when
   the policy retires onex-api, so ``lane_runs_phase`` answers False. The settle already
   builds and delivers nothing when that phase is off
   (``test_onex_api_pin_without_lab_overlay_omn23422``).
2. Positive control: an instance that retires nothing still runs the phase.
3. A retirement that does not name onex-api leaves the phase on.
"""

from __future__ import annotations

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.executor import (
    EnumInstancePhase,
    lane_config_for,
    lane_runs_phase,
    select_dev_instance,
)

pytestmark = pytest.mark.unit


def _retire(monkeypatch: pytest.MonkeyPatch, services: set[str]) -> None:
    select_dev_instance("dev-201")
    monkeypatch.setattr(executor_mod, "_folded_dev_services", lambda lane: services)


def test_a_retired_onex_api_turns_its_settle_phase_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _retire(monkeypatch, {"onex-api", "cloud-migration"})

    assert not lane_runs_phase(EnumRuntimeLane.DEV, EnumInstancePhase.ONEX_API_PIN)
    config = lane_config_for(EnumRuntimeLane.DEV)
    assert {"onex-api", "cloud-migration"} <= config.disabled_services


def test_an_instance_that_retires_nothing_still_runs_the_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _retire(monkeypatch, set())

    assert lane_runs_phase(EnumRuntimeLane.DEV, EnumInstancePhase.ONEX_API_PIN)


def test_a_retirement_that_does_not_name_onex_api_leaves_the_phase_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _retire(monkeypatch, {"intelligence-api"})

    assert lane_runs_phase(EnumRuntimeLane.DEV, EnumInstancePhase.ONEX_API_PIN)
    assert "intelligence-api" in lane_config_for(EnumRuntimeLane.DEV).disabled_services
