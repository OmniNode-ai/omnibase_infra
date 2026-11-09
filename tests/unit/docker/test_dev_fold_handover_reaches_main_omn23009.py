# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23009: the fold handovers the policy renders reach the folded main runtime.

The runtime policy contract declares, per folded profile, the consumer group
handover from the retired service (effects and workers, ``last_ran``), and
``render_runtime_policy_env`` writes it as ``DEV_RUNTIME_MAIN_FOLD_HANDOVERS``.
The runtime reads ``ONEX_RUNTIME_FOLD_HANDOVERS``. Dev ran folded for hours with
no handover because the fold overlay never set that name on ``omninode-runtime``:
the container held ``ONEX_RUNTIME_FOLDED_PROFILES`` and nothing else. These tests
read the real overlay and the real policy, each with a failing control.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_contract_subscribe_preflight import (
    FOLD_HANDOVERS_ENV_VAR,
)
from scripts.render_runtime_policy_env import load_contract, render_env
from tests.helpers.lane_overlay_fixtures import load_compose

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
OVERLAY = ROOT / "docker" / "docker-compose.dev-tenant-fold.yml"
RENDERED_KEY = "DEV_RUNTIME_MAIN_FOLD_HANDOVERS"
SERVICE = "omninode-runtime"


def _main_environment(compose: dict) -> dict[str, str]:
    environment = compose["services"][SERVICE]["environment"]
    assert isinstance(environment, dict)
    return {str(k): str(v) for k, v in environment.items()}


def _unreached_handovers(environment: dict[str, str]) -> list[str]:
    """Every folded role with a rendered handover that the runtime would not read."""
    rendered = json.loads(render_env(load_contract())[RENDERED_KEY])
    value = environment.get(FOLD_HANDOVERS_ENV_VAR, "")
    if f"${{{RENDERED_KEY}" not in value:
        return sorted(rendered)
    return []


def test_the_runtime_reads_the_name_the_overlay_sets() -> None:
    assert FOLD_HANDOVERS_ENV_VAR == "ONEX_RUNTIME_FOLD_HANDOVERS"


def test_the_policy_renders_a_handover_for_effects_and_workers() -> None:
    """Positive control: there is something for the overlay to carry."""
    rendered = json.loads(render_env(load_contract())[RENDERED_KEY])
    assert {"effects", "workers"} <= set(rendered)


def test_the_overlay_passes_the_rendered_handovers_to_the_main_runtime() -> None:
    assert _unreached_handovers(_main_environment(load_compose(OVERLAY))) == []
    value = _main_environment(load_compose(OVERLAY))[FOLD_HANDOVERS_ENV_VAR]
    assert value.startswith(f"${{{RENDERED_KEY}:?")


def test_an_overlay_without_the_name_leaves_effects_and_workers_unhanded() -> None:
    """Failing control: the overlay as dev ran it from the fold's merge."""
    environment = _main_environment(load_compose(OVERLAY))
    environment.pop(FOLD_HANDOVERS_ENV_VAR)
    assert _unreached_handovers(environment) == ["effects", "workers"]


def test_an_overlay_that_sets_the_name_from_another_variable_is_caught() -> None:
    """Failing control: the name is set, but not from the rendered policy value."""
    environment = _main_environment(load_compose(OVERLAY))
    environment[FOLD_HANDOVERS_ENV_VAR] = "${DEV_RUNTIME_MAIN_SOMETHING_ELSE:-}"
    assert _unreached_handovers(environment) == ["effects", "workers"]
