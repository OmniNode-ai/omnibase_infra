# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""node_deploy_agent_reading_compute judges the dev deploy agent ticket's readings (OMN-26724)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_deploy_agent_reading_compute/contract.yaml"
)


def _handler() -> Any:
    spec = yaml.safe_load(CONTRACT.read_text())["handler_routing"]["handlers"][0][
        "handler"
    ]
    return getattr(importlib.import_module(spec["module"]), spec["name"])()


def _judge(**overrides: Any) -> Any:
    from omnibase_infra.models.model_deploy_agent_reading_request import (
        ModelDeployAgentReadingRequest,
    )

    good: dict[str, Any] = {
        "dev_loaded_code_sha": "a" * 40,
        "dev_uptime_seconds": 600,
        "dev_loaded_sha_has_merge": True,
        "merge_age_seconds": 3600,
        "dev_last_status": "success",
        "dev_last_runtime_lane": "dev",
        "dev_last_phase_result_count": 7,
        "stability_uptime_seconds": 90000,
        "prod_uptime_seconds": 80000,
    }
    good.update(overrides)
    return _handler().handle(ModelDeployAgentReadingRequest(**good))


def test_known_good_readings_meet_all_three_criteria() -> None:
    verdict = _judge()
    assert (verdict.ac1_met, verdict.ac2_met, verdict.ac3_met) == (True, True, True)
    assert verdict.reasons == ()


@pytest.mark.parametrize(
    "bad",
    [
        {"dev_loaded_code_sha": ""},
        {"dev_loaded_sha_has_merge": False},
        {"dev_uptime_seconds": None},
        {"dev_uptime_seconds": 0},
        {"dev_uptime_seconds": 7200},
    ],
)
def test_ac1_is_not_met_without_a_restart_onto_the_merge(bad: dict[str, Any]) -> None:
    verdict = _judge(**bad)
    assert verdict.ac1_met is False
    assert any(reason.startswith("AC1") for reason in verdict.reasons)


@pytest.mark.parametrize(
    "bad",
    [
        {"dev_last_status": "failed"},
        {"dev_last_runtime_lane": "stability"},
        {"dev_last_phase_result_count": 0},
    ],
)
def test_ac2_is_not_met_without_a_dev_success_with_phases(bad: dict[str, Any]) -> None:
    verdict = _judge(**bad)
    assert verdict.ac2_met is False
    assert verdict.ac1_met is True


@pytest.mark.parametrize(
    "bad",
    [
        {"prod_uptime_seconds": 300},
        {"stability_uptime_seconds": 600},
        {"stability_uptime_seconds": None},
        {"prod_uptime_seconds": None},
    ],
)
def test_ac3_is_not_met_when_another_agent_restarted_or_is_unread(
    bad: dict[str, Any],
) -> None:
    verdict = _judge(**bad)
    assert verdict.ac3_met is False
    assert verdict.ac1_met is True
