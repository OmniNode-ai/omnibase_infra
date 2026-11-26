# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev runtime declares the total time limit of a hosted dod-verify start command (OMN-25856).

omnimarket's ``dod-verify-start`` handler refuses every hosted start when
``DOD_VERIFY_START_BUDGET_S`` is absent (no code default), so the runtime that hosts
``node_dod_verify`` must declare it before that handler change merges. The value must be
above 0 and below the 600 s dispatch deadline; 540 leaves the verdict write and publish
inside the deadline. It sits beside ``DOD_VERIFY_PRODUCT_CLONE_PIN_FILE`` in the same two
host services (``test_dev_runtime_product_clone_pin_omn25901.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.lane_overlay_fixtures import load_compose

pytestmark = [pytest.mark.unit]

DOCKER = Path(__file__).resolve().parents[2] / "docker"
BUDGET_ENV = "DOD_VERIFY_START_BUDGET_S"
DISPATCH_DEADLINE_S = 600.0

HOSTS: tuple[tuple[str, str], ...] = (
    ("docker-compose.dev-lane.yml", "runtime-effects"),
    ("docker-compose.dev-tenant-fold.yml", "omninode-runtime"),
)


def _environment(filename: str, service: str) -> dict[str, object]:
    found = load_compose(DOCKER / filename)["services"][service]["environment"]
    assert isinstance(found, dict)
    return found


@pytest.mark.parametrize(("filename", "service"), HOSTS, ids=[h[1] for h in HOSTS])
def test_start_budget_is_declared_as_540_seconds(filename: str, service: str) -> None:
    assert _environment(filename, service).get(BUDGET_ENV) == "540"


@pytest.mark.parametrize(("filename", "service"), HOSTS, ids=[h[1] for h in HOSTS])
def test_start_budget_is_inside_the_dispatch_deadline(
    filename: str, service: str
) -> None:
    value = float(str(_environment(filename, service)[BUDGET_ENV]))
    assert 0 < value < DISPATCH_DEADLINE_S


@pytest.mark.parametrize(("filename", "service"), HOSTS, ids=[h[1] for h in HOSTS])
def test_declared_beside_the_product_clone_pin_variable(
    filename: str, service: str
) -> None:
    """Positive control: the same service already carries the other DOD_VERIFY_ variable."""
    environment = _environment(filename, service)
    assert "DOD_VERIFY_PRODUCT_CLONE_PIN_FILE" in environment
    assert BUDGET_ENV in environment
