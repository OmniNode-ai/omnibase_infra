# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19508 AC1 and AC2: what staging delivery admits from the .202 dev lane.

Task B6 of the second-deploy-slot plan (epic OMN-19500), under the operator
ruling of 2026-09-25T00:56:45Z: an omnimarket revision may be proven on the
second deployed dev lane (receipt lane ``compose-dev-202``); omnibase_infra
stays proven on .201.

* AC1 ``sibling_compose_dev_202``: the sibling read in
  ``deliver-dev-candidate-to-staging.yml`` admits a PASS for a pinned
  omnimarket revision under ``compose-dev`` or ``compose-dev-202``, refuses when
  neither exists, and stays ``compose-dev`` alone for every other sibling.
* AC2 ``own_sha_requires_compose_dev``: the own-sha read still requires
  ``compose-dev``, never names ``compose-dev-202``, and a ``compose-dev-202``
  PASS alone is refused.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest
import yaml

from scripts.ci.lab_pass_receipt import (
    ANY_OF_DEFAULT_LANES,
    EnumLabLane,
    evaluate_gate,
)
from tests.scripts.ci._lab_pass_fixtures import (
    REPO,
    SHA_4ACA,
    FakeSurface,
    receipt,
    ts,
)

pytestmark = pytest.mark.ci

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "deliver-dev-candidate-to-staging.yml"

SIBLING_STEP = "Read the compose-dev lab-pass receipt for the pinned sibling revision"
OWN_SHA_STEP = "Read the lab-pass receipt for the delivered sha"


def _step_run(name: str) -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if step.get("name") == name:
                return str(step["run"])
    raise AssertionError(f"no step named {name!r}")


def _code_lines(run: str) -> str:
    return "\n".join(
        line for line in run.splitlines() if not line.lstrip().startswith("#")
    )


def _gate(
    surface: FakeSurface,
    monkeypatch: pytest.MonkeyPatch,
    *,
    lanes: list[EnumLabLane],
    required: list[EnumLabLane] | None = None,
) -> int:
    monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)
    return evaluate_gate(
        REPO,
        SHA_4ACA,
        lanes,
        io.StringIO(),
        required=required or [],
        now=lambda: ts("2026-09-25T02:00:00Z"),
    )


# --- AC1 ----------------------------------------------------------------------


def test_sibling_compose_dev_202_is_added_for_omnimarket_only() -> None:
    code = _code_lines(_step_run(SIBLING_STEP))
    assert "SIBLING_LANES=(--lane compose-dev)" in code
    guarded = re.search(
        r'if \[ "\$\{PINNED_REPO\}" = "omnimarket" \]; then\s+'
        r"SIBLING_LANES\+=\(--lane compose-dev-202\)\s+fi",
        code,
    )
    assert guarded, "compose-dev-202 is added only when the pinned repo is omnimarket"
    assert code.count("compose-dev-202") == 1
    assert '"${SIBLING_LANES[@]}"' in code


@pytest.mark.parametrize(
    "present",
    [[EnumLabLane.COMPOSE_DEV], [EnumLabLane.COMPOSE_DEV_202]],
    ids=["compose-dev", "compose-dev-202"],
)
def test_sibling_compose_dev_202_either_lane_admits(
    monkeypatch: pytest.MonkeyPatch, present: list[EnumLabLane]
) -> None:
    surface = FakeSurface()
    surface.add(*(receipt(SHA_4ACA, lane) for lane in present))
    lanes = [EnumLabLane.COMPOSE_DEV, EnumLabLane.COMPOSE_DEV_202]
    assert _gate(surface, monkeypatch, lanes=lanes) == 0


def test_sibling_compose_dev_202_neither_lane_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = FakeSurface()
    surface.add(receipt(SHA_4ACA, EnumLabLane.ONEX_LAB))
    lanes = [EnumLabLane.COMPOSE_DEV, EnumLabLane.COMPOSE_DEV_202]
    assert _gate(surface, monkeypatch, lanes=lanes) != 0


def test_sibling_compose_dev_202_other_siblings_refuse_a_202_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flipped sibling: a non-omnimarket pin reads compose-dev alone."""
    surface = FakeSurface()
    surface.add(receipt(SHA_4ACA, EnumLabLane.COMPOSE_DEV_202))
    assert _gate(surface, monkeypatch, lanes=[EnumLabLane.COMPOSE_DEV]) != 0


# --- AC2 ----------------------------------------------------------------------


def test_own_sha_requires_compose_dev_and_never_names_202() -> None:
    code = _code_lines(_step_run(OWN_SHA_STEP))
    assert "--require-lane compose-dev" in code
    assert "compose-dev-202" not in code


def test_own_sha_requires_compose_dev_a_202_pass_alone_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = FakeSurface()
    surface.add(
        receipt(SHA_4ACA, EnumLabLane.COMPOSE_DEV_202),
        receipt(SHA_4ACA, EnumLabLane.ONEX_LAB),
    )
    code = _gate(
        surface,
        monkeypatch,
        lanes=list(ANY_OF_DEFAULT_LANES),
        required=[EnumLabLane.COMPOSE_DEV],
    )
    assert code != 0


def test_own_sha_requires_compose_dev_positive_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = FakeSurface()
    surface.add(receipt(SHA_4ACA, EnumLabLane.COMPOSE_DEV))
    code = _gate(
        surface,
        monkeypatch,
        lanes=list(ANY_OF_DEFAULT_LANES),
        required=[EnumLabLane.COMPOSE_DEV],
    )
    assert code == 0
