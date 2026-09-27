# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19507 AC1: the compose-dev-202 receipt lane, and what it may not stand in for.

Task B5 of the second-deploy-slot plan (epic OMN-19500). The second deployed
dev lane, ``dev-202`` on the ``.202`` host, emits its lab-pass receipt under its
own lane name, because ``evaluate_gate`` reads one emitter per name. The
operator ruled (2026-09-25T00:56:45Z) that this receipt proves omnimarket
changes only; omnibase_infra stays proven on ``.201``. So the value exists, it
is a legal ``--lane`` for emit and verify, its artifact name follows the
convention, and it is NOT in ``ANY_OF_DEFAULT_LANES``: an unqualified gate, or
one that requires ``compose-dev``, refuses a sha whose only PASS is this one.
"""

from __future__ import annotations

import io

import pytest

from scripts.ci.lab_pass_receipt import (
    ANY_OF_DEFAULT_LANES,
    EnumLabLane,
    artifact_name,
    build_parser,
    evaluate_gate,
)
from tests.scripts.ci._lab_pass_fixtures import (
    REPO,
    SHA_4ACA,
    FakeSurface,
    receipt,
    ts,
)

pytestmark = pytest.mark.unit

LANE = EnumLabLane.COMPOSE_DEV_202


def test_compose_dev_202_is_a_lab_lane_value() -> None:
    assert LANE.value == "compose-dev-202"
    assert EnumLabLane("compose-dev-202") is LANE


def test_compose_dev_202_is_not_in_the_any_of_default_lanes() -> None:
    assert LANE not in ANY_OF_DEFAULT_LANES
    assert (
        EnumLabLane.COMPOSE_DEV,
        EnumLabLane.ONEX_LAB,
        EnumLabLane.ONEX_LAB_K3S,
    ) == ANY_OF_DEFAULT_LANES, "adding the lane must not widen the any-of premise"


def test_compose_dev_202_artifact_name_follows_the_convention() -> None:
    assert (
        artifact_name(LANE, SHA_4ACA) == f"lab-pass-receipt-compose-dev-202-{SHA_4ACA}"
    )


def _gate(
    surface: FakeSurface,
    monkeypatch: pytest.MonkeyPatch,
    *,
    required: list[EnumLabLane],
) -> tuple[int, str]:
    monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)
    out = io.StringIO()
    code = evaluate_gate(
        REPO,
        SHA_4ACA,
        list(ANY_OF_DEFAULT_LANES),
        out,
        required=required,
        now=lambda: ts("2026-09-25T02:00:00Z"),
    )
    return code, out.getvalue()


def test_compose_dev_202_pass_alone_never_satisfies_an_unqualified_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = FakeSurface()
    surface.add(receipt(SHA_4ACA, LANE))
    code, _output = _gate(surface, monkeypatch, required=[])
    assert code != 0


def test_compose_dev_202_pass_alone_never_satisfies_a_compose_dev_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = FakeSurface()
    surface.add(receipt(SHA_4ACA, LANE))
    code, output = _gate(surface, monkeypatch, required=[EnumLabLane.COMPOSE_DEV])
    assert code != 0
    assert "compose-dev" in output


def test_compose_dev_202_pass_satisfies_a_gate_that_requires_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: the lane is readable when a caller names it."""
    surface = FakeSurface()
    surface.add(receipt(SHA_4ACA, LANE), receipt(SHA_4ACA, EnumLabLane.ONEX_LAB))
    code, output = _gate(surface, monkeypatch, required=[LANE])
    assert code == 0, output


_SHA_ARG = ["--sha", SHA_4ACA]
_REQUIRED_ARGS: dict[str, list[str]] = {
    "emit": [
        *_SHA_ARG,
        "--started-at",
        "2026-09-25T02:00:00Z",
        "--finished-at",
        "2026-09-25T02:10:00Z",
        "--out",
        "receipt.json",
    ],
    "verify": [*_SHA_ARG, "--path", "receipt.json"],
}


@pytest.mark.parametrize("subcommand", sorted(_REQUIRED_ARGS))
def test_compose_dev_202_is_an_accepted_lane_argument(subcommand: str) -> None:
    args = build_parser().parse_args(
        [subcommand, "--lane", "compose-dev-202", *_REQUIRED_ARGS[subcommand]]
    )
    assert args.lane == "compose-dev-202"
