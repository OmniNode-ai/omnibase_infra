# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18438: a governed warm refresh must run the dev lane's cloud one-shots.

``docker/docker-compose.dev-lane.yml`` declares ``cloud-migration-files`` and
``cloud-migration`` (OMN-17530, ``omnibase_infra#3332``). They apply the
``omninode_cloud`` corpus that ``onex-api`` owns ``tenants`` and
``tenant_api_keys`` in. Nothing ran them: ``scripts/deploy-runtime.sh`` excluded
them from every refresh set because the RT-6 readback resolved a RUNNING
container and a one-shot has already exited by then. ``omnibase_infra#3330``
closed that blocker with a one-shot-aware readback, but no refresh set gained
the two services afterwards.

Measured on the .201 compose dev lane 2026-09-16: ``docker ps -a`` showed **no
container had ever existed** for either service, and ``omninode_cloud`` held 0
tables against 78 in ``omnibase_infra``.

The fix cannot be an entry in ``RUNTIME_MIGRATION_SERVICES``. That array is
lane-agnostic -- every lane's ``docker compose ... up`` names it verbatim, and
``docker compose up`` fails the WHOLE invocation on one undefined service name,
so a lane-agnostic entry would break every prod, stability-test and judge deploy
at its first migration step. It must be a lane-scoped array, keyed the way
``resolve_lane_overlay_filename`` keys the existing lane-only runtime arrays.

Ticket: OMN-18438. Prerequisite: OMN-16729 (``#3330``). Epic: OMN-17530.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

DEPLOY_RUNTIME = REPO_ROOT / "scripts" / "deploy-runtime.sh"
DEV_LANE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.dev-lane.yml"

CLOUD_ONESHOTS = ("cloud-migration-files", "cloud-migration")
DEV_LANE_ARRAY = "DEV_LANE_ONLY_MIGRATION_SERVICES"
DEV_LANE_ONESHOT_ARRAY = "DEV_LANE_ONLY_MIGRATION_ONESHOTS"
PREFLIGHT_FUNCTION = "run_runtime_migration_preflight"

# Every lane overlay that merges the base compose file. None of these declares
# the cloud one-shots, which is exactly why the lane-agnostic array is fatal.
LANES_WITHOUT_THE_ONESHOTS = (
    "docker-compose.stability-test.yml",
    "docker-compose.prod.yml",
    "docker-compose.judge.yml",
)


def _script_text() -> str:
    return DEPLOY_RUNTIME.read_text(encoding="utf-8")


def _array_entries(name: str) -> list[str]:
    """The entries of a ``readonly NAME=( ... )`` array, comments dropped."""
    text = _script_text()
    match = re.search(rf"^readonly {name}=\((.*?)^\)", text, re.MULTILINE | re.DOTALL)
    assert match is not None, (
        f"{DEPLOY_RUNTIME.name} declares no {name} array -- the dev lane's "
        "omninode_cloud one-shots have no refresh set to belong to"
    )
    return [
        line.strip()
        for line in match.group(1).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _function_body(name: str) -> str:
    """A shell function body, from its opening brace to the next top-level one."""
    text = _script_text()
    start = text.index(f"{name}() {{")
    end = text.index("\n}\n", start)
    return text[start:end]


# ---------------------------------------------------------------------------
# AC3 -- the one-shots have a refresh set, and it is lane-scoped
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dev_lane_migration_array_carries_both_cloud_oneshots() -> None:
    """AC3: both services are in the dev-lane migration set, in order.

    ``cloud-migration-files`` copies the corpus into the shared volume;
    ``cloud-migration`` applies it. A warm ``up -d --no-deps`` is exactly what
    switches ``depends_on`` off, so the ORDER in this array is the ordering.
    """
    entries = _array_entries(DEV_LANE_ARRAY)

    assert entries == list(CLOUD_ONESHOTS), (
        f"{DEV_LANE_ARRAY} is {entries}; it must be {list(CLOUD_ONESHOTS)} -- the "
        "files one-shot copies the corpus the migration one-shot then applies, "
        "and --no-deps means this array is the only thing ordering them"
    )


@pytest.mark.unit
def test_both_cloud_services_are_declared_one_shots() -> None:
    """AC3: the preflight must ``docker wait`` on each, not just start it.

    A one-shot that is started and never waited on lets the runtime restart race
    an unapplied corpus -- the failure OMN-13220 fixed for
    ``intelligence-migration``.
    """
    oneshots = _array_entries(DEV_LANE_ONESHOT_ARRAY)

    for service in CLOUD_ONESHOTS:
        assert service in oneshots, (
            f"{service} is not in {DEV_LANE_ONESHOT_ARRAY}, so the preflight "
            "would start it and move on without proving it exited 0"
        )


@pytest.mark.unit
def test_cloud_oneshots_never_join_the_lane_agnostic_array() -> None:
    """AC3: the lane-agnostic array is the one place they must NOT appear.

    ``RUNTIME_MIGRATION_SERVICES`` is named verbatim by every lane's
    ``docker compose up``, and compose fails the WHOLE invocation on one
    undefined service. prod, stability-test and judge declare neither of these
    services, so an entry there turns a dev-lane fix into a production outage.
    """
    for array in ("RUNTIME_MIGRATION_SERVICES", "RUNTIME_MIGRATION_ONESHOTS"):
        entries = _array_entries(array)
        for service in CLOUD_ONESHOTS:
            assert service not in entries, (
                f"{service} is in the lane-agnostic {array}; prod, "
                "stability-test and judge declare no such service and their "
                "next deploy would fail at the migration preflight"
            )


@pytest.mark.unit
def test_preflight_consults_the_dev_lane_array() -> None:
    """AC3: an array nothing reads is a comment.

    Both prior lane-only arrays are keyed on the resolved overlay filename
    rather than on a lane string, so an unknown lane falls through to the
    fail-closed default of naming nothing.
    """
    body = _function_body(PREFLIGHT_FUNCTION)

    assert DEV_LANE_ARRAY in body, (
        f"{PREFLIGHT_FUNCTION} never reads {DEV_LANE_ARRAY} -- the array exists "
        "and no refresh would run the one-shots it names"
    )
    assert "resolve_lane_overlay_filename" in body, (
        f"{PREFLIGHT_FUNCTION} does not resolve the lane overlay, so it cannot "
        "scope these services to the only lane that declares them"
    )


@pytest.mark.unit
def test_preflight_is_fail_closed_for_lanes_without_the_overlay() -> None:
    """AC3: a lane that is not the dev lane must be handed no extra service.

    The existing build resolver states this contract in its own default branch:
    naming a service that exists in none of a lane's compose files fails the
    whole ``docker compose up``.
    """
    body = _function_body(PREFLIGHT_FUNCTION)

    # The dev lane is the bare `omnibase-infra` project -- resolve_lane_overlay_
    # filename returns the empty string for it, and every other lane returns a
    # filename. A case statement keyed that way is the fail-closed shape.
    assert re.search(r'case\s+"\$\{?\w*overlay\w*\}?"\s+in', body), (
        f"{PREFLIGHT_FUNCTION} does not branch on the resolved overlay filename; "
        "without that branch the extra services reach every lane"
    )


# ---------------------------------------------------------------------------
# The compose half -- the services the array names must exist on that lane
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dev_lane_overlay_declares_both_services() -> None:
    """The array names services; compose must declare them, or every up fails."""
    text = DEV_LANE_COMPOSE.read_text(encoding="utf-8").replace("!override", "")
    compose = yaml.safe_load(text) or {}
    services = compose.get("services", {})

    for service in CLOUD_ONESHOTS:
        assert service in services, (
            f"{DEV_LANE_COMPOSE.name} declares no {service}; "
            f"{DEV_LANE_ARRAY} would name a service compose cannot resolve"
        )


@pytest.mark.unit
def test_both_services_are_one_shots_in_compose() -> None:
    """``restart: "no"`` is what makes the RT-6 readback treat them as one-shots.

    ``omnibase_infra#3330`` partitions that readback by restart policy. A
    service in this array that compose declares long-running would be certified
    against the wrong assertion.
    """
    text = DEV_LANE_COMPOSE.read_text(encoding="utf-8").replace("!override", "")
    compose = yaml.safe_load(text) or {}

    for service in CLOUD_ONESHOTS:
        restart = compose["services"][service].get("restart")
        assert str(restart) == "no", (
            f"{service} declares restart={restart!r}; the one-shot-aware RT-6 "
            'readback keys on restart: "no"'
        )


@pytest.mark.unit
@pytest.mark.parametrize("overlay", LANES_WITHOUT_THE_ONESHOTS)
def test_no_other_lane_declares_the_cloud_oneshots(overlay: str) -> None:
    """The premise the lane scoping rests on, asserted rather than assumed.

    If another lane ever declares these services, the scoping above is wrong in
    a way no other test would notice -- it would silently skip a lane that could
    have run them.
    """
    path = REPO_ROOT / "docker" / overlay
    if not path.exists():
        pytest.skip(f"{overlay} is not present in this tree")

    text = path.read_text(encoding="utf-8").replace("!override", "")
    services = (yaml.safe_load(text) or {}).get("services", {})

    for service in CLOUD_ONESHOTS:
        assert service not in services, (
            f"{overlay} now declares {service}; the OMN-18438 lane scoping "
            "assumes only the dev lane does, and must be widened deliberately"
        )
