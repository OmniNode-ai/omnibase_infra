# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18438 -- the agent path must run the dev lane's omninode_cloud one-shots.

``omnibase_infra#3636`` put ``cloud-migration-files`` and ``cloud-migration``
into ``DEV_LANE_ONLY_MIGRATION_SERVICES`` in ``scripts/deploy-runtime.sh`` and
wired them into that script's migration preflight. They still never ran. This
is OMN-18108's finding repeated one layer over: **the deploy agent does not
invoke ``deploy-runtime.sh`` at all**, so an array expanded only inside that
script is unreachable from the path that actually deploys the dev lane. Every
mention of ``deploy-runtime.sh`` in the agent package is a comment.

Measured on the .201 dev lane after the 14:43Z governed rebuild (agent command
``ee3d2cc6``, ref ``3586e65dc``, which carried those arrays): zero
``cloud-migration`` containers had ever been created, ``omninode_cloud`` held 0
tables against 78 in ``omnibase_infra``, and 450 lines of deploy journal
contained 0 ``cloud-migration`` mentions against 2 migration controls.

WHY THESE ARE NOT IN ``services_for_scope``
-------------------------------------------
That function answers "what does a runtime deploy restart", and its result is
handed to ``_compose_up`` and to the build set. These two are one-shots on
upstream images -- ``postgres:16`` and the tag-referenced migrate image -- so
they are not built, and starting them alongside the runtime family would run
them unordered and leave the up-readback waiting on containers that are
supposed to exit. They belong in the migration preflight, which is the phase
that already exists for run-to-completion boot-order work, and a test below
pins that placement so a later editor does not "tidy" them into the scope list.

ORDER IS THE ORDERING. ``cloud-migration-files`` copies the corpus and its
MANIFEST out of the migrate image into the shared volume; ``cloud-migration``
then applies it. ``--no-deps``, which every one of these commands uses, is
exactly what switches compose's ``depends_on`` off.

Ticket: OMN-18438. Prior layer: OMN-18108. Epic: OMN-17530.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent.events import (
    DEV_LANE_ONLY_MIGRATION_SERVICES,
    DEV_LANE_ONLY_RUNTIME_SERVICES,
    EnumRuntimeLane,
    Scope,
    services_for_scope,
)
from deploy_agent.executor import (
    RUNTIME_MIGRATION_SERVICES,
    DeployExecutor,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"
DEV_LANE_OVERLAY = REPO_ROOT / "docker" / "docker-compose.dev-lane.yml"

CLOUD_ONESHOTS = ("cloud-migration-files", "cloud-migration")


def _bash_array(name: str) -> list[str]:
    """Return the entries of a ``readonly NAME=( ... )`` array in the script.

    Parsed rather than duplicated -- the same helper contract as the OMN-18108
    anti-drift test, which is the file this one is modelled on.
    """
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        rf"^readonly\s+{re.escape(name)}=\((?P<body>.*?)^\)", text, re.M | re.S
    )
    if match is None:
        raise AssertionError(f"{name} array not found in {DEPLOY_SCRIPT}")
    entries: list[str] = []
    for raw_line in match.group("body").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            entries.extend(line.split())
    return entries


def _ok() -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


# ---------------------------------------------------------------------------
# Positive control -- an empty parse must never read as agreement
# ---------------------------------------------------------------------------


class TestTheArrayIsFoundAtAll:
    def test_bash_migration_array_is_non_empty(self) -> None:
        assert len(_bash_array("DEV_LANE_ONLY_MIGRATION_SERVICES")) == 2

    def test_python_declaration_is_non_empty(self) -> None:
        assert len(DEV_LANE_ONLY_MIGRATION_SERVICES) == 2


# ---------------------------------------------------------------------------
# The bidirectional binding -- one declaration, two readers
# ---------------------------------------------------------------------------


class TestTheTwoDeclarationsCannotDrift:
    def test_python_declaration_equals_the_bash_array(self) -> None:
        """The whole point: a service added on one side and not the other is
        the OMN-18108 / OMN-18438 defect recurring, and it is invisible at
        runtime because the lane simply keeps not running the one-shot."""
        assert list(DEV_LANE_ONLY_MIGRATION_SERVICES) == _bash_array(
            "DEV_LANE_ONLY_MIGRATION_SERVICES"
        )

    def test_order_is_preserved_because_order_is_the_ordering(self) -> None:
        """``--no-deps`` switches compose's ``depends_on`` off, so the sequence
        in this tuple is the only thing sequencing the copy before the apply."""
        assert tuple(DEV_LANE_ONLY_MIGRATION_SERVICES) == CLOUD_ONESHOTS

    def test_every_member_is_a_one_shot_on_the_bash_side(self) -> None:
        """Why Python needs one tuple where bash carries two arrays.

        ``deploy-runtime.sh`` separates services from one-shots because its
        lane-agnostic set mixes a keepalive (``migration-gate``) in. Every
        member of the dev-lane set is a one-shot, so a second Python tuple
        would be a second thing to drift rather than a distinction.
        """
        assert _bash_array("DEV_LANE_ONLY_MIGRATION_ONESHOTS") == _bash_array(
            "DEV_LANE_ONLY_MIGRATION_SERVICES"
        )

    def test_no_duplicate_entries_on_either_side(self) -> None:
        bash = _bash_array("DEV_LANE_ONLY_MIGRATION_SERVICES")
        assert len(set(bash)) == len(bash)
        assert len(set(DEV_LANE_ONLY_MIGRATION_SERVICES)) == len(
            DEV_LANE_ONLY_MIGRATION_SERVICES
        )


# ---------------------------------------------------------------------------
# Placement -- the migration preflight, not the runtime restart set
# ---------------------------------------------------------------------------


class TestPlacement:
    def test_one_shots_are_not_in_the_dev_runtime_restart_set(self) -> None:
        """They are not restarted with the runtime family; they are run to
        completion before it. A member here would be built, started unordered,
        and waited on as though it were supposed to stay up."""
        for service in DEV_LANE_ONLY_MIGRATION_SERVICES:
            assert service not in DEV_LANE_ONLY_RUNTIME_SERVICES

    @pytest.mark.parametrize(
        "scope", [Scope.CORE, Scope.RUNTIME, Scope.FULL], ids=lambda s: str(s)
    )
    def test_services_for_scope_never_returns_a_one_shot(self, scope: Scope) -> None:
        resolved = services_for_scope(scope, lane=EnumRuntimeLane.DEV)
        for service in DEV_LANE_ONLY_MIGRATION_SERVICES:
            assert service not in resolved

    def test_one_shots_are_not_in_the_lane_agnostic_migration_set(self) -> None:
        """``RUNTIME_MIGRATION_SERVICES`` is named on every lane. prod,
        stability-test and judge declare neither of these, and compose fails the
        WHOLE invocation on one undefined service name."""
        for service in DEV_LANE_ONLY_MIGRATION_SERVICES:
            assert service not in RUNTIME_MIGRATION_SERVICES


# ---------------------------------------------------------------------------
# The behaviour -- the preflight actually runs them, in order, on DEV only
# ---------------------------------------------------------------------------


def _preflight_calls(lane: EnumRuntimeLane) -> list[list[str]]:
    """Run ``_ensure_runtime_migrations_ready`` with every effect mocked and
    return the compose commands it issued."""
    executor = DeployExecutor()
    calls: list[list[str]] = []

    def _record(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(list(cmd))
        if "psql" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="t\n", stderr=""
            )
        return _ok()

    with (
        patch("deploy_agent.executor._run", side_effect=_record),
        patch(
            "deploy_agent.executor.verify_containers_up",
            return_value=(True, []),
        ),
        patch(
            "deploy_agent.executor.verify_oneshots_completed",
            return_value=(True, []),
        ),
    ):
        executor._ensure_runtime_migrations_ready(lane=lane)
    return calls


def _services_started(calls: list[list[str]]) -> list[str]:
    """The service name each compose ``up`` command targeted, in order."""
    return [cmd[-1] for cmd in calls if "up" in cmd]


class TestThePreflightRunsThem:
    def test_dev_preflight_starts_both_one_shots(self) -> None:
        started = _services_started(_preflight_calls(EnumRuntimeLane.DEV))
        for service in DEV_LANE_ONLY_MIGRATION_SERVICES:
            assert service in started, (
                f"the DEV migration preflight never started {service} -- this is "
                "the defect: the agent does not call deploy-runtime.sh, so an "
                "array declared only there is unreachable"
            )

    def test_the_corpus_copy_precedes_the_apply(self) -> None:
        started = _services_started(_preflight_calls(EnumRuntimeLane.DEV))
        assert started.index("cloud-migration-files") < started.index(
            "cloud-migration"
        ), (
            "cloud-migration ran before the corpus was copied; --no-deps means "
            "this ordering is not enforced by compose"
        )

    def test_the_one_shots_run_after_the_lane_agnostic_migrations(self) -> None:
        started = _services_started(_preflight_calls(EnumRuntimeLane.DEV))
        assert started.index("forward-migration") < started.index(
            "cloud-migration-files"
        ), (
            "the omninode_cloud corpus was applied before the forward stream; "
            "the lane-agnostic set is the boot-order contract and runs first"
        )

    def test_every_start_forces_a_recreate(self) -> None:
        """A one-shot that exited on a previous deploy is not restarted by a
        plain ``up`` -- compose considers it converged. Without
        ``--force-recreate`` a new migrate image tag never reaches the database.
        """
        calls = _preflight_calls(EnumRuntimeLane.DEV)
        for cmd in calls:
            if "up" in cmd and cmd[-1] in DEV_LANE_ONLY_MIGRATION_SERVICES:
                assert "--force-recreate" in cmd

    def test_the_dev_lane_overlay_is_passed(self) -> None:
        """These services exist only in the lane overlay. Without it on the
        command line compose fails the whole invocation as undefined."""
        calls = _preflight_calls(EnumRuntimeLane.DEV)
        for cmd in calls:
            if "up" in cmd and cmd[-1] in DEV_LANE_ONLY_MIGRATION_SERVICES:
                assert any(DEV_LANE_OVERLAY.name in token for token in cmd)

    @pytest.mark.parametrize(
        "lane",
        [EnumRuntimeLane.PROD, EnumRuntimeLane.STABILITY_TEST],
        ids=lambda lane: str(lane),
    )
    def test_no_other_lane_is_handed_a_cloud_one_shot(
        self, lane: EnumRuntimeLane
    ) -> None:
        """The fail-closed direction. Those lanes declare neither service, so a
        leak here turns a dev-lane fix into a deploy outage on the proof lane."""
        started = _services_started(_preflight_calls(lane))
        for service in DEV_LANE_ONLY_MIGRATION_SERVICES:
            assert service not in started


class TestExitZeroGating:
    def test_a_one_shot_that_did_not_exit_zero_fails_the_preflight(self) -> None:
        """A one-shot that is merely RUNNING is not done.

        ``_service_satisfied`` deliberately accepts ``running`` for the
        lane-agnostic set, which mixes in a keepalive. Applied to a corpus
        migration that tolerance would let the runtime start against a
        half-applied database and report success.
        """
        executor = DeployExecutor()

        def _record(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            if "psql" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="t\n", stderr=""
                )
            return _ok()

        with (
            patch("deploy_agent.executor._run", side_effect=_record),
            patch(
                "deploy_agent.executor.verify_containers_up",
                return_value=(True, []),
            ),
            patch(
                "deploy_agent.executor.verify_oneshots_completed",
                return_value=(False, ["cloud-migration"]),
            ),
            pytest.raises(RuntimeError, match="cloud-migration"),
        ):
            executor._ensure_runtime_migrations_ready(lane=EnumRuntimeLane.DEV)


# ---------------------------------------------------------------------------
# The compose half -- the names must resolve on the lane that gets them
# ---------------------------------------------------------------------------


class TestTheOverlayDeclaresThem:
    def test_dev_lane_overlay_declares_both_services(self) -> None:
        text = DEV_LANE_OVERLAY.read_text(encoding="utf-8")
        for service in DEV_LANE_ONLY_MIGRATION_SERVICES:
            assert re.search(rf"^  {re.escape(service)}:$", text, re.M), (
                f"{DEV_LANE_OVERLAY.name} declares no {service}; the preflight "
                "would name a service compose cannot resolve and fail the whole up"
            )
