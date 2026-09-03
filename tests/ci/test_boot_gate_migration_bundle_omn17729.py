# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17729 (child of OMN-17702) — the boot gate migrates the lane before it
judges the candidate.

WHAT WENT WRONG. ``candidate-boot-gate`` booted the candidate against a lane
that had never had a migration applied to it. The runtime image's ENTRYPOINT
stamps a schema fingerprint into ``public.db_metadata`` in ``omnibase_infra``
and treats that database as PRIMARY/owned, so every runtime-family pod exited
in a shell script before Python started::

    [entrypoint] ERROR: omnibase_infra (PRIMARY/owned DB) fingerprint stamp
    failed -- aborting boot

Byte-identical in ``omninode-runtime``, ``-effects`` and ``-worker``, and
byte-identical across two candidates built from two different commits (delivery
runs 33686311428 and 33719871544). There was no execution path in which the gate
could pass: it is a required predecessor of the announce job, so NO candidate
was announced to omninode_infra between 2026-09-02T13:45:30Z and this fix, and
the gate never once exercised the OMN-17502 / OMN-17510 / OMN-17519 auto-wiring
class it was built to catch.

WHAT THESE TESTS PIN. Not that the migration works -- only a real run proves
that, and OMN-17729's evidence requires one. They pin the properties a future
edit could silently take away and leave a gate that looks the same and proves
less:

* the gate consumes the SAME bundle build the announcement will announce, so the
  schema the candidate boots against is the one shipped beside it;
* the migration is a BARRIER, not a peer -- it runs after the overlay is applied
  and before the readiness wait, so a ``0/1`` in that wait means the candidate
  and not the schema;
* a failed or stalled migration fails the gate rather than falling through;
* the failure is legible on the first read, in the step log and in the artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DELIVER = REPO_ROOT / ".github" / "workflows" / "deliver-dev-candidate-to-staging.yml"

GATE_JOB = "candidate-boot-gate"
MIGRATE_BUILD_JOB = "build-migrate-bundle"
MIGRATE_STEP = "Apply the candidate's migration bundle to the lane"
APPLY_STEP = "Apply"
WAIT_STEP = "Wait for every runtime-family Deployment and the projection topic"


@pytest.fixture(scope="module")
def deliver() -> dict[str, Any]:
    loaded = yaml.safe_load(DELIVER.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture(scope="module")
def gate(deliver: dict[str, Any]) -> dict[str, Any]:
    return dict(deliver["jobs"][GATE_JOB])


def _step_names(job: dict[str, Any]) -> list[str]:
    return [str(step.get("name", "")) for step in job["steps"]]


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    for step in job["steps"]:
        if step.get("name") == name:
            return dict(step)
    msg = f"the {GATE_JOB} job has no step named {name!r}; it has {_step_names(job)}"
    raise AssertionError(msg)


def _commands(step: dict[str, Any]) -> str:
    """The step's shell with comment lines removed.

    The comment block quotes the entrypoint error and names the rejected
    alternatives, so a naive substring check on the raw script would read the
    explanation as the behaviour.
    """
    return "\n".join(
        line
        for line in str(step.get("run", "")).splitlines()
        if not line.lstrip().startswith("#")
    )


# ---------------------------------------------------------------------------
# the bundle under test is the bundle being announced
# ---------------------------------------------------------------------------
def test_gate_depends_on_the_migrate_bundle_build(gate: dict[str, Any]) -> None:
    """Before OMN-17702 the gate needed only the runtime build.

    That is worth stating because the ticket's own first draft said the gate
    "already holds" the migrate digest. It did not -- the ANNOUNCE job did. A
    gate that resolved the bundle any other way (newest in ECR, a static pin)
    would migrate the lane with a schema that is not the one shipping.
    """
    assert MIGRATE_BUILD_JOB in gate["needs"], (
        f"{GATE_JOB} does not depend on {MIGRATE_BUILD_JOB}, so it cannot boot "
        "the candidate against the schema that is about to be announced with it."
    )


def test_the_migration_step_pins_that_builds_own_output(gate: dict[str, Any]) -> None:
    step = _step(gate, MIGRATE_STEP)
    env = step.get("env") or {}
    assert any(
        f"needs.{MIGRATE_BUILD_JOB}.outputs.image-ref" in str(value)
        for value in env.values()
    ), (
        "the migration step does not take its image from "
        f"needs.{MIGRATE_BUILD_JOB}.outputs.image-ref. Resolving the bundle any "
        "other way decouples the schema from the candidate."
    )
    assert "--migrate-image" in _commands(step), (
        "the rendered Job must be re-pinned to this run's bundle; without "
        "--migrate-image it runs whatever digest the manifest happens to carry."
    )


def test_the_migration_step_refuses_an_unpinned_bundle(gate: dict[str, Any]) -> None:
    commands = _commands(_step(gate, MIGRATE_STEP))
    assert "*@sha256:*" in commands, (
        "the step does not check that the bundle reference is digest-pinned. A "
        "mutable tag is not a migratable identity, and the Job's own IMAGE_TAG "
        "guard only catches the literal string 'latest'."
    )


# ---------------------------------------------------------------------------
# it is a barrier, not a peer
# ---------------------------------------------------------------------------
def test_the_migration_runs_after_apply_and_before_the_readiness_wait(
    gate: dict[str, Any],
) -> None:
    """Ordering IS the fix.

    In the overlay it would race the Deployments (kustomize renders one document
    set and kubectl apply has no ordering), and the runtime pods would
    crash-loop until it finished. Placed here the readiness wait cannot start
    until the migration has completed, so a NOT READY in that step means the
    candidate.
    """
    names = _step_names(gate)
    for required in (APPLY_STEP, MIGRATE_STEP, WAIT_STEP):
        assert required in names, f"the gate lost its {required!r} step"
    assert (
        names.index(APPLY_STEP) < names.index(MIGRATE_STEP) < names.index(WAIT_STEP)
    ), (
        "the migration must sit between the overlay apply (which creates the "
        f"lane Postgres and its Secrets) and the readiness wait. Order is {names}."
    )


def test_the_migration_waits_for_the_lane_postgres_first(gate: dict[str, Any]) -> None:
    """The Job connects on its first statement and gives up after MAX_RETRIES."""
    commands = _commands(_step(gate, MIGRATE_STEP))
    assert "rollout status" in commands and "LAB_DB_SERVICE" in commands, (
        "the step does not wait for the lane's Postgres. A slow initdb would be "
        "reported as a failed migration."
    )


def test_the_job_is_rendered_for_the_lane_and_not_applied_as_shipped(
    gate: dict[str, Any],
) -> None:
    """The manifest targets `data-plane` and RDS; the lane is neither.

    ``metadata.namespace`` in particular cannot be a ``kubectl apply -n`` flag:
    an explicit namespace in the manifest wins, and the Job would land where
    neither its Secrets nor the server are. ``--postgres-target in-cluster`` is
    equally load-bearing -- on the ``rds`` branch the runner does cluster-role
    DDL as the ordinary service login, and migrations 052/094/103 abort.
    """
    commands = _commands(_step(gate, MIGRATE_STEP))
    assert "render_migration_manifest_for_target.py" in commands, (
        "the step must render omninode_infra's own manifest through "
        "omninode_infra's own renderer, never a copy vendored into this repo."
    )
    for flag in ("--namespace", "--postgres-target in-cluster", "--pgsslmode disable"):
        assert flag in commands, f"the render is missing {flag!r}"


# ---------------------------------------------------------------------------
# fail-closed and legible
# ---------------------------------------------------------------------------
def test_a_failed_migration_fails_the_gate(gate: dict[str, Any]) -> None:
    """Falling through would report the schema failure as a candidate defect.

    That misreading is the whole of OMN-17702: a `0/1 NOT READY` was read on
    2026-09-02 as "the readiness signature matches the onex-dev failure" when it
    was a different failure at an earlier stage.
    """
    step = _step(gate, MIGRATE_STEP)
    assert step.get("continue-on-error") in (None, False), (
        "continue-on-error on the migration step turns the barrier back into a "
        "suggestion."
    )
    assert step.get("if") is None, (
        "an `if:` here is a way to let an unmigrated lane through; the step must "
        "run on every gate execution."
    )
    commands = _commands(step)
    assert 'type=="Failed"' in commands, (
        "the step does not read the Job's Failed condition, so a bundle that "
        "died immediately would burn the whole timeout and be reported as "
        "'timed out' rather than 'failed'."
    )
    assert commands.count("exit 1") >= 3, (
        "the step must exit non-zero on an unpinned reference, a failed Job and "
        "a stalled Job. Found fewer explicit failures than that."
    )


def test_the_migration_logs_reach_the_diagnostics_artifact(
    gate: dict[str, Any],
) -> None:
    """A Job is invisible in the Deployment table the gate already collects."""
    diagnostics = _commands(_step(gate, "Collect diagnostics"))
    assert "get jobs" in diagnostics, (
        "the diagnostics do not record the Jobs in the namespace, so a failed "
        "migration's pod logs arrive with nothing naming which Job they came "
        "from."
    )


def test_the_summary_names_the_bundle_the_lane_was_migrated_with(
    gate: dict[str, Any],
) -> None:
    """A green run's summary must say what schema it proved the candidate on."""
    summary = str(_step(gate, "Summarise").get("run", ""))
    assert f"needs.{MIGRATE_BUILD_JOB}.outputs.image-ref" in summary, (
        "the step summary names the candidate but not the migrate bundle, so a "
        "reader cannot tell which schema the candidate booted against."
    )
