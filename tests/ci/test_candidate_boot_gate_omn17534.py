# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17534 — a candidate cannot be announced to omninode_infra until it has
booted against the real onex-dev manifests.

WHAT THIS GATE IS FOR. Between "the image built" and "staging runs it" there was
nothing. Three times on 2026-09-01/02 a candidate passed unit and integration
tests, was announced, was applied to onex-dev, and took the runtime plane down:

* OMN-17502 — ``BIFROST_LANE_OVERLAY_PATH`` bound by no manifest; the fail-closed
  Bifrost renderer put omninode-runtime, -effects and -worker into
  CrashLoopBackOff.
* OMN-17510 — ``HandlerSavingsCorrelation`` ``TypeError`` at
  ``service_handler_resolver.py:226``, unregistered pool, fatal under strict mode.
* OMN-17519 — strict-mode auto-wiring ``ValueError``: projection handlers require
  ``tenant_projection:ONEX_TENANT_DB_URL``, which no onex-dev manifest binds.

Every one is a BOOT failure against the real manifests. None is visible to a
compose lane, because ``ONEX_WIRING_STRICT_MODE`` is ``${...:-0}`` in
``docker-compose.infra.yml`` and no lane overlay sets it, while onex-dev sets
``"1"``; and because the Bifrost overlay is a bind mount in compose and must be a
ConfigMap in k8s.

These tests are static-artifact assertions on the workflow, plus unit coverage of
``scripts/ci/boot_gate.py``. They cannot prove the gate boots anything -- only a
real run does that, and OMN-17534's evidence section requires one. What they DO
prove is the set of properties a passing run would otherwise silently lose: that
the announce step cannot fire without the gate, that the gate reads the staging
pin with a token that cannot write it, that the diagnostics are complete enough
to diagnose from once, and that no credential value reaches an artifact.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci import boot_gate

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DELIVER = REPO_ROOT / ".github" / "workflows" / "deliver-dev-candidate-to-staging.yml"

GATE_JOB = "candidate-boot-gate"
ANNOUNCE_JOB = "dispatch-to-staging"
BUILD_JOB = "build-runtime-candidate"


@pytest.fixture(scope="module")
def deliver() -> dict[str, Any]:
    loaded = yaml.safe_load(DELIVER.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture(scope="module")
def gate(deliver: dict[str, Any]) -> dict[str, Any]:
    assert GATE_JOB in deliver["jobs"], (
        f"{DELIVER.name} has no {GATE_JOB!r} job. Without it every candidate is "
        "announced on the strength of unit tests alone, which is the state that "
        "took onex-dev down three times in two days."
    )
    return deliver["jobs"][GATE_JOB]


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return list(job.get("steps") or [])


def _run_text(job: dict[str, Any]) -> str:
    return "\n".join(step.get("run", "") for step in _steps(job))


def _run_commands(job: dict[str, Any]) -> str:
    """The job's shell with comment lines removed.

    Needed because these assertions are about what the gate DOES. The workflow
    explains why it takes full logs by quoting the ``--tail=30`` it is replacing,
    and a naive substring check on the raw script would read that explanation as
    the offence it describes.
    """
    lines: list[str] = []
    for step in _steps(job):
        for line in step.get("run", "").splitlines():
            if line.lstrip().startswith("#"):
                continue
            lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AC-1: the announce step cannot execute unless the gate concluded success
# ---------------------------------------------------------------------------
def test_announce_depends_on_the_boot_gate(deliver: dict[str, Any]) -> None:
    needs = deliver["jobs"][ANNOUNCE_JOB]["needs"]
    assert GATE_JOB in needs, (
        f"{ANNOUNCE_JOB} does not depend on {GATE_JOB}. A gate the announcement "
        "does not wait for is a report, not a gate."
    )


def test_announce_has_no_condition_that_could_survive_a_failed_gate(
    deliver: dict[str, Any],
) -> None:
    """A skipped or cancelled gate must not read as a pass.

    GitHub's default ``success()`` already has the wanted semantics. Any ``if:``
    here is a way to reintroduce the hole -- ``always()`` most obviously, but
    equally ``!cancelled()``, which lets a CANCELLED gate through while looking
    conservative.
    """
    condition = deliver["jobs"][ANNOUNCE_JOB].get("if")
    assert condition is None, (
        f"{ANNOUNCE_JOB} declares `if: {condition}`. The default success() gate "
        "is what makes a failed, skipped or cancelled boot gate block the "
        "announcement; overriding it is how that protection gets lost."
    )


def test_gate_runs_after_the_runtime_build(gate: dict[str, Any]) -> None:
    assert BUILD_JOB in gate["needs"], (
        "the gate must consume the built candidate, not rebuild or guess it"
    )


def test_gate_pins_the_built_candidates_own_digest(gate: dict[str, Any]) -> None:
    """The gate must boot the image that is about to be announced.

    Booting `latest`, or the digest the manifests already pin, would produce a
    green gate that says nothing about this candidate.
    """
    text = yaml.safe_dump(gate)
    assert f"needs.{BUILD_JOB}.outputs.image-ref" in text
    assert f"needs.{BUILD_JOB}.outputs.digest" in text
    assert "pin-image" in _run_text(gate)


# ---------------------------------------------------------------------------
# AC-6: reads the staging pin, never writes it
# ---------------------------------------------------------------------------
def test_the_omninode_infra_token_cannot_write(gate: dict[str, Any]) -> None:
    """Not withheld by convention -- not granted.

    The sibling ``dispatch-to-staging`` job in this same file mints
    ``permission-contents: write`` on the same repository, so "the app cannot
    write" is not an argument available here. This job must ask for read.
    """
    token_steps = [
        step
        for step in _steps(gate)
        if str(step.get("uses", "")).startswith("actions/create-github-app-token@")
    ]
    assert token_steps, "the gate mints no omninode_infra token at all"
    for step in token_steps:
        with_block = step.get("with") or {}
        assert with_block.get("permission-contents") == "read", (
            "the boot gate's omninode_infra token must be contents:read "
            f"(got {with_block.get('permission-contents')!r}). OMN-17534 AC-6: the "
            "gate reads the staging pin and never writes it."
        )
        assert with_block.get("repositories") == "omninode_infra"


def test_gate_never_pushes_or_dispatches(gate: dict[str, Any]) -> None:
    run = _run_commands(gate)
    for forbidden in (
        "git push",
        "repos/OmniNode-ai/omninode_infra/dispatches",
        "gh pr ",
    ):
        assert forbidden not in run, (
            f"the boot gate runs {forbidden!r}. It is a read-only prover; the "
            "announcement is a different job with a different token."
        )


# ---------------------------------------------------------------------------
# It must apply the REAL manifests
# ---------------------------------------------------------------------------
def test_gate_renders_the_onex_lab_overlay_and_not_a_local_copy(
    gate: dict[str, Any],
) -> None:
    run = _run_text(gate)
    assert "k8s/onex-lab" in yaml.safe_dump(gate), (
        "the gate must render omninode_infra's k8s/onex-lab overlay -- a thin "
        "overlay over k8s/onex-dev/runtime. A manifest written in this repo "
        "would prove nothing about what staging applies."
    )
    checkouts = [
        step
        for step in _steps(gate)
        if str(step.get("uses", "")).startswith("actions/checkout@")
        and (step.get("with") or {}).get("repository") == "OmniNode-ai/omninode_infra"
    ]
    assert checkouts, "the gate does not check out omninode_infra"
    assert (checkouts[0].get("with") or {}).get("ref") == "dev", (
        "the gate must render the manifests at omninode_infra dev -- the "
        "revision staging actually applies"
    )
    assert "kubectl apply" in run


def test_gate_applies_the_cluster_scoped_prerequisites(gate: dict[str, Any]) -> None:
    """The runtime family declares a cluster-scoped PriorityClass it cannot supply.

    All 14 pod specs carry ``priorityClassName: omninode-standard``. A
    PriorityClass is cluster-scoped, so it lives in omninode_infra's
    ``k8s/base`` and is outside the namespace-scoped kustomization the lab
    overlay bases on. Without it every ReplicaSet is refused with ``no
    PriorityClass with name omninode-standard was found`` and ZERO runtime pods
    are created -- 14 Deployments at 0/1 for a reason that has nothing to do
    with the candidate. The first real run of this gate (33674463837) failed
    exactly that way.

    The file must be applied from omninode_infra's own tree, never re-authored
    here: a second copy of a PriorityClass is a value that can silently drift
    from the one staging schedules against.
    """
    run = _run_commands(gate)
    assert "k8s/base/priority-classes.yaml" in run, (
        "the gate does not apply the cluster-scoped PriorityClass set, so no "
        "runtime pod can be scheduled and every Deployment fails for a lane "
        "reason rather than a candidate reason"
    )
    assert "omninode_infra/k8s/base/priority-classes.yaml" in run, (
        "the PriorityClass manifest must come from the checked-out "
        "omninode_infra tree, not from a copy in this repo"
    )


def test_gate_proves_strict_mode_off_the_cluster(gate: dict[str, Any]) -> None:
    """Strict mode is half the gate's value; assert it, do not assume it.

    At ``ONEX_WIRING_STRICT_MODE=0`` an unwireable handler is quarantined instead
    of killing boot, so a lane that silently ran 0 would go GREEN on OMN-17510
    and OMN-17519 -- two of the three defects this gate exists to catch.
    """
    run = _run_text(gate)
    assert "ONEX_WIRING_STRICT_MODE" in run
    assert "kubectl get deploy" in run, (
        "strict mode must be read back off the applied Deployment, not out of "
        "the render; the render is the thing under test"
    )


def test_readiness_wait_is_at_least_fifteen_minutes(gate: dict[str, Any]) -> None:
    """Scope item 6 says up to 15 minutes, and the probes need it.

    The runtime Deployments carry a startupProbe with failureThreshold 60 at
    period 10s -- 600 seconds -- precisely because contract discovery is slow.
    A shorter wait would report a healthy candidate as broken, and the fix for
    THAT is always to loosen the gate, which is how gates die.
    """
    wait_seconds = int(gate["env"]["BOOT_WAIT_SECONDS"])
    assert wait_seconds >= 900, (
        f"BOOT_WAIT_SECONDS={wait_seconds} is under the 900s the ticket scopes"
    )


def test_cluster_tooling_is_pinned(gate: dict[str, Any]) -> None:
    """An unpinned kind is an unpinned Kubernetes version."""
    env = gate["env"]
    assert re.fullmatch(r"v\d+\.\d+\.\d+", env["KIND_VERSION"]), env["KIND_VERSION"]
    node_image = env["KIND_NODE_IMAGE"]
    assert re.fullmatch(r"kindest/node:v\d+\.\d+\.\d+", node_image), node_image
    assert not node_image.endswith(":latest")


def test_every_third_party_action_is_sha_pinned(gate: dict[str, Any]) -> None:
    for step in _steps(gate):
        uses = str(step.get("uses", ""))
        if not uses or uses.startswith("./"):
            continue
        ref = uses.split("@", 1)[1]
        assert re.fullmatch(r"[0-9a-f]{40}", ref), (
            f"{uses} is not pinned to a full commit SHA"
        )


# ---------------------------------------------------------------------------
# AC-2: the diagnostics must be usable on the FIRST run
# ---------------------------------------------------------------------------
def test_log_capture_is_complete_and_not_truncated(gate: dict[str, Any]) -> None:
    """The 30-line truncation is why OMN-17519 needed a second deploy run.

    Its fatal ``Auto-wiring failed for 2 contract(s)`` block sat behind several
    hundred contract-discovery WARNING lines, and the staging deploy's rollout
    diagnostics read ``kubectl logs --tail=30``. The first run produced no usable
    error at all.
    """
    run = _run_commands(gate)
    assert "kubectl logs" in run
    assert "--tail" not in run, (
        "the gate truncates pod logs. A fatal traceback preceded by hundreds of "
        "discovery WARNINGs would not survive the truncation, and the run would "
        "cost a second run to diagnose -- exactly the OMN-17519 failure."
    )
    assert "--previous" in run, (
        "a CrashLoopBackOff pod's actual death is in the PREVIOUS container; the "
        "current one is usually still starting when the logs are read"
    )
    for expected in ("kubectl describe pod", "kubectl get events", "-o wide"):
        assert expected in run, f"the gate captures no {expected!r}"


def test_diagnostics_are_uploaded_even_when_the_gate_fails(
    gate: dict[str, Any],
) -> None:
    upload_steps = [
        step
        for step in _steps(gate)
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]
    assert upload_steps, "the gate uploads no artifact"
    for step in upload_steps:
        assert str(step.get("if", "")).strip() == "always()", (
            "the artifact must upload on failure -- the failure case is the one "
            "it exists for"
        )
    collect = next(
        step for step in _steps(gate) if step.get("name") == "Collect diagnostics"
    )
    assert str(collect.get("if", "")).strip() == "always()"


def test_the_parity_ledger_travels_with_the_logs(gate: dict[str, Any]) -> None:
    """A green run must never be read in isolation from what it excluded."""
    assert "parity_exclusions.yaml" in _run_text(gate), (
        "the gate does not upload the lane's parity_exclusions.yaml. Without it a "
        "reader cannot tell that the run proves nothing about MSK IAM, RDS, real "
        "Infisical or the migration bundle."
    )


def test_the_render_is_redacted_before_upload(gate: dict[str, Any]) -> None:
    """AC-5: no credential value reaches the artifact.

    The rendered manifest contains the lane's generated Secret payloads. That
    they were generated for this run is not a reason to publish them.
    """
    run = _run_text(gate)
    assert "boot_gate.py redact" in run
    assert "onex-lab-render.redacted.yaml" in run
    upload = next(
        step
        for step in _steps(gate)
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    )
    upload_path = (upload.get("with") or {})["path"]
    # Constructed rather than written literally: this is the path the WORKFLOW
    # renders to on the runner, and a bare "/tmp/..." literal here trips ruff's
    # S108 (insecure temp file) on a string this test never opens.
    unredacted_render = "/".join(("", "tmp", "onex-lab-render.yaml"))
    assert unredacted_render not in upload_path, (
        "the UNREDACTED render is in the artifact path"
    )


def test_secret_rendering_forbids_reaching_for_a_real_credential(
    gate: dict[str, Any],
) -> None:
    assert "--forbid-github-secrets" in _run_text(gate), (
        "the gate must render the lane's CI-only Secrets with "
        "--forbid-github-secrets, which makes 'no real staging credential "
        "reaches a throwaway cluster' a hard error instead of a convention"
    )


# ---------------------------------------------------------------------------
# scripts/ci/boot_gate.py
# ---------------------------------------------------------------------------
def test_pin_image_appends_a_usable_override(tmp_path: Path) -> None:
    kustomization = tmp_path / "kustomization.yaml"
    kustomization.write_text(
        "apiVersion: kustomize.config.k8s.io/v1beta1\nkind: Kustomization\n"
        "resources:\n  - ../onex-dev/runtime\n"
    )
    digest = "sha256:" + "a" * 64
    assert boot_gate.pin_image(kustomization, "registry/repo", digest, None) == 0
    document = yaml.safe_load(kustomization.read_text())
    assert document["resources"] == ["../onex-dev/runtime"]
    assert document["images"] == [
        {"name": "registry/repo", "newName": "registry/repo", "digest": digest}
    ]


def test_pin_image_refuses_a_non_digest(tmp_path: Path) -> None:
    """A mutable tag is not a deployable identity.

    The delivery workflow already refuses to announce one; the gate must refuse
    to BOOT one, or a green gate could describe a different image than the
    announcement does.
    """
    kustomization = tmp_path / "kustomization.yaml"
    kustomization.write_text("kind: Kustomization\n")
    assert boot_gate.pin_image(kustomization, "registry/repo", "latest", None) == 1
    assert "images" not in kustomization.read_text()


def test_pin_image_refuses_to_overwrite_an_existing_pin(tmp_path: Path) -> None:
    """An existing block means the overlay changed upstream -- refuse, don't guess.

    OMN-17533 AC-5 forbids the overlay committing a digest, so a block appearing
    there is a signal, not a merge case. Silently winning would boot an image
    nobody chose.
    """
    kustomization = tmp_path / "kustomization.yaml"
    kustomization.write_text(
        "kind: Kustomization\nimages:\n  - name: registry/repo\n    newTag: something\n"
    )
    digest = "sha256:" + "b" * 64
    assert boot_gate.pin_image(kustomization, "registry/repo", digest, None) == 1
    assert digest not in kustomization.read_text()


def test_redact_strips_secret_values_and_keeps_everything_else(
    tmp_path: Path,
) -> None:
    source = tmp_path / "render.yaml"
    source.write_text(
        "apiVersion: v1\nkind: Secret\nmetadata:\n  name: onex-runtime-credentials\n"
        "data:\n  OMNINODE_INTERNAL_DB_URL: cG9zdGdyZXNxbDovL3NlY3JldA==\n"
        "---\n"
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: omninode-runtime\n"
    )
    destination = tmp_path / "redacted.yaml"
    assert boot_gate.redact_render(source, destination) == 0
    documents = [d for d in yaml.safe_load_all(destination.read_text()) if d]
    secret = next(d for d in documents if d["kind"] == "Secret")
    # The key NAME survives -- it is the useful diagnostic and it is already
    # public in the manifests. The value does not.
    assert "OMNINODE_INTERNAL_DB_URL" in secret["data"]
    assert secret["data"]["OMNINODE_INTERNAL_DB_URL"] == boot_gate.REDACTED
    assert "cG9zdGdyZXNxbDovL3NlY3JldA==" not in destination.read_text()
    assert any(d["kind"] == "Deployment" for d in documents)


def test_wait_reports_every_not_ready_deployment_not_only_the_first(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Naming one offender costs a second run to find the rest.

    The 2026-09-02 staging deploy failed on exactly 3 of 15 Deployments; a gate
    that reported the first would have hidden two of them.
    """
    rows = [
        ("omnimarket-projection-api", 0, 1, "MinimumReplicasUnavailable: x"),
        ("omninode-runtime", 0, 1, "MinimumReplicasUnavailable: y"),
        ("omninode-runtime-effects", 0, 1, ""),
        ("omninode-runtime-worker", 1, 1, ""),
        ("onex-lab-redpanda", 1, 1, ""),
    ]
    monkeypatch.setattr(boot_gate, "_deployment_rows", lambda namespace: rows)
    monkeypatch.setattr(boot_gate, "_topic_exists", lambda *a, **k: True)

    result = boot_gate.wait_for_boot(
        namespace="onex-dev",
        timeout_seconds=0,
        poll_seconds=1,
        lane_prefix="onex-lab-",
        broker_deployment="onex-lab-redpanda",
        require_topic=True,
    )
    assert result == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    for name in (
        "omnimarket-projection-api",
        "omninode-runtime",
        "omninode-runtime-effects",
    ):
        assert name in combined, f"{name} is missing from the failure report"


def test_wait_fails_closed_when_nothing_was_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty namespace is a failure, not a vacuous pass.

    ``all()`` over an empty roster is True, which is exactly how a gate that
    applied nothing reports success.
    """
    monkeypatch.setattr(boot_gate, "_deployment_rows", lambda namespace: [])
    monkeypatch.setattr(boot_gate, "_topic_exists", lambda *a, **k: True)
    assert (
        boot_gate.wait_for_boot(
            namespace="onex-dev",
            timeout_seconds=30,
            poll_seconds=1,
            lane_prefix="onex-lab-",
            broker_deployment="onex-lab-redpanda",
            require_topic=True,
        )
        == 1
    )


def test_wait_still_fails_when_every_deployment_is_ready_but_the_topic_is_absent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ready is not the whole gate.

    omnimarket-projection-api's /ready is fail-closed on
    onex.snapshot.projection.consumer-flow.v1, and on 2026-09-02 that topic was
    absent from MSK entirely while other pods looked fine. A gate that only
    counted Ready replicas would have passed the candidate that caused it.
    """
    rows = [("omninode-runtime", 1, 1, ""), ("onex-lab-redpanda", 1, 1, "")]
    monkeypatch.setattr(boot_gate, "_deployment_rows", lambda namespace: rows)
    monkeypatch.setattr(boot_gate, "_topic_exists", lambda *a, **k: False)
    result = boot_gate.wait_for_boot(
        namespace="onex-dev",
        timeout_seconds=0,
        poll_seconds=1,
        lane_prefix="onex-lab-",
        broker_deployment="onex-lab-redpanda",
        require_topic=True,
    )
    assert result == 1
    combined = "".join(capsys.readouterr())
    assert boot_gate.REQUIRED_TOPIC in combined


def test_wait_passes_only_when_both_conditions_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [("omninode-runtime", 1, 1, ""), ("onex-lab-redpanda", 1, 1, "")]
    monkeypatch.setattr(boot_gate, "_deployment_rows", lambda namespace: rows)
    monkeypatch.setattr(boot_gate, "_topic_exists", lambda *a, **k: True)
    assert (
        boot_gate.wait_for_boot(
            namespace="onex-dev",
            timeout_seconds=60,
            poll_seconds=1,
            lane_prefix="onex-lab-",
            broker_deployment="onex-lab-redpanda",
            require_topic=True,
        )
        == 0
    )


def test_a_deployment_the_manifests_scale_to_zero_counts_as_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four of the onex-dev runtime family are `replicas: 0` in the manifests.

    omnibase-intelligence-api, omninode-agent-actions-consumer,
    omninode-contract-resolver and omninode-skill-lifecycle-consumer all declare
    zero replicas. A readiness predicate of `ready == desired and desired > 0`
    marks each of them NOT READY forever, which makes this gate structurally
    un-passable -- no candidate, however healthy, could satisfy it.

    The first real run (33674463837) reported exactly that: four rows at 0/0.
    This test is the RED that fix is held to.
    """
    rows = [
        ("omnibase-intelligence-api", 0, 0, ""),
        ("omninode-agent-actions-consumer", 0, 0, ""),
        ("omninode-contract-resolver", 0, 0, ""),
        ("omninode-skill-lifecycle-consumer", 0, 0, ""),
        ("omninode-runtime", 1, 1, ""),
        ("onex-lab-redpanda", 1, 1, ""),
    ]
    monkeypatch.setattr(boot_gate, "_deployment_rows", lambda namespace: rows)
    monkeypatch.setattr(boot_gate, "_topic_exists", lambda *a, **k: True)
    assert (
        boot_gate.wait_for_boot(
            namespace="onex-dev",
            timeout_seconds=60,
            poll_seconds=1,
            lane_prefix="onex-lab-",
            broker_deployment="onex-lab-redpanda",
            require_topic=True,
        )
        == 0
    ), "a Deployment the manifests scale to zero must not block the gate forever"


def test_a_plane_scaled_entirely_to_zero_still_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The `desired > 0` guard is moved, not deleted.

    Treating 0/0 as Ready per row would otherwise let a render that scaled the
    WHOLE runtime plane to zero pass vacuously -- every row Ready, nothing
    running, nothing proven. The guard now applies once to the roster.
    """
    rows = [
        ("omninode-runtime", 0, 0, ""),
        ("omninode-runtime-effects", 0, 0, ""),
        ("onex-lab-redpanda", 1, 1, ""),
    ]
    monkeypatch.setattr(boot_gate, "_deployment_rows", lambda namespace: rows)
    monkeypatch.setattr(boot_gate, "_topic_exists", lambda *a, **k: True)
    assert (
        boot_gate.wait_for_boot(
            namespace="onex-dev",
            timeout_seconds=60,
            poll_seconds=1,
            lane_prefix="onex-lab-",
            broker_deployment="onex-lab-redpanda",
            require_topic=True,
        )
        == 1
    )
    assert "zero replicas" in "".join(capsys.readouterr())


def test_a_slow_lane_stand_in_is_not_reported_as_a_candidate_defect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The broker/Postgres/Valkey stand-ins are lane scaffolding, not the candidate.

    They are still waited on -- a lane whose broker never came up proves nothing --
    but they are separated by name prefix so the failure report says which side is
    at fault.
    """
    rows = [("omninode-runtime", 1, 1, ""), ("onex-lab-redpanda", 0, 1, "")]
    monkeypatch.setattr(boot_gate, "_deployment_rows", lambda namespace: rows)
    monkeypatch.setattr(boot_gate, "_topic_exists", lambda *a, **k: True)
    assert (
        boot_gate.wait_for_boot(
            namespace="onex-dev",
            timeout_seconds=0,
            poll_seconds=1,
            lane_prefix="onex-lab-",
            broker_deployment="onex-lab-redpanda",
            require_topic=True,
        )
        == 1
    )
