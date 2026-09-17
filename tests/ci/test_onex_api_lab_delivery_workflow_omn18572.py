# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shape invariants for the onex-api lab delivery (OMN-18572).

Each assertion here is a property that, if it silently regressed, would turn the
workflow into the false green it exists to replace. They are asserted on the
parsed workflow, never by matching prose: a comment that says a job runs on the
fleet and a `runs-on` that says otherwise are two different facts, and only one
of them decides where the job runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "onex-api-lab-delivery-reusable.yml"
GUARD = REPO_ROOT / "scripts" / "ci" / "check_lane_onex_api_revision.py"


def _load() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return list(job["steps"])


def _step(job: dict[str, Any], fragment: str) -> dict[str, Any]:
    matches = [s for s in _steps(job) if fragment in str(s.get("name", ""))]
    assert len(matches) == 1, f"expected one step matching {fragment!r}, got {matches}"
    return matches[0]


# --------------------------------------------------------------------------- #
# The interface                                                                #
# --------------------------------------------------------------------------- #


def test_it_is_callable_only() -> None:
    """A delivery is asked for by the repository that merged, never self-fired.

    A `push` trigger here would fire on omnibase_infra's own merges, which have
    their own trigger and their own receipt, and would publish a second command
    for every one of them.
    """
    triggers = _load()[True]
    assert set(triggers) == {"workflow_call"}


def test_the_caller_must_supply_every_fact_and_none_is_defaulted() -> None:
    """Rule 8: a defaulted sha or branch is a delivery nobody asked for."""
    inputs = _load()[True]["workflow_call"]["inputs"]
    for required in ("source_sha", "base_branch", "changed_files", "requested_by"):
        assert inputs[required]["required"] is True, required
        assert "default" not in inputs[required], required
    # The one optional input is the pin escape hatch, and it defaults to empty
    # so `inputs.infra_ref || github.job_workflow_sha` resolves to the pin.
    assert inputs["infra_ref"]["required"] is False
    assert inputs["infra_ref"]["default"] == ""


def test_the_sasl_pair_is_required_and_no_hmac_secret_is() -> None:
    """The lane declares SASL and the publisher refuses to downgrade it.

    The deploy-agent HMAC belongs to the runtime deploy EFFECT that emits
    rebuild-requested, not to this upstream start command; declaring it here
    would make a caller supply a secret this path never uses.
    """
    secrets = _load()[True]["workflow_call"]["secrets"]
    assert set(secrets) == {"KAFKA_SASL_USERNAME", "KAFKA_SASL_PASSWORD"}
    for name in secrets:
        assert secrets[name]["required"] is True, name


# --------------------------------------------------------------------------- #
# The publish job                                                              #
# --------------------------------------------------------------------------- #


def test_a_branch_other_than_dev_is_refused_before_anything_is_published() -> None:
    """The refusal is the FIRST step, so a wrong branch publishes nothing.

    Without it, `main` maps to the stability-test lane inside the publisher and
    this path would quietly request a rebuild of a governed lane.
    """
    job = _load()["jobs"]["trigger-delivery"]
    first = _steps(job)[0]
    assert "Refuse a branch" in first["name"]
    assert 'if [ "${BASE_BRANCH}" != "dev" ]' in first["run"]
    assert "exit 1" in first["run"]


def test_the_publisher_is_the_shared_one_and_names_omninode_infra() -> None:
    """One publisher. Two divergent copies is how this org got two already."""
    job = _load()["jobs"]["trigger-delivery"]
    run = _step(job, "Publish the redeploy-start")["run"]
    assert "scripts/trigger_rebuild_on_merge.py" in run
    assert '--source-repo "omninode_infra"' in run
    # A sibling-shaped publish REQUIRES a primary ref: the agent resets its
    # omnibase_infra build-context clone to the published git_ref, and an
    # omninode_infra sha names no commit there.
    assert "--primary-ref" in run


def test_the_publisher_stays_off_the_shared_trusted_seam() -> None:
    """LAN-bound: the dev control bus is a tailnet host:port (OMN-17888).

    The dedicated variable is deliberately unset so the self-hosted literal
    operates; reading the trusted seam here is what relocated the sibling
    publisher onto hosted compute that cannot resolve the broker name.
    """
    job = _load()["jobs"]["trigger-delivery"]
    assert "OMNI_RUNTIME_REBUILD_RUNS_ON_JSON" in job["runs-on"]
    assert "OMNI_TRUSTED_CI_RUNS_ON_JSON" not in job["runs-on"]


# --------------------------------------------------------------------------- #
# The convergence job                                                          #
# --------------------------------------------------------------------------- #


def test_the_verify_job_runs_where_the_lane_is() -> None:
    """The lane's docker daemon and published ports are on the LAN."""
    job = _load()["jobs"]["verify-onex-api-delivered"]
    assert job["runs-on"] == ["self-hosted", "omnibase-deploy"]


def test_the_verify_job_runs_only_for_a_published_dev_command() -> None:
    """A no-op publish has nothing to wait for, and no other lane is in scope."""
    job = _load()["jobs"]["verify-onex-api-delivered"]
    condition = " ".join(job["if"].split())
    assert "needs.trigger-delivery.outputs.published == 'true'" in condition
    assert "needs.trigger-delivery.outputs.runtime_lane == 'dev'" in condition


def test_the_guard_it_invokes_exists_and_is_the_onex_api_one() -> None:
    """Not the runtime guard and not the sibling guard: a third fact, third reader."""
    assert GUARD.exists()
    job = _load()["jobs"]["verify-onex-api-delivered"]
    run = _step(job, "Wait for the lane's onex-api")["run"]
    assert "scripts/ci/check_lane_onex_api_revision.py" in run
    assert "--expect-revision" in run
    assert "check_dev_lane_staleness.py" not in run
    assert "check_lane_sibling_revision.py" not in run


def test_the_guard_checks_out_this_repo_at_the_caller_s_pin() -> None:
    """OMN-18387: a caller pinned to older YAML invoking a newer script FAILs.

    The YAML spelling a script's arguments and the script itself must be one
    commit behind one pin, or a required argument added in one commit is missing
    from every call the other makes.
    """
    job = _load()["jobs"]["verify-onex-api-delivered"]
    checkout = _step(job, "Checkout omnibase_infra (the guard's own repository)")
    assert checkout["with"]["ref"] == (
        "${{ inputs.infra_ref || github.job_workflow_sha }}"
    )
    assert checkout["with"]["persist-credentials"] is False


def test_the_health_probe_reaches_the_host_gateway_not_localhost() -> None:
    """This job is a CONTAINER on the lane's host, not in its network namespace.

    Measured on the neighbouring workflows: from inside the deploy runner,
    `localhost:8085` answers 000 while the `host.docker.internal` form answers
    200. Every compose-dev receipt emitted before that was understood carried
    three identical connection-refused failures and none was ever a PASS.

    The URL is PARSED and its parts asserted, rather than scanned for a
    substring. A substring check for the loopback name is incomplete
    sanitization: it passes for a host that merely begins with that name and
    fails for a perfectly good path containing the word, so it decides on the
    wrong thing in both directions. The host is a field; assert the field.
    """
    job = _load()["jobs"]["verify-onex-api-delivered"]
    env = _step(job, "Probe onex-api on the lane")["env"]
    url = urlparse(env["ONEX_API_URL"])
    assert url.scheme == "http"
    assert url.hostname == "host.docker.internal"
    assert url.port == 8090
    assert url.path == "/health"


def test_the_receipt_records_both_the_revision_and_the_health_claims() -> None:
    """Two separate checks, because they fail separately.

    A revision that converged with a dead API, and a healthy API on the previous
    image, are different outcomes. One boolean would hide both.
    """
    job = _load()["jobs"]["verify-onex-api-delivered"]
    run = _step(job, "Emit the compose-dev lab-pass receipt")["run"]
    assert '--check "onex_api_revision:' in run
    assert '--check "onex_api_health:' in run
    assert "--checks-json" in run


def test_every_recording_step_runs_on_both_outcomes() -> None:
    """A receipt that exists only when everything worked cannot tell a failure
    apart from a run nobody made, and telling those apart is its whole job."""
    job = _load()["jobs"]["verify-onex-api-delivered"]
    for fragment in (
        "Probe onex-api on the lane",
        "Probe the dev lane",
        "Emit the compose-dev lab-pass receipt",
        "Assert the compose-dev lab-pass receipt is present",
        "Assert the compose-dev receipt on disk is this job's own",
        "Upload the compose-dev lab-pass receipt",
    ):
        step = _step(job, fragment)
        assert step.get("if") == "always()", fragment


def test_the_artifact_is_keyed_by_the_omninode_infra_sha() -> None:
    """Exact-sha keyed by construction: the name is the gate's lookup key.

    Keyed by the omninode_infra sha and not this repository's, because that is
    the revision the lane's onex-api was proven to carry. It collides with
    neither the omnibase_infra receipt nor a sibling's.
    """
    job = _load()["jobs"]["verify-onex-api-delivered"]
    upload = _step(job, "Upload the compose-dev lab-pass receipt")
    assert upload["with"]["name"] == (
        "lab-pass-receipt-compose-dev-${{ needs.trigger-delivery.outputs.source_sha }}"
    )
    assert upload["with"]["if-no-files-found"] == "error"


def test_the_receipt_identity_is_asserted_before_it_is_uploaded() -> None:
    """OMN-18420: presence is not identity on a shared self-hosted host."""
    job = _load()["jobs"]["verify-onex-api-delivered"]
    names = [str(s.get("name", "")) for s in _steps(job)]
    verify_at = next(
        i for i, n in enumerate(names) if "receipt on disk is this job's own" in n
    )
    upload_at = next(i for i, n in enumerate(names) if n.startswith("Upload the"))
    assert verify_at < upload_at


def test_the_receipt_directory_is_private_to_this_execution() -> None:
    """A fixed /tmp path on a shared runner is how a stale receipt got published."""
    job = _load()["jobs"]["verify-onex-api-delivered"]
    run = _step(job, "Record the lab-pass window start")["run"]
    assert (
        "${RUNNER_TEMP}/lab-pass-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-${GITHUB_JOB}"
        in run
    )
    assert "rm -rf" in run


def test_the_job_ceiling_agrees_with_the_budget_it_declares() -> None:
    """DERIVED, not chosen: converge 25m + declared settle + reserved tail."""
    job = _load()["jobs"]["verify-onex-api-delivered"]
    probe_env = _step(job, "Probe the dev lane")["env"]
    assert job["timeout-minutes"] == probe_env["JOB_TIMEOUT_MINUTES"]
