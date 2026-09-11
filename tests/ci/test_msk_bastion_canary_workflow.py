# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PR-time guard on the MSK bastion canary workflow (OMN-15744).

The canary in .github/workflows/msk-bastion-canary.yml is the only thing that
watches the lab -> SNI bastion -> MSK routing path. A scheduled job is easy to
defang by accident: add `continue-on-error`, drop the schedule, point it at a
runner that cannot see the bastion, or quietly stop invoking the tests. Each of
those leaves a green badge over an unwatched path, which is the state that let
the original defect run for a month.

This test runs on every PR and makes each of those a red test instead of a
review catch. Mirrors tests/ci/test_chain_canary_workflow.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.ci, pytest.mark.unit]

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "msk-bastion-canary.yml"
)
TEST_DIR = Path(__file__).resolve().parents[1] / "integration" / "bastion"
TEST_MODULE = TEST_DIR / "test_msk_sni_bastion_chain.py"


def _workflow() -> dict[str, Any]:
    assert WORKFLOW.is_file(), f"canary workflow is missing: {WORKFLOW}"
    loaded = yaml.safe_load(WORKFLOW.read_text())
    assert isinstance(loaded, dict)
    return loaded


def _job() -> dict[str, Any]:
    jobs = _workflow()["jobs"]
    assert len(jobs) == 1, f"expected exactly one job, found {sorted(jobs)}"
    job = next(iter(jobs.values()))
    assert isinstance(job, dict)
    return job


def test_canary_is_scheduled() -> None:
    """A canary nobody runs is a recorded manual recipe, which is what failed."""
    triggers = _workflow()["on"]
    assert "schedule" in triggers, (
        "the bastion canary has no schedule trigger. Without it nothing watches "
        "the routing path between hand-run diagnoses."
    )
    assert triggers["schedule"], "schedule block is present but empty"


def test_canary_runs_where_the_bastion_is_reachable() -> None:
    """GitHub-hosted compute cannot see the bastion; only this runner can."""
    runs_on = _job()["runs-on"]
    assert "self-hosted" in runs_on and "omnibase-deploy" in runs_on, (
        f"runs-on is {runs_on!r}. The bastion is reachable only from the .201 "
        "tailnet, so any other runner can report only 'I cannot see it'."
    )


def test_canary_cannot_be_made_non_failing() -> None:
    """`continue-on-error` turns a canary into a status badge."""
    job = _job()
    assert not job.get("continue-on-error"), (
        "continue-on-error is set on the canary job; a canary that cannot fail "
        "watches nothing."
    )
    for step in job["steps"]:
        assert not step.get("continue-on-error"), (
            f"step {step.get('name')!r} sets continue-on-error, which hides the "
            "one signal this workflow exists to produce."
        )


def test_canary_actually_invokes_both_chains() -> None:
    """The workflow must run the golden chain AND the error chain."""
    run_steps = " ".join(str(step.get("run", "")) for step in _job()["steps"])
    assert "tests/integration/bastion" in run_steps, (
        "the canary does not invoke the bastion test package at all"
    )
    for selector in ("golden_every_broker", "error_chain", "covers_every_live_broker"):
        assert selector in run_steps, (
            f"the canary no longer selects {selector!r}. All three legs must run: "
            "the golden chain proves each broker answers as itself, the error "
            "chain proves an unmapped name fails closed, and the resize detector "
            "proves the map still covers the live cluster."
        )


def test_the_chain_tests_exist_and_carry_their_markers() -> None:
    """The workflow selects by -m integration; the module must carry it."""
    assert TEST_MODULE.is_file(), f"missing chain test module: {TEST_MODULE}"
    source = TEST_MODULE.read_text()
    assert "pytest.mark.integration" in source, (
        "the chain tests do not carry the integration marker, so the canary's "
        "`-m integration` selection would silently run nothing — a green job "
        "over zero tests."
    )
    for name in (
        "def test_bastion_golden_every_broker_is_reachable_and_distinct",
        "def test_bastion_error_chain_unmapped_sni_fails_closed",
        "def test_bastion_map_covers_every_live_broker",
    ):
        assert name in source, f"chain test {name!r} was removed or renamed"


def test_canary_does_not_produce_to_tenant_topics() -> None:
    """Reads only. A scheduled produce to a tenant topic is customer-visible."""
    source = TEST_MODULE.read_text()
    for forbidden in ("AIOKafkaProducer", "send_and_wait", "create_topics"):
        assert forbidden not in source, (
            f"{forbidden!r} appears in the bastion chain tests. This canary is "
            "read-only by design: it runs on a schedule against a live cloud "
            "cluster, so a produce or a topic creation here is a recurring "
            "side effect on tenant data."
        )


def test_nightly_integration_sweep_excludes_the_bastion_tests() -> None:
    """The nightly sweep must not collect tests it structurally cannot run.

    nightly-integration.yml runs `pytest tests/integration/ -m integration` on
    the `omnibase-ci` runner, which has neither AWS credentials nor tailnet
    reach to the bastion. Collecting these there produces a permanent red that
    everyone learns to ignore, which is worse than not running them: it trains
    people to skim past the one signal that matters.
    """
    nightly = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "nightly-integration.yml"
    )
    assert nightly.is_file(), f"missing {nightly}"
    text = nightly.read_text()
    assert "not bastion" in text, (
        "nightly-integration.yml no longer deselects the `bastion` marker, so "
        "the MSK bastion canary tests will be collected on a runner that cannot "
        "reach the bastion and will fail every night. Restore the "
        '`-m "integration and not bastion"` selector.'
    )
