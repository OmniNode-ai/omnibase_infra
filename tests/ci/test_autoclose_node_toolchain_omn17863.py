# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17863: the sweep must carry a Node toolchain, not only a Python one.

The evidence-autoclose sweep dispatches ``dod_verify``, which executes a
contract's behaviour checks verbatim in their declared ``cwd`` (``test_passes``
is a semantic alias for ``command``, OMN-16824). Every provisioning step in this
job is Python-shaped — ``setup-python``, ``setup-uv``, a gate venv, a dispatch
venv — because until now every behaviour check was.

The behaviour proof drafted for OMN-17863 is not:

.. code-block:: yaml

    - id: "dod-omn17863-audit-verdict-behavior-proof"
      source: "manual"
      checks:
        - check_type: "test_passes"
          check_value: "pnpm test:audit-verdict"
          cwd: "${OMNI_HOME}/omniweb"

On a job with no Node and no corepack that command exits 127 and is recorded
``failed=1`` — the verifier's own missing runtime asserted as a product defect,
and indistinguishable in the receipt from the product genuinely failing. That is
the same conflation OMN-17863 itself is about one layer up, where an unreachable
advisory registry was indistinguishable from an advisory. It is also strictly
more blocking than the missing proof it replaces, so the check was held unpushed
rather than landed into a job that could not run it.

This module pins the provisioning closed. Removing it must fail here rather than
silently returning the job to recording a toolchain gap as a product failure.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "evidence-autoclose-sweep.yml"

JOB = "evidence-autoclose-sweep"

# The two steps that dispatch dod_verify, and therefore the two steps whose
# environment decides whether a JS behaviour check can execute at all.
_DOD_VERIFY_STEPS = ("Run evidence autoclose sweep", "Diagnose verdict divergence")

_SHA_PINNED = re.compile(r"^actions/setup-node@[0-9a-f]{40}\s*(#.*)?$")

# ``_DEFAULT_CHECK_TIMEOUT_S`` in omnimarket's evidence_collector. Restated
# here because this repo does not import that module; it is the ceiling a step
# that sets no override actually runs under.
_DEFAULT_CHECK_CEILING_S = 30

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def steps(workflow: dict) -> list[dict]:
    return workflow["jobs"][JOB]["steps"]


def _named(steps: list[dict], fragment: str) -> dict:
    for step in steps:
        if fragment in (step.get("name") or ""):
            return step
    raise AssertionError(f"no step whose name contains {fragment!r}")


def _index(steps: list[dict], fragment: str) -> int:
    for i, step in enumerate(steps):
        if fragment in (step.get("name") or ""):
            return i
    raise AssertionError(f"no step whose name contains {fragment!r}")


def test_the_job_provisions_node(steps: list[dict]) -> None:
    """AC1. The runner class this job uses has no ambient Node.

    Every other workflow in this repo is Python-only and omniweb's own CI
    obtains pnpm through ``pnpm/action-setup`` + ``setup-node`` on every run, so
    nothing establishes that a Node runtime is present on the trusted fleet.
    The job provisions its own rather than assuming one.
    """
    step = _named(steps, "Provision Node")
    uses = (step.get("uses") or "").strip()
    assert uses.startswith("actions/setup-node@"), uses


def test_the_node_action_is_pinned_by_sha(steps: list[dict]) -> None:
    """AC1. A moving tag is a supply-chain hole and an unreproducible job.

    ``actions/checkout`` in this same job is already SHA-pinned; a new
    third-party action must meet the same bar rather than the looser one the
    older ``setup-python@v7`` / ``setup-uv@v7`` lines were admitted under.
    """
    step = _named(steps, "Provision Node")
    uses = (step.get("uses") or "").strip()
    assert _SHA_PINNED.match(uses), (
        f"{uses!r} must be actions/setup-node pinned to a 40-hex commit sha, "
        "with the human-readable version in a trailing comment"
    )


def test_the_node_version_comes_from_the_projects_own_declaration(
    steps: list[dict],
) -> None:
    """AC1. The version is the product's statement, not this file's guess.

    A version hardcoded here drifts from what the JS projects under
    adjudication actually require, and the drift is invisible until a check
    fails for a syntax reason that reads as a product defect. It is derived from
    the materialised repos' own ``.nvmrc`` / ``engines.node`` instead.
    """
    derive = _named(steps, "Derive the Node version")
    assert derive.get("id"), "the derivation step must publish an output"
    provision = _named(steps, "Provision Node")
    version = str((provision.get("with") or {}).get("node-version", ""))
    assert f"steps.{derive['id']}.outputs" in version, version


def test_corepack_is_enabled_in_its_own_step(steps: list[dict]) -> None:
    """AC1. ``setup-node`` does not enable corepack, and pnpm is not bundled.

    corepack is what makes a project's ``packageManager`` pin the thing that
    runs. Without it the job has Node and still no pnpm, which is the same exit
    127 by a different route.
    """
    step = _named(steps, "Enable corepack")
    body = step.get("run") or ""
    assert "corepack enable" in body, body
    # Provisioning that reports success without being usable is the failure
    # mode this repo keeps re-learning (OMN-17307): assert, do not assume.
    assert "corepack --version" in body or "node --version" in body, body


def test_the_toolchain_is_provisioned_before_every_step_that_dispatches_dod_verify(
    steps: list[dict],
) -> None:
    """AC2. Ordering is the whole mechanism.

    The diagnose step exists to reproduce the sweep's environment exactly; a
    toolchain present for one and not the other would make the diagnostic lie
    about the thing it exists to diagnose (the same argument the OMN-16902 D2
    comment already makes about ``UV_NO_SYNC``).
    """
    corepack = _index(steps, "Enable corepack")
    provision = _index(steps, "Provision Node")
    derive = _index(steps, "Derive the Node version")
    assert derive < provision < corepack

    for fragment in _DOD_VERIFY_STEPS:
        assert corepack < _index(steps, fragment), (
            f"{fragment!r} dispatches dod_verify before the Node toolchain "
            "exists, so a JS behaviour check would still exit 127"
        )


def test_both_dod_verify_steps_route_the_staged_toolchain_off_the_workspace(
    steps: list[dict],
) -> None:
    """AC2. The staged trees must not be built inside a checkout.

    ``node_modules`` for a real project is hundreds of megabytes. Left in
    ``GITHUB_WORKSPACE`` it would be picked up by the gate-venv purity
    assertions' own tree walks and by any later checkout, so it is placed under
    ``RUNNER_TEMP`` where the dispatch venv already lives.
    """
    for fragment in _DOD_VERIFY_STEPS:
        env = _named(steps, fragment).get("env") or {}
        root = str(env.get("DOD_VERIFY_HERMETIC_NODE_ROOT", ""))
        assert root, (
            f"{fragment!r} does not place the staged JS trees, so they would "
            "land in the runner's HOME cache and outlive the job"
        )
        assert "runner.temp" in root or "RUNNER_TEMP" in root, root


def test_the_install_budget_is_separated_from_the_per_check_ceiling(
    steps: list[dict],
) -> None:
    """AC2. A cold pnpm store is a download, not a slow product.

    ``DOD_VERIFY_CHECK_TIMEOUT_S`` is 180 here. Charging a first install to it
    would kill the check mid-flight and record CHECK_BUDGET_EXCEEDED — a
    verifier-side ceiling reported as a fact about the ticket, the confusion
    OMN-17795 had to unpick. The build has its own variable, exactly as the uv
    sync does.
    """
    for fragment in _DOD_VERIFY_STEPS:
        env = _named(steps, fragment).get("env") or {}
        budget = env.get("DOD_VERIFY_HERMETIC_SYNC_TIMEOUT_S")
        assert budget is not None, (
            f"{fragment!r} leaves the hermetic build budget at its default "
            "while overriding the per-check ceiling; the two are separate on "
            "purpose"
        )
        # The sweep step overrides the per-check ceiling to 180; the diagnose
        # step leaves it at the collector's own 30s default. Compare against
        # whichever actually applies in that step, so this asserts the
        # separation rather than the presence of one particular override.
        ceiling = int(
            str(env.get("DOD_VERIFY_CHECK_TIMEOUT_S", _DEFAULT_CHECK_CEILING_S))
        )
        assert int(str(budget)) > ceiling, (
            "the build budget must exceed the per-check ceiling, or the "
            "separation buys nothing"
        )


def test_corepack_never_blocks_on_an_interactive_prompt(steps: list[dict]) -> None:
    """AC2. A scheduled run has nobody to answer a download prompt.

    corepack asks before fetching a pinned package manager it does not have.
    In a job triggered by cron that is an indefinite hang inside the 60-minute
    budget, reported as nothing at all.
    """
    for fragment in _DOD_VERIFY_STEPS:
        env = _named(steps, fragment).get("env") or {}
        assert str(env.get("COREPACK_ENABLE_DOWNLOAD_PROMPT")) == "0", (
            f"{fragment!r} may hang waiting for a corepack download prompt"
        )
