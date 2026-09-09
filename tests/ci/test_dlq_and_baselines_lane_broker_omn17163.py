# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PR-time guard for the two scheduled bus probes' runner + broker wiring.

WHAT THIS EXISTS TO STOP HAPPENING AGAIN
----------------------------------------
``dlq-depth-monitor.yml`` shipped on 2026-08-28 with ``runs-on`` reading the
trusted-CI routing variable and ``KAFKA_BOOTSTRAP_SERVERS`` defaulting to the
literal ``localhost:19092``. It then failed **545 consecutive runs** -- 543
``failure``, 2 ``cancelled``, 0 ``success`` -- with the same
``KafkaConnectionError: Unable to bootstrap from [('localhost', 19092,
AF_UNSPEC)]``. ``baselines-scheduler.yml`` carried the identical defect.

Neither workflow has a ``pull_request`` trigger, so nothing at PR time noticed:
they merged green because their own CI never runs them, and the alert surface
they provide is "a failing scheduled run", which had been unconditionally red
since the first tick. Red carried no information, and the sink the DLQ monitor
watches -- ``onex.dlq.omnibase-infra.quarantine.v1``, ~8.88M records, the one
that hid a total delegation outage (OMN-16767) -- was unmeasured for the
monitor's entire existence.

That is CLAUDE.md Rule 5 exactly: detection that is not enforced at merge time
is advisory, and an unenforced probe rots silently. These tests are the
enforcement.

WHAT EACH PIN IS FOR, AND THE INCIDENT BEHIND IT
------------------------------------------------
* **``runs-on`` pinned literally to ``[self-hosted, omnibase-deploy]``** --
  the constraint is NETWORK NAMESPACE, not trust. The lane's broker is a
  ``.201`` host-published port. GitHub-hosted compute cannot reach it, and
  neither can an ``omnibase-ci`` runner: those are containers with no host
  networking, where ``localhost`` is the container's own loopback (the old
  header comment named ``omnibase-ci`` as the fix and was wrong about it).
  ``omnibase-deploy`` is the ONE runner carrying
  ``extra_hosts: host.docker.internal:host-gateway``. Reading a routing
  variable here means an org-wide routing decision -- correct on its own terms
  and single-owner under OMN-16682 -- silently revokes this job's
  reachability.
* **no literal broker address anywhere** -- an address literal is wrong twice
  over: it names a host only some readers resolve, and it carries no
  TRANSPORT, so nothing tells the client the listener started requiring SASL
  (OMN-18012 Phase B, 2026-09-07). Both halves are declared together in
  omnimarket ``config/ci_bus_lanes.yaml``.
* **the lane-transport resolve step** -- ``load_lane_transport`` RAISES on a
  missing lane, an in-memory lane, an undeclared protocol, or a host-alias
  broker. It never returns a default, which is the fail-closed behaviour the
  DLQ monitor's header claimed and the workflow defeated by supplying one the
  node then trusted.
* **SASL credentials as step ENVIRONMENT** -- never argv. A credential on a
  command line is readable through ``/proc/<pid>/cmdline`` and lands in the
  run log, which these steps echo.
* **no ``~/.omnibase/.env`` sourcing** -- it is a no-op on a hosted runner AND
  on the deploy runner, whose operator env is mounted at
  ``/home/runner/.runner-creds/operator.env``. It looked like a fallback and
  supplied nothing.

Every zero-result assertion below is paired with a POSITIVE CONTROL that feeds
the same predicate an input known to trip it. An empty result from a broken
matcher reads exactly like a clean bill of health, which is the failure mode
this whole file is about.

Ticket: OMN-17163 (amends OMN-16769, OMN-3335)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"

_DLQ_WORKFLOW = _WORKFLOWS / "dlq-depth-monitor.yml"
_BASELINES_WORKFLOW = _WORKFLOWS / "baselines-scheduler.yml"
#: The reference implementation both of the above were brought into line with.
#: Used as a live positive control: if a predicate here is so strict that even
#: the known-good workflow fails it, the predicate is the defect.
_CANARY_WORKFLOW = _WORKFLOWS / "chain-canary.yml"

_PUBLISHER = _REPO_ROOT / "scripts" / "run_baselines_batch_compute.py"

_LANE_OVERLAY_PATH = "config/ci_bus_lanes.yaml"
_LANE_OVERLAY_REPO = "OmniNode-ai/omnimarket"
_DEPLOY_RUNNER = ["self-hosted", "omnibase-deploy"]

#: The exact address that took 545 runs red, plus the shape of any other
#: `host:port` literal on the lane's broker port. Matching the PORT rather than
#: only the full string is deliberate: `127.0.0.1:19092` and
#: `host.docker.internal:19092` are the same defect wearing a different host.
_HOST_LOCAL_BROKER = re.compile(r"\b(?:localhost|127\.0\.0\.1|\[::1\])\s*:\s*\d+")
_LANE_BROKER_PORT = re.compile(r":\s*19092\b")

#: A `KAFKA_BOOTSTRAP_SERVERS:` assignment in a workflow -- the line that
#: supplied the default the node was fail-closed against.
_BOOTSTRAP_ASSIGNMENT = re.compile(r"^\s*KAFKA_BOOTSTRAP_SERVERS\s*:", re.MULTILINE)

#: The routing variable this job must not read (AC8). Flipping it is OMN-16682,
#: single-owner, and would be the wrong fix regardless.
_ROUTING_VARIABLE = "OMNI_TRUSTED_CI_RUNS_ON_JSON"

_OMNIBASE_ENV_SOURCING = re.compile(r"source\s+~?/?\S*\.omnibase/\.env")

_PROBE_WORKFLOWS = (_DLQ_WORKFLOW, _BASELINES_WORKFLOW)


def _shell_expansions_of(name: str) -> re.Pattern[str]:
    """`$NAME` or `${NAME...}` — the two forms that put a value on a command line."""
    return re.compile(rf"\$\{{?{re.escape(name)}\b")


def _executable(text: str) -> str:
    """``text`` with whole-line comments removed.

    These workflows' headers describe the defect in detail, and describing it
    accurately means naming the retired literal. Scanning raw text would then
    fail the file for its own postmortem -- the CLAUDE.md rule 15 shape, where
    prose that merely MENTIONS a trigger fires the gate. It bit this very test
    on its first run: both workflows failed ``test_no_literal_broker_address``
    on comment lines quoting the line that was removed.

    Only whole-line comments are dropped, never a trailing one, so
    ``FOO: 'localhost:19092'  # why`` still has its code half scanned. A
    comment cannot connect to a broker, so nothing enforceable is lost, and
    the alternative -- deleting the explanation to satisfy the matcher --
    would trade the record of a 545-run outage for a regex's convenience.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _workflow(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(_text(path))
    assert isinstance(loaded, dict), f"{path.name} is not a YAML mapping"
    return loaded


def _sole_job(path: Path) -> dict[str, object]:
    jobs = _workflow(path)["jobs"]
    assert isinstance(jobs, dict) and len(jobs) == 1, (
        f"{path.name} is expected to declare exactly one job; a second job "
        "would need its own runner/broker pins added here"
    )
    job = next(iter(jobs.values()))
    assert isinstance(job, dict)
    return job


def _steps(path: Path) -> list[dict[str, object]]:
    steps = _sole_job(path)["steps"]
    assert isinstance(steps, list)
    return [step for step in steps if isinstance(step, dict)]


def _mapping(step: dict[str, object], key: str) -> dict[str, object]:
    """``step[key]`` when it is a mapping, else an empty one.

    A step with no ``with:``/``env:`` block, and a step whose block is a
    scalar, are both "does not match" rather than a crash — the difference
    matters because these predicates run over EVERY step in the file.
    """
    value = step.get(key)
    return value if isinstance(value, dict) else {}


# --------------------------------------------------------------------------
# Positive controls. These run FIRST by file order on purpose: a zero-match
# assertion below is only evidence if the matcher can match something.
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_positive_control_broker_matchers_fire_on_the_original_defect() -> None:
    """The exact lines that were in these workflows must trip the matchers."""
    original_dlq_line = (
        "          KAFKA_BOOTSTRAP_SERVERS: "
        "${{ secrets.KAFKA_BOOTSTRAP_SERVERS || 'localhost:19092' }}"
    )
    assert _HOST_LOCAL_BROKER.search(original_dlq_line)
    assert _LANE_BROKER_PORT.search(original_dlq_line)
    assert _BOOTSTRAP_ASSIGNMENT.search(original_dlq_line)

    original_baselines_source = "          source ~/.omnibase/.env 2>/dev/null || true"
    assert _OMNIBASE_ENV_SOURCING.search(original_baselines_source)

    original_runs_on = (
        "        || fromJSON(vars.OMNI_TRUSTED_CI_RUNS_ON_JSON "
        '|| \'["self-hosted","omnibase-ci"]\')'
    )
    assert _ROUTING_VARIABLE in original_runs_on

    leaked = 'uv run onex probe --sasl-password "$KAFKA_SASL_PASSWORD"'
    assert _shell_expansions_of("KAFKA_SASL_PASSWORD").search(leaked)
    assert _shell_expansions_of("KAFKA_SASL_PASSWORD").search(
        'echo "${KAFKA_SASL_PASSWORD}"'
    )

    # And the inverse: a clean line must NOT trip them, or every assertion
    # below would pass for the wrong reason.
    # And the comment stripper must drop ONLY comments: if it ate code, every
    # zero-match assertion below would pass vacuously.
    mixed = (
        "# KAFKA_BOOTSTRAP_SERVERS: 'localhost:19092'  <- the retired line\n"
        "          KAFKA_BOOTSTRAP_SERVERS: 'localhost:19092'  # still a defect\n"
    )
    stripped = _executable(mixed)
    assert stripped.count("localhost:19092") == 1
    assert _HOST_LOCAL_BROKER.search(stripped)
    assert not _HOST_LOCAL_BROKER.search(
        _executable("# KAFKA_BOOTSTRAP_SERVERS: 'localhost:19092'")
    )

    clean = 'echo "declared broker: ${KAFKA_BOOTSTRAP_SERVERS}"'
    assert not _HOST_LOCAL_BROKER.search(clean)
    assert not _BOOTSTRAP_ASSIGNMENT.search(clean)
    assert _ROUTING_VARIABLE not in clean
    assert not _shell_expansions_of("KAFKA_SASL_PASSWORD").search(clean)


@pytest.mark.unit
def test_positive_control_the_reference_workflow_satisfies_these_pins() -> None:
    """chain-canary.yml is the shape these two were brought into line with.

    If it fails any of the three structural pins, the pin is wrong rather than
    the workflow under test -- so this check is what keeps the rest of the file
    honest rather than merely strict.
    """
    canary_text = _executable(_text(_CANARY_WORKFLOW))
    assert _sole_job(_CANARY_WORKFLOW)["runs-on"] == _DEPLOY_RUNNER
    assert not _HOST_LOCAL_BROKER.search(canary_text)
    assert _ROUTING_VARIABLE not in canary_text
    assert _LANE_OVERLAY_PATH in canary_text


# --------------------------------------------------------------------------
# The pins.
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("workflow_path", _PROBE_WORKFLOWS, ids=lambda p: p.name)
def test_workflow_exists(workflow_path: Path) -> None:
    assert workflow_path.is_file(), f"missing workflow at {workflow_path}"


@pytest.mark.unit
@pytest.mark.parametrize("workflow_path", _PROBE_WORKFLOWS, ids=lambda p: p.name)
def test_runs_on_the_only_runner_in_the_lane_reachability_domain(
    workflow_path: Path,
) -> None:
    assert _sole_job(workflow_path)["runs-on"] == _DEPLOY_RUNNER, (
        f"{workflow_path.name} must pin the deploy runner literally: it is the "
        "one runner with the host-gateway alias, and that is a network fact no "
        "routing variable can express"
    )


@pytest.mark.unit
@pytest.mark.parametrize("workflow_path", _PROBE_WORKFLOWS, ids=lambda p: p.name)
def test_no_routing_variable_can_move_the_job_off_the_lane_host(
    workflow_path: Path,
) -> None:
    """OMN-17163 AC8 — the fix must not be silently reversible by a flip."""
    assert _ROUTING_VARIABLE not in _executable(_text(workflow_path))


@pytest.mark.unit
@pytest.mark.parametrize("workflow_path", _PROBE_WORKFLOWS, ids=lambda p: p.name)
def test_no_literal_broker_address(workflow_path: Path) -> None:
    """OMN-17163 AC1 — no host-local literal, and no `19092` literal at all."""
    text = _executable(_text(workflow_path))
    host_local = _HOST_LOCAL_BROKER.findall(text)
    assert not host_local, (
        f"{workflow_path.name} carries host-local broker literal(s) "
        f"{host_local}: the lane's broker is a declaration, not an address a "
        "workflow gets to guess"
    )
    on_port = _LANE_BROKER_PORT.findall(text)
    assert not on_port, (
        f"{workflow_path.name} names the lane broker port literally; resolve "
        f"the address from {_LANE_OVERLAY_REPO} {_LANE_OVERLAY_PATH} instead"
    )


@pytest.mark.unit
@pytest.mark.parametrize("workflow_path", _PROBE_WORKFLOWS, ids=lambda p: p.name)
def test_no_bootstrap_default_defeats_the_nodes_fail_closed_check(
    workflow_path: Path,
) -> None:
    """The node fails closed on an unset broker; the workflow must not supply one.

    The original defect was not that the node lacked a guard -- it had one.
    The workflow handed it a value, so the guard passed on a broker that could
    never work.
    """
    assignments = _BOOTSTRAP_ASSIGNMENT.findall(_executable(_text(workflow_path)))
    assert not assignments, (
        f"{workflow_path.name} assigns KAFKA_BOOTSTRAP_SERVERS in the workflow; "
        "it must arrive from the lane-transport resolve step through $GITHUB_ENV"
    )


@pytest.mark.unit
@pytest.mark.parametrize("workflow_path", _PROBE_WORKFLOWS, ids=lambda p: p.name)
def test_no_omnibase_env_sourcing(workflow_path: Path) -> None:
    """It supplied nothing on either runner class and read as a fallback."""
    assert not _OMNIBASE_ENV_SOURCING.search(_executable(_text(workflow_path)))


@pytest.mark.unit
@pytest.mark.parametrize("workflow_path", _PROBE_WORKFLOWS, ids=lambda p: p.name)
def test_checks_out_the_lane_declaration_from_omnimarket(
    workflow_path: Path,
) -> None:
    overlay_steps = [
        step
        for step in _steps(workflow_path)
        if str(step.get("uses", "")).startswith("actions/checkout@")
        and _mapping(step, "with").get("repository") == _LANE_OVERLAY_REPO
    ]
    assert len(overlay_steps) == 1, (
        f"{workflow_path.name} must sparse-check-out the lane declaration "
        f"from {_LANE_OVERLAY_REPO} exactly once"
    )
    with_block = _mapping(overlay_steps[0], "with")
    assert _LANE_OVERLAY_PATH in str(with_block.get("sparse-checkout", ""))
    assert with_block.get("persist-credentials") is False, (
        "no git operations follow this checkout — do not leave the Actions "
        "token in local git config (zizmor artipacked)"
    )


@pytest.mark.unit
@pytest.mark.parametrize("workflow_path", _PROBE_WORKFLOWS, ids=lambda p: p.name)
def test_resolves_the_transport_through_the_fail_closed_loader(
    workflow_path: Path,
) -> None:
    """`load_lane_transport` raises rather than returning a default.

    Asserting on the loader NAME rather than on the exported variables is the
    point: any hand-rolled YAML read could export the same three variables and
    silently fall back to a default on a malformed lane.
    """
    resolve_steps = [
        step
        for step in _steps(workflow_path)
        if "load_lane_transport" in str(step.get("run", ""))
    ]
    assert len(resolve_steps) == 1, (
        f"{workflow_path.name} must resolve its broker through "
        "load_lane_transport exactly once"
    )
    run = str(resolve_steps[0]["run"])
    assert "lane_transport_env" in run
    assert "GITHUB_ENV" in run, "the resolved transport must reach later steps"
    assert _LANE_OVERLAY_PATH in run


@pytest.mark.unit
@pytest.mark.parametrize("workflow_path", _PROBE_WORKFLOWS, ids=lambda p: p.name)
def test_sasl_credentials_arrive_as_environment_never_argv(
    workflow_path: Path,
) -> None:
    """OMN-18012 Phase B made the dev listener SASL_PLAINTEXT/SCRAM-SHA-256."""
    credential_steps = [
        step
        for step in _steps(workflow_path)
        if {"KAFKA_SASL_USERNAME", "KAFKA_SASL_PASSWORD"}
        <= _mapping(step, "env").keys()
    ]
    assert len(credential_steps) == 1, (
        f"{workflow_path.name} must inject the dev-lane SCRAM principal as "
        "step environment on exactly the step that speaks to the broker"
    )
    env_block = _mapping(credential_steps[0], "env")
    for name in ("KAFKA_SASL_USERNAME", "KAFKA_SASL_PASSWORD"):
        expression = str(env_block[name])
        assert f"secrets.{name}" in expression, (
            f"{name} must come from the org secret of the same name, by NAME"
        )

    # And the credential is never DEREFERENCED in a shell body: argv is
    # world-readable through /proc/<pid>/cmdline, and these steps echo what
    # they ran into the run log. Naming the variable in the `env:` block is the
    # whole delivery mechanism; expanding it in `run:` is how it escapes.
    for step in _steps(workflow_path):
        run = str(step.get("run", ""))
        for name in ("KAFKA_SASL_USERNAME", "KAFKA_SASL_PASSWORD"):
            assert not _shell_expansions_of(name).search(run), (
                f"{workflow_path.name} expands {name} inside a run block; the "
                "client reads it from the environment and it must never reach "
                "argv or the log"
            )


@pytest.mark.unit
def test_publisher_script_has_no_broker_default() -> None:
    """The workflow was not the only place the literal lived.

    `scripts/run_baselines_batch_compute.py` carried its own
    `DEFAULT_BOOTSTRAP_SERVERS = "localhost:19092"`, so fixing only the
    workflow would have left the same guess one layer down -- and argparse
    would have re-supplied it.
    """
    source = _text(_PUBLISHER)
    assert "DEFAULT_BOOTSTRAP_SERVERS" not in source
    # The docstring names the retired literal to explain why it is gone; the
    # matcher must not read that prose as the defect returning, so only
    # executable lines are scanned.
    code_lines = [
        line
        for line in source.splitlines()
        if line.strip().startswith(("DEFAULT_", "default=", '"bootstrap.servers"'))
    ]
    for line in code_lines:
        assert not _HOST_LOCAL_BROKER.search(line), (
            f"publisher still defaults a broker address: {line.strip()!r}"
        )
    assert "build_confluent_auth_config_from_env" in source, (
        "the publisher uses confluent-kafka, which needs the confluent-side "
        "projection of the same lane transport the aiokafka clients read"
    )
