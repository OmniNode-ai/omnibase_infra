# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PR-time guard for the OMN-16773 event-chain canary wiring.

The canary itself only runs on a schedule against a live lane, so it has no
`pull_request` trigger and nothing at PR time would notice if its wiring
rotted — the exact "detection that is never enforced" shape CLAUDE.md Rule 5
warns about. These tests are that enforcement: they assert the workflow, the
skill mapping, and the node entry point stay in lock-step, so the canary
cannot be silently unwired by an unrelated edit.

Pinned deliberately, each for a reason a past incident supplies:

* **runs-on** — `omnibase-deploy` is the ONE runner with the host-gateway
  alias (`docker/docker-compose.runners.yml` `extra_hosts`). Move this job
  to any other runner and `host.docker.internal` stops resolving, every run
  reports `ingress_unreachable`, and a permanently-red check is a disabled
  check.
* **the probe host** — a `localhost` probe from inside a runner container
  hits the container itself (OMN-14958), manufacturing a false RED.
* **the skill invocation** — the workflow is a thin shim over
  `onex skill chain_canary`; if the mapping row and the workflow's flags
  drift apart, the dispatch fails on an unknown option rather than
  reporting a chain verdict.
* **no pull_request trigger** — this job publishes a REAL delegation
  command onto the dev lane. That is a lane mutation and does not belong on
  every PR.

Ticket: OMN-16773
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "chain-canary.yml"
_SKILL_MAPPING = _REPO_ROOT / "src" / "omnibase_infra" / "cli" / "skill_mapping.yaml"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

_NODE_NAME = "node_chain_canary_effect"
_SKILL_NAME = "chain_canary"


@pytest.fixture(scope="module")
def workflow() -> dict[str, object]:
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def workflow_text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def canary_job(workflow: dict[str, object]) -> dict[str, object]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["chain-canary"]
    assert isinstance(job, dict)
    return job


@pytest.mark.unit
def test_workflow_exists() -> None:
    assert _WORKFLOW.is_file(), f"missing canary workflow at {_WORKFLOW}"


@pytest.mark.unit
def test_runs_on_the_only_runner_that_can_reach_the_lane(
    canary_job: dict[str, object],
) -> None:
    assert canary_job["runs-on"] == ["self-hosted", "omnibase-deploy"]


@pytest.mark.unit
def test_scheduled_and_manually_dispatchable(workflow: dict[str, object]) -> None:
    # PyYAML parses a bare `on:` key as the boolean True.
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    assert "schedule" in triggers, "a canary nobody schedules is a recipe"
    assert "workflow_dispatch" in triggers, "must be runnable on demand"
    schedule = triggers["schedule"]
    assert isinstance(schedule, list) and schedule
    cron = schedule[0]["cron"]
    # Explicit hour list, not a `*/2` or Quartz-style `1/2` step: the latter
    # is not portable across cron parsers and would silently never fire.
    assert cron == "41 1,3,5,7,9,11,13,15,17,19,21,23 * * *"


@pytest.mark.unit
def test_no_pull_request_trigger(workflow: dict[str, object]) -> None:
    """The probe publishes a real command onto the lane — not a PR action."""
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    assert "pull_request" not in triggers
    assert "merge_group" not in triggers


@pytest.mark.unit
def test_probes_the_lane_through_the_host_gateway(workflow_text: str) -> None:
    assert "http://host.docker.internal:8085" in workflow_text
    # A localhost probe from inside the runner container hits the container
    # itself and reports a false RED (OMN-14958).
    assert "http://localhost:8085" not in workflow_text


def _dispatch_inputs(workflow: dict[str, object]) -> dict[str, object]:
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert isinstance(inputs, dict)
    return inputs


def _step(job: dict[str, object], name_fragment: str) -> dict[str, object]:
    for step in job["steps"]:
        if name_fragment.lower() in str(step.get("name", "")).lower():
            assert isinstance(step, dict)
            return step
    message = f"no step matching {name_fragment!r}"
    raise AssertionError(message)


@pytest.mark.unit
def test_no_broker_leg_carries_a_host_alias_default(
    workflow: dict[str, object], canary_job: dict[str, object]
) -> None:
    """OMN-17926 — asserted STRUCTURALLY, on values, never on file text.

    The predecessor of this test grepped the whole file for the literal, and
    that is exactly why it could not catch the fix landing wrong: a sentence
    in the header explaining the literal satisfies a text grep just as well as
    a live default does. Rule 15's failure mode, one level down. So this reads
    the dispatch input defaults and the probe step's env VALUES — the only
    places a broker address can actually reach the client.
    """
    from omnibase_infra.nodes.node_chain_canary_effect.lane_transport import (
        host_aliases_in,
    )

    inputs = _dispatch_inputs(workflow)
    for name in ("quarantine_bootstrap_servers", "terminal_bootstrap_servers"):
        default = str(inputs[name].get("default", ""))
        assert host_aliases_in(default) == (), (
            f"dispatch input {name} defaults to a Docker-Desktop host alias; "
            "that literal is what took 25 consecutive runs red"
        )
        assert default == "", (
            f"{name} must default to empty so the LANE DECLARATION supplies "
            "the broker; any literal default re-creates the defect"
        )

    probe_env = _step(canary_job, "Fire one live delegation")["env"]
    assert isinstance(probe_env, dict)
    for key, value in probe_env.items():
        assert host_aliases_in(str(value)) == () or "PROBE_URL" in key, (
            f"probe step env {key} carries a Docker-Desktop host alias"
        )


@pytest.mark.unit
def test_the_broker_comes_from_the_declared_lane_overlay(
    canary_job: dict[str, object],
) -> None:
    """The address and its transport are read from one declaration."""
    checkout = _step(canary_job, "Fetch the declared CI bus lanes")
    assert checkout["with"]["repository"] == "OmniNode-ai/omnimarket"
    assert "config/ci_bus_lanes.yaml" in str(checkout["with"]["sparse-checkout"])

    resolve = _step(canary_job, "Resolve the declared lane transport")
    body = str(resolve["run"])
    assert "load_lane_transport" in body
    assert "lane_transport_env" in body
    assert "ci_bus_lanes.yaml" in body


@pytest.mark.unit
def test_sasl_credentials_reach_the_client_as_env_never_argv(
    canary_job: dict[str, object],
) -> None:
    """A credential on a command line lands in the process list and the log."""
    probe = _step(canary_job, "Fire one live delegation")
    env = probe["env"]
    assert isinstance(env, dict)
    assert "KAFKA_SASL_USERNAME" in env
    assert "KAFKA_SASL_PASSWORD" in env

    body = str(probe["run"])
    for flag in ("--sasl-username", "--sasl-password", "--kafka-password"):
        assert flag not in body
    assert "KAFKA_SASL_PASSWORD" not in body, (
        "the credential must be read by the client from the environment, not "
        "interpolated into the command it runs"
    )

    resolve_body = str(_step(canary_job, "Resolve the declared lane transport")["run"])
    assert "KAFKA_SASL_USERNAME" not in resolve_body
    assert "KAFKA_SASL_PASSWORD" not in resolve_body


@pytest.mark.unit
def test_invokes_the_skill_with_flags_the_mapping_declares(
    workflow_text: str,
) -> None:
    assert f"onex skill {_SKILL_NAME}" in workflow_text

    registry = yaml.safe_load(_SKILL_MAPPING.read_text(encoding="utf-8"))
    mapping = next(
        (s for s in registry["skills"] if s["skill_name"] == _SKILL_NAME), None
    )
    assert mapping is not None, f"skill_mapping.yaml has no '{_SKILL_NAME}' row"
    assert mapping["node_name"] == _NODE_NAME

    declared = {f"--{arg['name']}" for arg in mapping["args"]}
    for flag in (
        "--probe-url",
        "--task-type",
        "--budget-ms",
        "--quarantine-bootstrap-servers",
        # OMN-16931: without this flag the run reports
        # TERMINAL_READBACK_NOT_CONFIGURED and the canary asserts nothing
        # about the terminal. Dropping it does not quietly restore the old
        # ingress-derived behaviour, but it does silently retire link 4.
        "--terminal-bootstrap-servers",
    ):
        assert flag in workflow_text, f"workflow no longer passes {flag}"
        assert flag in declared, f"skill mapping no longer declares {flag}"


@pytest.mark.unit
def test_terminal_readback_broker_is_wired_to_the_lane(workflow_text: str) -> None:
    """OMN-16931 — link 4 needs a reachable broker, from this runner.

    The readback consumes the lane's published broker port through the
    host-gateway alias, exactly like the quarantine leg. A localhost value
    here would resolve to the runner container and fail closed on every run,
    and a permanently-red check is a disabled check.
    """
    assert "TERMINAL_BOOTSTRAP:" in workflow_text
    assert "--terminal-bootstrap-servers" in workflow_text
    assert "terminal_bootstrap_servers:" in workflow_text, (
        "the workflow_dispatch input is how an operator retargets the "
        "readback; without it the broker is only settable by editing the file"
    )


@pytest.mark.unit
def test_summary_reports_per_link_verdicts_not_just_a_colour(
    workflow_text: str,
) -> None:
    """OMN-16931 — a 3-of-5 probe must never render as a 5-link proof.

    Run 33215999994 reported GREEN and was read as "the OMN-16025 gate is
    met". The receipt now carries a status per link and the summary prints
    the proven/total count; this test is what stops that rendering from
    being quietly simplified back to one word.
    """
    assert "link_verdicts" in workflow_text
    assert "links_proven" in workflow_text
    assert "chain_proof_complete" in workflow_text
    assert "links proven" in workflow_text
    # The green path must not print a bare "GREEN" that reads as a chain
    # proof — it says PROBE-GREEN and carries the link count with it.
    assert "chain canary PROBE-GREEN" in workflow_text
    assert 'print(f"chain canary GREEN' not in workflow_text


@pytest.mark.unit
def test_node_is_registered_as_an_entry_point() -> None:
    """Without the entry point, `onex skill` cannot resolve the contract."""
    pyproject = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    nodes = pyproject["project"]["entry-points"]["onex.nodes"]
    assert nodes[_NODE_NAME] == f"omnibase_infra.nodes.{_NODE_NAME}"


@pytest.mark.unit
def test_red_verdict_fails_the_run(workflow_text: str) -> None:
    """The whole point: a dead chain must produce a failing run.

    The dispatch step deliberately does NOT `set -e` (a failed dispatch must
    still leave a receipt), so the verdict step is the only thing standing
    between a red chain and a green check mark.
    """
    assert "sys.exit(1)" in workflow_text
    assert "::error::chain canary RED" in workflow_text
    assert "if: always()" in workflow_text
