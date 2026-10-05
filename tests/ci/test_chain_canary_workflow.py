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

* **runs-on** — the overlay's runner pool (``vars.LAB_PROBE_RUNS_ON_JSON``),
  never a host label (OMN-19894, operator rulings 2026-09-28T01:58:06Z and
  01:58:17Z). Until then the job was pinned to the .201 verify runner and
  reached the lane through that host's Docker gateway alias; the lane it
  grades is now the first answering lane of ``vars.LAB_LANES_JSON``.
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

import importlib.util
import tempfile
import textwrap
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
def test_runs_on_the_overlay_pool_not_one_host(
    canary_job: dict[str, object],
) -> None:
    """OMN-19894: any runner of the overlay's pool, never a host label."""
    assert canary_job["runs-on"] == "${{ fromJSON(vars.LAB_PROBE_RUNS_ON_JSON) }}"


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
def test_probes_the_lane_the_overlay_resolved(workflow_text: str) -> None:
    """OMN-19894: the addresses are the resolved lane's own declarations."""
    assert "./.github/actions/resolve-lab-lane" in workflow_text
    assert "require: ingress_url gateway_url" in workflow_text
    assert "${PROBE_URL_OVERRIDE:-${LANE_INGRESS_URL:-}}" in workflow_text
    assert "${GATEWAY_URL_OVERRIDE:-${LANE_GATEWAY_URL:-}}" in workflow_text
    # A localhost probe from inside the runner container hits the container
    # itself and reports a false RED (OMN-14958); a gateway alias reaches a
    # different machine from every runner (OMN-19894).
    for pin in ("http://localhost:", "host.docker.internal"):
        assert pin not in workflow_text


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


@pytest.mark.unit
def test_the_projection_dsn_is_declared_by_name_and_injected_as_env(
    canary_job: dict[str, object], workflow_text: str
) -> None:
    """OMN-18060: the link-2 DSN reaches the node through the environment.

    Run 34281968883 failed closed on ``projection_readback_not_configured``
    because no DSN was declared anywhere. The fix declares its NAME in the same
    lane overlay the broker comes from and injects the VALUE as a job secret.
    Both halves are asserted, because either one alone is the bug: a name with
    no secret makes a readback that silently never runs, and a secret with no
    name makes one nobody can review.
    """
    resolve = _step(canary_job, "Resolve the declared projection-readback DSN")
    body = str(resolve["run"])
    assert "load_lane_projection_readback" in body
    assert "projection_readback_env" in body
    assert "ci_bus_lanes.yaml" in body

    probe = _step(canary_job, "Fire one live delegation")
    env = probe["env"]
    assert isinstance(env, dict)
    assert "CHAIN_CANARY_PROJECTION_DSN" in env, (
        "the DSN must be injected as a job secret under the literal name the "
        "lane declares"
    )
    assert str(env["CHAIN_CANARY_PROJECTION_DSN"]).startswith("${{ secrets."), (
        "the DSN must come from a secret, never from a literal in the workflow"
    )


@pytest.mark.unit
def test_no_dsn_ever_reaches_a_command_line(
    canary_job: dict[str, object], workflow_text: str
) -> None:
    """argv is world-readable through /proc and this step echoes its config.

    The flag the workflow passes carries the variable's NAME. There is
    deliberately no ``--projection-dsn`` flag anywhere -- not in the workflow,
    and not in the skill mapping, which is what builds argv.
    """
    probe = _step(canary_job, "Fire one live delegation")
    body = str(probe["run"])

    assert "--projection-dsn-env" in body
    assert "--projection-dsn " not in body
    assert "--projection-dsn=" not in body
    assert "CHAIN_CANARY_PROJECTION_DSN}" not in body, (
        "the run block must interpolate the NAME variable "
        "(CHAIN_CANARY_PROJECTION_DSN_ENV), never the DSN one"
    )

    registry = yaml.safe_load(_SKILL_MAPPING.read_text(encoding="utf-8"))
    mapping = next(
        (s for s in registry["skills"] if s["skill_name"] == _SKILL_NAME), None
    )
    assert mapping is not None
    declared = {f"--{arg['name']}" for arg in mapping["args"]}
    assert "--projection-dsn-env" in declared
    assert "--projection-dsn" not in declared, (
        "a --projection-dsn flag would put the credential in argv, in this "
        "run's log, and -- because `onex skill` serialises every arg into the "
        "node payload -- durably into the event log"
    )

    # And no connection string is spelled anywhere in the workflow itself.
    for marker in ("postgres://", "postgresql://", "password="):
        assert marker not in workflow_text, (
            f"the workflow spells {marker!r}; the DSN is a secret and belongs "
            "in the lab store under the declared name"
        )


@pytest.mark.unit
def test_the_deploy_agent_is_the_resolved_lanes_own(
    workflow_text: str,
) -> None:
    """OMN-19811 -- the canary reads the deploy agent of the lane it grades.

    Run 36202173467 went RED because a deploy recreated the runtime inside the
    probe's budget. The probe now reads the lane's deploy agent, and the URL
    comes from the lane's overlay entry (OMN-19894), never a literal.
    """
    registry = yaml.safe_load(_SKILL_MAPPING.read_text(encoding="utf-8"))
    mapping = next(s for s in registry["skills"] if s["skill_name"] == _SKILL_NAME)
    declared = {f"--{arg['name']}" for arg in mapping["args"]}
    for flag in ("--deploy-agent-url", "--deploy-wait-seconds"):
        assert flag in workflow_text, f"workflow no longer passes {flag}"
        assert flag in declared, f"skill mapping no longer declares {flag}"

    # OMN-19894: the agent is the resolved lane's own declaration, so a run on
    # any runner reads the agent of the lane it grades.
    assert "DEPLOY_AGENT_URL=${LANE_DEPLOY_AGENT_URL}" in workflow_text
    assert ":8098" not in workflow_text


# ---------------------------------------------------------------------------
# OMN-20594 -- a red canary reaches the operator, once per change of verdict
# ---------------------------------------------------------------------------
#
# OMN-16773 deferred alerting: until this, a red chain canary was a failed
# workflow run that nobody watched. The h201 dev lane is the protected
# delegation lane (the lane manifest says so), so a scheduled red posts to the
# Slack channel the fleet canary and dev-lane-liveness already alert to,
# through the same two secrets, and a green after a red posts the recovery.
# The decision is edge-triggered against the previous completed scheduled run,
# so a lane that stays red posts once, not once a run.

_ALERT_STEP = "Tell the operator when the delegation verdict changes (OMN-20594)"


def _alert_step(canary_job: dict[str, object]) -> dict[str, object]:
    steps = canary_job["steps"]
    assert isinstance(steps, list)
    matches = [s for s in steps if isinstance(s, dict) and s.get("name") == _ALERT_STEP]
    assert len(matches) == 1, f"exactly one {_ALERT_STEP!r} step"
    return matches[0]


def _alert_module(canary_job: dict[str, object]) -> dict[str, object]:
    """Execute the step's own inline program as a module, without running main."""
    run = str(_alert_step(canary_job)["run"])
    start = run.index("<<'PY'\n") + len("<<'PY'\n")
    end = run.index("\nPY", start)
    source = Path(tempfile.mkdtemp()) / "chain_canary_alert.py"
    source.write_text(textwrap.dedent(run[start:end]), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("chain_canary_alert", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return vars(module)


@pytest.mark.unit
def test_alert_step_runs_last_on_every_scheduled_outcome(
    canary_job: dict[str, object],
) -> None:
    step = _alert_step(canary_job)
    steps = canary_job["steps"]
    assert isinstance(steps, list)
    assert steps[-1] is step, "the alert reads job.status, so it runs after the verdict"
    assert step["if"] == "always() && github.event_name == 'schedule'"


@pytest.mark.unit
def test_alert_token_reaches_the_step_as_env_never_argv(
    canary_job: dict[str, object],
) -> None:
    step = _alert_step(canary_job)
    env = step["env"]
    assert isinstance(env, dict)
    assert env["SLACK_BOT_TOKEN"] == "${{ secrets.SLACK_BOT_TOKEN }}"
    assert env["SLACK_CHANNEL_ID"] == "${{ secrets.SLACK_CHANNEL_ID }}"
    assert env["JOB_STATUS"] == "${{ job.status }}"
    assert "secrets." not in str(step["run"]), (
        "a secret is never interpolated into the script"
    )


@pytest.mark.unit
def test_alert_job_may_read_its_own_run_history(canary_job: dict[str, object]) -> None:
    permissions = canary_job.get("permissions")
    assert isinstance(permissions, dict)
    assert permissions.get("actions") == "read"
    assert permissions.get("contents") == "read"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("job_status", "previous", "expected"),
    [
        ("failure", "success", "RED"),
        ("failure", None, "RED"),
        ("failure", "failure", None),
        ("failure", "timed_out", None),
        ("success", "failure", "RECOVERED"),
        ("success", "success", None),
        ("success", None, None),
        ("cancelled", "success", None),
    ],
)
def test_alert_posts_only_on_a_change_of_verdict(
    canary_job: dict[str, object],
    job_status: str,
    previous: str | None,
    expected: str | None,
) -> None:
    decide = _alert_module(canary_job)["decide"]
    assert callable(decide)
    assert decide(job_status, previous) == expected


@pytest.mark.unit
def test_alert_previous_verdict_skips_this_run_and_cancelled_runs(
    canary_job: dict[str, object],
) -> None:
    previous_conclusion = _alert_module(canary_job)["previous_conclusion"]
    assert callable(previous_conclusion)
    runs = [
        {"id": 9, "conclusion": None},
        {"id": 8, "conclusion": "cancelled"},
        {"id": 7, "conclusion": "skipped"},
        {"id": 6, "conclusion": "failure"},
        {"id": 5, "conclusion": "success"},
    ]
    assert previous_conclusion(runs, 9) == "failure"
    assert previous_conclusion(runs[:3], 9) is None


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


@pytest.mark.unit
def test_alert_red_posts_one_message_naming_the_verdict_and_run(
    canary_job: dict[str, object], tmp_path: Path
) -> None:
    import json as _json

    module = _alert_module(canary_job)
    main = module["main"]
    assert callable(main)
    receipt = tmp_path / "chain-canary-receipt.json"
    receipt.write_text(
        _json.dumps(
            {"result": {"verdict": "terminal_missing", "detail": "no terminal"}}
        ),
        encoding="utf-8",
    )
    sent: list[object] = []

    def opener(request: object, timeout: float = 0) -> _FakeResponse:
        url = getattr(request, "full_url", "")
        if "api.github.com" in url:
            runs = {"workflow_runs": [{"id": 1, "conclusion": "success"}]}
            return _FakeResponse(_json.dumps(runs).encode())
        sent.append(request)
        return _FakeResponse(b'{"ok": true, "ts": "1.2"}')

    env = {
        "JOB_STATUS": "failure",
        "RUN_ID": "2",
        "RUN_URL": "https://github.com/o/r/actions/runs/2",
        "GITHUB_REPOSITORY": "o/r",
        "GITHUB_API_URL": "https://api.github.com",
        "GITHUB_TOKEN": "gh-token",
        "BRANCH": "dev",
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_CHANNEL_ID": "C123",
        "RECEIPT_PATH": str(receipt),
    }
    assert main(env, opener) == 0
    assert len(sent) == 1
    request = sent[0]
    body = _json.loads(request.data)
    assert body["channel"] == "C123"
    assert "terminal_missing" in body["text"]
    assert "https://github.com/o/r/actions/runs/2" in body["text"]
    assert request.get_header("Authorization") == "Bearer xoxb-test"


@pytest.mark.unit
def test_alert_without_slack_secrets_warns_and_never_passes_silently(
    canary_job: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    import json as _json

    main = _alert_module(canary_job)["main"]
    assert callable(main)

    def opener(request: object, timeout: float = 0) -> _FakeResponse:
        return _FakeResponse(_json.dumps({"workflow_runs": []}).encode())

    env = {
        "JOB_STATUS": "failure",
        "RUN_ID": "2",
        "RUN_URL": "u",
        "GITHUB_REPOSITORY": "o/r",
        "GITHUB_TOKEN": "t",
        "BRANCH": "dev",
    }
    assert main(env, opener) == 0
    assert "::warning::" in capsys.readouterr().out
