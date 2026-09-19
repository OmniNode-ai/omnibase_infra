# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deps leg says what it is about to replace, and why (OMN-18640).

`omnibase_infra#3805` made the deps leg of a full rebuild CONVERGE instead of
force-recreate. That was the right change and it left one thing unsaid:
convergence still replaces a dependency, just only when compose's own config
hash says the declaration moved, and nothing anywhere announced that.

Measured on the .201 dev lane on 2026-09-19. Six consecutive `scope=full` jobs
ran the converging deps leg. Five left every dependency alone. One, command
`a3d6a355` staged against `6b3cfb0d`, replaced the broker, because that commit
added a healthcheck, two mounts and two environment keys to the `redpanda`
service in the dev-lane overlay. The only difference an operator could read
between that job and its five neighbours was a phase duration: 77 seconds
against 5 to 10. No log line and no field on the terminal event named the
service, the reason, or the fact that a replacement had happened at all.

So this is a REPORT, not a gate, and the distinction is load-bearing. Refusing
a dependency change on a warm deploy was considered and rejected on evidence:
nothing in the fleet emits a deps-only scope for a deliberate refresh to route
to (`handler_redeploy_orchestrator` hardcodes `full`, and
`EnumRedeployScope.CORE` has no emitter), so a refusal would mean the declared
change never applies at all. The first thing it would have stranded is the
broker readiness probe from OMN-18789, which exists precisely because the
healthcheck it replaces read healthy through a 97-minute outage.

What is pinned here:

1. The observation runs BEFORE the deps leg acts.
2. A dependency whose declaration moved is named, with both hashes, and the
   finding says it is about to be replaced.
3. A dependency whose declaration did not move is recorded as unchanged. That
   is a fact, not an absence.
4. The comparison NEVER gates: an unreadable hash, an absent container and a
   failed render each produce a finding carrying its reason, and the deps leg
   still runs.
5. Environment is compared by key, and no environment VALUE reaches the
   record. Those values are the lane's broker and database credentials.
6. Every compose argv the deploy issues is recorded, including the deps leg's,
   because until this existed the flags were readable only by sampling the
   host process table while the child ran.
7. Both reach the terminal event as typed fields.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any
from unittest.mock import patch

import pytest
from deploy_agent.events import (
    SCOPE_SERVICES,
    EnumRuntimeLane,
    ModelComposeInvocation,
    ModelDepsConvergenceFinding,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import DeployExecutor

pytestmark = [pytest.mark.unit, pytest.mark.deps_convergence]

CORE = tuple(SCOPE_SERVICES[Scope.CORE])
OLD_HASH = "c9b0b5beb8ab99c52887955c98756fa1c99af17499468d8ede83f38ade6604a3"
NEW_HASH = "1111111111111111111111111111111111111111111111111111111111111111"
SECRET = "a-broker-password-that-must-never-be-recorded"


def _noop(phase: Phase, status: PhaseStatus) -> None:
    return None


def _ok(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _inspect_json(config_hash: str, *, test: list[str], mounts: list[str]) -> str:
    """One `docker inspect` reply for a running core container."""
    return json.dumps(
        [
            {
                "Config": {
                    "Image": "redpandadata/redpanda:v24.2.7",
                    "Labels": {"com.docker.compose.config-hash": config_hash},
                    "Healthcheck": {"Test": test},
                    "Env": [f"DEV_KAFKA_SASL_PASSWORD={SECRET}"],
                },
                "Mounts": [{"Destination": m} for m in mounts],
            }
        ]
    )


class _Lane:
    """A fake lane: what compose renders, and what is running."""

    def __init__(
        self,
        *,
        rendered: dict[str, str],
        running: dict[str, str],
        render_json_rc: int = 0,
        hash_rc: int = 0,
    ) -> None:
        self.rendered = rendered
        self.running = running
        self.render_json_rc = render_json_rc
        self.hash_rc = hash_rc
        self.calls: list[list[str]] = []

    def run(
        self, cmd: list[str], timeout: int, **_: object
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        if "config" in cmd and "--hash" in cmd:
            if self.hash_rc:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=self.hash_rc, stdout="", stderr="boom"
                )
            body = "\n".join(f"{s} {h}" for s, h in self.rendered.items())
            return _ok(body)
        if "config" in cmd and "--format" in cmd:
            if self.render_json_rc:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=self.render_json_rc, stdout="", stderr="boom"
                )
            return _ok(
                json.dumps(
                    {
                        "services": {
                            "redpanda": {
                                "image": "redpandadata/redpanda:v24.2.7",
                                "healthcheck": {
                                    "test": [
                                        "CMD",
                                        "/usr/local/bin/onex-broker-readiness-probe",
                                    ]
                                },
                                "volumes": [
                                    {"target": "/var/lib/redpanda/data"},
                                    {
                                        "target": "/usr/local/bin/onex-broker-readiness-probe"
                                    },
                                ],
                                "environment": {
                                    "DEV_KAFKA_SASL_PASSWORD": SECRET,
                                    "DEV_KAFKA_SASL_USERNAME": "probe",
                                },
                            },
                            "postgres": {"image": "postgres:16"},
                            "valkey": {"image": "valkey:8"},
                        }
                    }
                )
            )
        if cmd[:2] == ["docker", "inspect"]:
            service = cmd[2].rsplit("-", 1)[-1]
            running = self.running.get(service)
            if running is None:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=1, stdout="", stderr="No such object"
                )
            return _ok(
                _inspect_json(
                    running,
                    test=["CMD-SHELL", "rpk cluster health | grep -q 'Healthy:.*true'"],
                    mounts=["/var/lib/redpanda/data"],
                )
            )
        return _ok()


def _observe(lane_state: _Lane) -> list[ModelDepsConvergenceFinding]:
    executor = DeployExecutor()
    with (
        patch("deploy_agent.executor._run", side_effect=lane_state.run),
        patch("deploy_agent.executor._compose_env", return_value={}),
    ):
        return executor.observe_deps_convergence(EnumRuntimeLane.DEV)


def _by_service(
    findings: list[ModelDepsConvergenceFinding],
) -> dict[str, ModelDepsConvergenceFinding]:
    return {f.service: f for f in findings}


class TestItNamesWhatIsAboutToBeReplaced:
    """The 2026-09-19 broker replacement, made legible before it happens."""

    def test_a_changed_dependency_is_named_with_both_hashes(self) -> None:
        lane = _Lane(
            rendered={"postgres": OLD_HASH, "redpanda": NEW_HASH, "valkey": OLD_HASH},
            running={"postgres": OLD_HASH, "redpanda": OLD_HASH, "valkey": OLD_HASH},
        )
        found = _by_service(_observe(lane))
        broker = found["redpanda"]
        assert broker.differs is True
        assert broker.running_config_hash == OLD_HASH
        assert broker.rendered_config_hash == NEW_HASH
        assert "REPLACE" in broker.describe()
        assert "redpanda" in broker.describe()

    def test_the_untouched_dependencies_are_recorded_as_unchanged(self) -> None:
        """A fact, not an absence: the deps were left alone on purpose."""
        lane = _Lane(
            rendered={"postgres": OLD_HASH, "redpanda": NEW_HASH, "valkey": OLD_HASH},
            running={"postgres": OLD_HASH, "redpanda": OLD_HASH, "valkey": OLD_HASH},
        )
        found = _by_service(_observe(lane))
        assert set(found) == set(CORE)
        assert found["postgres"].differs is False
        assert found["valkey"].differs is False
        assert "left running" in found["postgres"].describe()

    def test_it_names_which_fields_moved(self) -> None:
        """The account. `redpanda differs` and this are different amounts of help."""
        lane = _Lane(
            rendered={"postgres": OLD_HASH, "redpanda": NEW_HASH, "valkey": OLD_HASH},
            running={"postgres": OLD_HASH, "redpanda": OLD_HASH, "valkey": OLD_HASH},
        )
        fields = _by_service(_observe(lane))["redpanda"].changed_fields
        assert "healthcheck" in fields
        assert "mounts" in fields
        assert "environment keys" in fields
        assert "image" not in fields, (
            "the image did not move; saying it did is a false lead"
        )

    def test_no_environment_value_reaches_the_record(self) -> None:
        """Those values are the lane's broker credentials."""
        lane = _Lane(
            rendered={"postgres": OLD_HASH, "redpanda": NEW_HASH, "valkey": OLD_HASH},
            running={"postgres": OLD_HASH, "redpanda": OLD_HASH, "valkey": OLD_HASH},
        )
        findings = _observe(lane)
        blob = json.dumps([f.model_dump(mode="json") for f in findings])
        assert SECRET not in blob
        assert "DEV_KAFKA_SASL_PASSWORD" not in blob


class TestItObservesAndNeverGates:
    """A refusal here would strand the change; nothing emits a deps-only scope."""

    @pytest.mark.parametrize(
        ("lane", "expect_in_reason"),
        [
            (
                _Lane(rendered={}, running=dict.fromkeys(CORE, OLD_HASH), hash_rc=1),
                "unreadable",
            ),
            (
                _Lane(rendered=dict.fromkeys(CORE, OLD_HASH), running={}),
                "not inspectable",
            ),
        ],
    )
    def test_an_unreadable_comparison_is_reported_not_raised(
        self, lane: _Lane, expect_in_reason: str
    ) -> None:
        findings = _observe(lane)
        assert findings, "an unreadable lane must still produce findings"
        assert all(f.unreadable_reason for f in findings)
        assert any(expect_in_reason in f.unreadable_reason for f in findings)
        assert all(f.differs is False for f in findings), (
            "an unreadable comparison is UNKNOWN, and must never be reported as a "
            "confident 'nothing changed' nor as a replacement"
        )

    def test_a_failed_field_render_still_yields_the_authoritative_hashes(self) -> None:
        """The field account is best-effort; the hash is not."""
        lane = _Lane(
            rendered={"postgres": OLD_HASH, "redpanda": NEW_HASH, "valkey": OLD_HASH},
            running={"postgres": OLD_HASH, "redpanda": OLD_HASH, "valkey": OLD_HASH},
            render_json_rc=1,
        )
        broker = _by_service(_observe(lane))["redpanda"]
        assert broker.differs is True
        assert broker.changed_fields == ()
        assert "the hash covers more" in broker.describe(), (
            "an empty field list must never read as 'nothing changed'"
        )

    def test_the_deps_leg_still_runs_after_the_observation(self) -> None:
        executor = DeployExecutor()
        lane = _Lane(
            rendered={"postgres": OLD_HASH, "redpanda": NEW_HASH, "valkey": OLD_HASH},
            running={"postgres": OLD_HASH, "redpanda": OLD_HASH, "valkey": OLD_HASH},
        )
        with (
            patch("deploy_agent.executor._run", side_effect=lane.run),
            patch("deploy_agent.executor._compose_env", return_value={}),
            patch(
                "deploy_agent.executor.verify_containers_up", return_value=(True, [])
            ),
        ):
            executor._compose_up(
                Phase.CORE, Scope.CORE, [], _noop, lane=EnumRuntimeLane.DEV
            )

        ups = [c for c in lane.calls if "up" in c and "--profile" in c]
        assert ups, "the deps leg must still have issued its compose up"
        assert executor.deps_convergence, "and the observation must have been recorded"

    def test_the_observation_precedes_the_compose_up(self) -> None:
        executor = DeployExecutor()
        lane = _Lane(
            rendered=dict.fromkeys(CORE, OLD_HASH),
            running=dict.fromkeys(CORE, OLD_HASH),
        )
        with (
            patch("deploy_agent.executor._run", side_effect=lane.run),
            patch("deploy_agent.executor._compose_env", return_value={}),
            patch(
                "deploy_agent.executor.verify_containers_up", return_value=(True, [])
            ),
        ):
            executor._compose_up(
                Phase.CORE, Scope.CORE, [], _noop, lane=EnumRuntimeLane.DEV
            )

        first_up = next(
            i for i, c in enumerate(lane.calls) if "up" in c and "--profile" in c
        )
        first_hash = next(i for i, c in enumerate(lane.calls) if "--hash" in c)
        assert first_hash < first_up, (
            "the point of the record is to say what is about to happen; after the "
            "fact the only trace is a container timestamp"
        )


class TestTheArgvIsRecorded:
    """Readable nowhere else once the child exits."""

    def test_the_deps_leg_argv_is_recorded_with_its_phase(self) -> None:
        executor = DeployExecutor()
        lane = _Lane(
            rendered=dict.fromkeys(CORE, OLD_HASH),
            running=dict.fromkeys(CORE, OLD_HASH),
        )
        with (
            patch("deploy_agent.executor._run", side_effect=lane.run),
            patch("deploy_agent.executor._compose_env", return_value={}),
            patch(
                "deploy_agent.executor.verify_containers_up", return_value=(True, [])
            ),
        ):
            executor._compose_up(
                Phase.CORE, Scope.CORE, [], _noop, lane=EnumRuntimeLane.DEV
            )

        recorded = executor.compose_invocations
        assert recorded, "no compose invocation was recorded"
        deps = [i for i in recorded if i.phase is Phase.CORE]
        assert deps, f"no CORE-phase argv among {[i.phase for i in recorded]}"
        argv = deps[-1].argv
        assert "up" in argv and "--profile" in argv
        issued = [c for c in lane.calls if "up" in c and "--profile" in c]
        assert list(argv) == issued[-1], (
            "the record must be the argv ACTUALLY issued, not a reconstruction"
        )

    def test_the_deps_leg_of_a_full_rebuild_records_an_argv_with_no_force_flag(
        self,
    ) -> None:
        """#3805's behaviour, now readable from the job record instead of `ps`."""
        from deploy_agent.events import BuildSource

        executor = DeployExecutor()
        lane = _Lane(
            rendered=dict.fromkeys(CORE, OLD_HASH),
            running=dict.fromkeys(CORE, OLD_HASH),
        )
        with (
            patch("deploy_agent.executor._run", side_effect=lane.run),
            patch("deploy_agent.executor._compose_env", return_value={}),
            patch.object(DeployExecutor, "_compose_build", return_value=None),
            patch.object(
                DeployExecutor, "_build_dev_lane_only_services", return_value=None
            ),
            patch.object(DeployExecutor, "_deploy_gateway_lane", return_value=None),
            patch.object(
                DeployExecutor, "_ensure_runtime_migrations_ready", return_value=None
            ),
            patch(
                "deploy_agent.executor.verify_containers_up", return_value=(True, [])
            ),
            patch(
                "deploy_agent.executor.verify_oneshots_completed",
                return_value=(True, []),
            ),
        ):
            executor.rebuild_scope(
                Scope.FULL,
                [],
                _noop,
                git_sha="0" * 40,
                git_ref="origin/dev",
                build_source=BuildSource.RELEASE,
                lane=EnumRuntimeLane.DEV,
            )

        core_argvs = [
            i.argv
            for i in executor.compose_invocations
            if "--profile" in i.argv and i.argv[i.argv.index("--profile") + 1] == "core"
        ]
        assert core_argvs, "the deps leg issued no recorded argv"
        assert "--force-recreate" not in core_argvs[-1]
        runtime_argvs = [
            i.argv
            for i in executor.compose_invocations
            if "--profile" in i.argv
            and i.argv[i.argv.index("--profile") + 1] == "runtime"
            and "--no-deps" in i.argv
        ]
        assert runtime_argvs, "the runtime leg issued no recorded argv"
        assert "--force-recreate" in runtime_argvs[-1], (
            "the runtime leg still forces, and the record must show that too"
        )

    def test_the_recorded_argv_is_the_list_handed_to_the_kernel(self) -> None:
        executor = DeployExecutor()
        cmd = ["docker", "compose", "-p", "omnibase-infra", "up", "-d"]
        executor._record_compose_invocation(Phase.RUNTIME, cmd)
        assert executor.compose_invocations[0].argv == tuple(cmd)

    def test_a_job_resets_both_records(self) -> None:
        executor = DeployExecutor()
        executor._record_compose_invocation(Phase.CORE, ["docker", "compose", "up"])
        executor.deps_convergence.append(
            ModelDepsConvergenceFinding(
                service="redpanda",
                lane=EnumRuntimeLane.DEV,
                compose_project="omnibase-infra",
                differs=False,
            )
        )
        executor.reset_deploy_observations()
        assert executor.compose_invocations == []
        assert executor.deps_convergence == []


class TestBothReachTheTerminalEvent:
    """A fact that lives only in a journal is one the next reader reconstructs."""

    def test_the_completion_payload_carries_both_fields(self) -> None:
        from datetime import UTC, datetime

        from deploy_agent.job_state import JobState
        from deploy_agent.publisher import build_completion_payload

        job = JobState(
            correlation_id="00000000-0000-4000-8000-000000000001",
            command={"runtime_lane": "dev", "scope": "full", "git_ref": "origin/dev"},
            accepted_at=datetime.now(UTC),
        )
        job.phase_results = {Phase.CORE: PhaseStatus.SUCCESS}
        payload = build_completion_payload(
            job,
            "0" * 40,
            deps_convergence=[
                ModelDepsConvergenceFinding(
                    service="redpanda",
                    lane=EnumRuntimeLane.DEV,
                    compose_project="omnibase-infra",
                    running_config_hash=OLD_HASH,
                    rendered_config_hash=NEW_HASH,
                    differs=True,
                    changed_fields=("healthcheck", "mounts"),
                )
            ],
            compose_invocations=[
                ModelComposeInvocation(
                    phase=Phase.CORE, argv=("docker", "compose", "up", "-d")
                )
            ],
        )
        assert payload["deps_convergence"][0]["service"] == "redpanda"
        assert payload["deps_convergence"][0]["differs"] is True
        assert payload["deps_convergence"][0]["changed_fields"] == [
            "healthcheck",
            "mounts",
        ]
        assert payload["compose_invocations"][0]["argv"] == [
            "docker",
            "compose",
            "up",
            "-d",
        ]

    def test_the_agent_hands_the_executor_records_to_the_payload(self) -> None:
        """The wiring, not the model: a field nothing populates is not a record."""
        import inspect

        from deploy_agent import agent as agent_mod

        source = inspect.getsource(agent_mod)
        assert "deps_convergence=self.executor.deps_convergence" in source
        assert "compose_invocations=self.executor.compose_invocations" in source


class TestTheHashComparisonIsCompoundedFromRealInterfaces:
    """Guard the two interface assumptions this whole record rests on."""

    def test_the_rendered_hash_is_read_from_compose_config_hash(self) -> None:
        executor = DeployExecutor()
        seen: list[list[str]] = []

        def run(
            cmd: list[str], timeout: int, **_: object
        ) -> subprocess.CompletedProcess[str]:
            seen.append(list(cmd))
            return _ok("\n".join(f"{s} {OLD_HASH}" for s in CORE))

        with (
            patch("deploy_agent.executor._run", side_effect=run),
            patch("deploy_agent.executor._compose_env", return_value={}),
        ):
            hashes = executor._rendered_config_hashes(EnumRuntimeLane.DEV, list(CORE))

        assert hashes == dict.fromkeys(CORE, OLD_HASH)
        assert len(seen) == 1, "one subprocess for the whole set, not one per service"
        argv = seen[0]
        assert argv[:2] == ["docker", "compose"]
        assert "config" in argv and "--hash" in argv
        assert argv[argv.index("--hash") + 1] == ",".join(CORE)

    def test_the_running_hash_is_read_from_the_compose_label(self) -> None:
        """Not from a name, a digest or an image id: the label compose writes."""
        lane = _Lane(
            rendered=dict.fromkeys(CORE, OLD_HASH),
            running=dict.fromkeys(CORE, OLD_HASH),
        )
        findings = _observe(lane)
        assert all(f.running_config_hash == OLD_HASH for f in findings)
        inspects = [c for c in lane.calls if c[:2] == ["docker", "inspect"]]
        assert len(inspects) == len(CORE)

    def test_image_supplied_environment_is_not_reported_as_a_change(self) -> None:
        """A container carries more env than its declaration names; that is normal."""
        executor = DeployExecutor()
        rendered: dict[str, Any] = {"environment": {"DECLARED": "x"}}
        inspected: dict[str, Any] = {
            "Config": {"Env": ["DECLARED=x", "PATH=/usr/bin", "FROM_IMAGE=1"]}
        }
        assert "environment keys" not in executor._changed_fields(rendered, inspected)
