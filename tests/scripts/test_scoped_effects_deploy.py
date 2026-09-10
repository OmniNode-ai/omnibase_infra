# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the scoped executor with only Docker and time replaced [OMN-17991]."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "runtime_build"
CURRENT_ID = "a" * 64
BROKER_ID = "b" * 64
CORE_ID = "c" * 64
CANDIDATE_ID = "d" * 64
ROLLBACK_ID = "e" * 64
OLD_IMAGE = "sha256:" + "1" * 64
NEW_IMAGE = "sha256:" + "2" * 64
INFRA_SOURCE = "7" * 40
PINS = {"omnibase_core": "3" * 40, "omnibase_compat": "4" * 40, "omnimarket": "5" * 40}
TOPIC = "onex.evt.omnimarket.closeout-proof-matrix-completed.v1"


@pytest.fixture
def executor(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("scoped_effects_deploy")


class DockerBoundary:
    """Stateful Docker boundary: real filesystem, executor and adjudication run."""

    def __init__(self, compose_file: Path) -> None:
        self.compose_file = compose_file
        self.calls: list[list[str]] = []
        self.clock = 0.0
        self.mode = "ok"
        self.last_target: dict[str, Any] | None = None
        self.rendered = {
            "name": "omnibase-infra",
            "services": {
                "runtime-effects": {
                    "image": "old-effects",
                    "environment": {"RUNTIME_PROFILE": "effects"},
                    "volumes": [],
                    "networks": {"default": None},
                    "depends_on": {
                        "redpanda": {"condition": "service_healthy"},
                        "omninode-runtime": {"condition": "service_healthy"},
                    },
                }
            },
            "networks": {"default": {"name": "omnibase-infra_default"}},
        }
        self.containers = {
            CURRENT_ID: self._container(
                CURRENT_ID, "runtime-effects", OLD_IMAGE, compose_file
            ),
            BROKER_ID: self._container(
                BROKER_ID, "redpanda", "broker-image", compose_file
            ),
            CORE_ID: self._container(
                CORE_ID, "omninode-runtime", "core-image", compose_file
            ),
        }
        self.image_config = {
            "Env": ["RUNTIME_PROFILE=effects"],
            "Entrypoint": ["entrypoint"],
        }

    def _config_hash(self, rendered: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(rendered["services"]["runtime-effects"], sort_keys=True).encode()
        ).hexdigest()

    def _container(
        self, cid: str, service: str, image: str, compose_file: Path
    ) -> dict[str, Any]:
        return {
            "Id": cid,
            "Name": "/" + service,
            "Image": image,
            "Config": {
                "Env": ["RUNTIME_PROFILE=effects"],
                "Entrypoint": ["entrypoint"],
                "Labels": {
                    "com.docker.compose.project": "omnibase-infra",
                    "com.docker.compose.service": service,
                    "com.docker.compose.project.working_dir": str(compose_file.parent),
                    "com.docker.compose.project.config_files": str(compose_file),
                    "com.docker.compose.version": "5.1.0",
                    "com.docker.compose.config-hash": self._config_hash(self.rendered),
                },
            },
            "HostConfig": {},
            "State": {
                "Running": True,
                "Status": "running",
                "StartedAt": "before",
                "Health": {"Status": "healthy"},
            },
            "Mounts": [],
            "NetworkSettings": {"Networks": {"omnibase-infra_default": {}}},
        }

    def _probe(self, image: str, *, live: bool = False) -> str:
        probe: dict[str, Any] = {
            "infra_source_ref": INFRA_SOURCE[:12],
            "base_image_id": OLD_IMAGE if image == NEW_IMAGE else None,
            "market_payload_fingerprint": ("e" if image == OLD_IMAGE else "f") * 64,
            "source_pins": PINS if image == NEW_IMAGE else {},
            "dependency_fingerprint": "a" * 64,
            "shared_runtime_fingerprint": ("b" if live else "c") * 64,
            "required_topics": [TOPIC],
        }
        if image == NEW_IMAGE:
            if self.mode == "dependencies":
                probe["dependency_fingerprint"] = "d" * 64
            elif self.mode == "shared_runtime":
                probe["shared_runtime_fingerprint"] = "d" * 64
            elif self.mode == "source_pins":
                probe["source_pins"] = {**PINS, "omnimarket": "6" * 40}
            elif self.mode == "topics":
                probe["required_topics"] = [TOPIC, "onex.evt.omnimarket.new-topic.v1"]
            elif self.mode == "infra_source":
                probe["infra_source_ref"] = "9" * 12
            elif self.mode == "base_image":
                probe["base_image_id"] = "sha256:" + "9" * 64
        if self.mode == "market_live_drift" and live and image == OLD_IMAGE:
            probe["market_payload_fingerprint"] = "d" * 64
        if self.mode == "market_after_drift" and live and image == NEW_IMAGE:
            probe["market_payload_fingerprint"] = "d" * 64
        return json.dumps(probe)

    def __call__(self, args: list[str], stdin: str | None, timeout: int) -> str:
        self.calls.append(args)
        assert timeout > 0
        if args[0] == "git":
            ref = args[-1].removesuffix("^{commit}")
            return (INFRA_SOURCE if ref == INFRA_SOURCE[:12] else ref) + "\n"
        if args[0] == "uv":
            assert "--frozen" in args and "--no-sync" in args
            assert args[args.index("--container") + 1] == "runtime-effects"
            actual_refs = {
                args[index + 1]
                for index, arg in enumerate(args)
                if arg == "--build-ref"
            }
            assert actual_refs == {
                f"{repo}={ref}"
                for repo, ref in {"omnibase_infra": INFRA_SOURCE, **PINS}.items()
            }
            assert "--skip-tripwire" not in args and "--cold-start" not in args
            if self.mode == "hotpatch":
                raise RuntimeError("hotpatch preflight failed")
            if self.mode == "lock_stolen":
                lock = self.compose_file.parent.parent / "state" / ".deploy.lock"
                lock.rename(lock.with_name(".saved-original-lock"))
                lock.mkdir()
                (lock / "pid").write_text("9999999\n")
            return "HOTPATCH-PREFLIGHT PASS\n"
        if args[:3] == ["docker", "ps", "--all"]:
            return "\n".join(self.containers)
        if args[:2] == ["docker", "inspect"]:
            return json.dumps([self.containers[cid] for cid in args[2:]])
        if args[:3] == ["docker", "image", "inspect"]:
            layers = ["sha256:" + "a" * 64]
            if args[-1] == NEW_IMAGE:
                layers.append("sha256:" + "b" * 64)
                if self.mode == "lineage":
                    layers[0] = "sha256:" + "9" * 64
            return json.dumps(
                [
                    {
                        "Id": args[-1],
                        "Config": self.image_config,
                        "RootFS": {"Layers": layers},
                    }
                ]
            )
        if args[:3] == ["docker", "image", "ls"]:
            return ""
        if args[:2] == ["docker", "tag"]:
            assert args[2] == OLD_IMAGE
            return ""
        if args[:2] == ["docker", "run"]:
            assert "--network" in args and args[args.index("--network") + 1] == "none"
            assert "--read-only" in args
            assert stdin
            image = next(arg for arg in args if arg in (OLD_IMAGE, NEW_IMAGE))
            if self.mode == "race_after_probe" and image == NEW_IMAGE:
                self.containers[CURRENT_ID]["Image"] = "sha256:" + "9" * 64
            return self._probe(image)
        if args[:2] == ["docker", "exec"]:
            if "rpk" in args:
                return "NAME PARTITIONS REPLICAS\n" + (
                    "" if self.mode == "missing_topic" else f"{TOPIC} 1 1\n"
                )
            cid = args[3] if args[2] == "-i" else args[2]
            result = self._probe(self.containers[cid]["Image"], live=True)
            if self.containers[cid]["Image"] == NEW_IMAGE:
                if self.mode == "target_stops_after_probe":
                    self.containers[cid]["State"].update(Running=False, Status="exited")
                elif self.mode == "target_pauses_after_probe":
                    self.containers[cid]["State"]["Paused"] = True
                elif self.mode == "core_stops_after_probe":
                    self.containers[CORE_ID]["State"].update(
                        Running=False, Status="exited"
                    )
                elif self.mode == "foreign_after_probe":
                    foreign = self.containers.pop(cid)
                    foreign["Id"] = "9" * 64
                    self.containers[foreign["Id"]] = foreign
            return result
        if args[:2] == ["docker", "compose"]:
            if "version" in args:
                return "5.1.0\n" if self.mode != "compose_version" else "5.2.0\n"
            if "config" in args:
                if "--hash" in args:
                    files = [
                        Path(args[index + 1])
                        for index, arg in enumerate(args)
                        if arg == "-f"
                    ]
                    rendered = (
                        json.loads(files[0].read_text())
                        if files[0].name == "admitted.compose.json"
                        else self.rendered
                    )
                    return "runtime-effects " + self._config_hash(rendered) + "\n"
                return json.dumps(self.rendered)
            assert "up" in args
            assert args[-1] == "runtime-effects"
            assert all(
                flag in args
                for flag in ("--no-deps", "--no-build", "--force-recreate", "--pull")
            )
            assert args[args.index("--pull") + 1] == "never"
            files = [
                Path(args[index + 1]) for index, arg in enumerate(args) if arg == "-f"
            ]
            image = json.loads(files[-1].read_text())["services"]["runtime-effects"][
                "image"
            ]
            old_id = next(
                (
                    cid
                    for cid, container in self.containers.items()
                    if container["Config"]["Labels"]["com.docker.compose.service"]
                    == "runtime-effects"
                ),
                None,
            )
            target = self.containers.pop(old_id) if old_id else self.last_target
            assert target is not None
            self.last_target = target
            new_id = CANDIDATE_ID if image == NEW_IMAGE else ROLLBACK_ID
            target["Id"] = new_id
            target["Image"] = image
            target["State"]["StartedAt"] = new_id
            target["State"].update(Running=True, Status="running", Paused=False)
            target["Config"]["Labels"]["com.docker.compose.project.config_files"] = (
                ",".join(map(str, files))
            )
            target["State"]["Health"]["Status"] = "healthy"
            if (
                self.mode in ("candidate_unhealthy", "rollback_unhealthy")
                and image == NEW_IMAGE
            ):
                target["State"]["Health"]["Status"] = "unhealthy"
            if self.mode == "rollback_unhealthy" and image == OLD_IMAGE:
                target["State"]["Health"]["Status"] = "unhealthy"
            self.containers[new_id] = target
            if self.mode == "collateral_change" and image == NEW_IMAGE:
                self.containers[CORE_ID]["State"]["StartedAt"] = "foreign-restart"
            if self.mode == "lost_ack" and image == NEW_IMAGE:
                raise RuntimeError("compose timed out after recreation")
            if self.mode == "missing_target" and image == NEW_IMAGE:
                del self.containers[new_id]
                raise RuntimeError("compose failed between removal and creation")
            if self.mode == "interrupted" and image == NEW_IMAGE:
                raise KeyboardInterrupt
            if self.mode == "unfenced" and image == NEW_IMAGE:
                raise importlib.import_module(
                    "scoped_effects_deploy"
                ).CommandProcessUnfencedError(
                    "plugin process group termination unproved"
                )
            if self.mode == "foreign_replacement" and image == NEW_IMAGE:
                target["Config"]["Labels"][
                    "com.docker.compose.project.config_files"
                ] = "/foreign/compose.yaml"
            if self.mode == "changed_compose_after_up" and image == NEW_IMAGE:
                self.compose_file.write_text("foreign compose change\n")
            if self.mode == "changed_snapshot_after_up" and image == NEW_IMAGE:
                files[0].chmod(0o600)
                files[0].write_text("foreign compose change\n")
            return ""
        raise AssertionError(f"Unexpected Docker boundary call: {args}")

    def sleep(self, seconds: float) -> None:
        self.clock += seconds


@pytest.fixture
def scenario(executor: Any, tmp_path: Path) -> tuple[Any, DockerBoundary, Path]:
    compose_file = tmp_path / "active" / "compose.yaml"
    compose_file.parent.mkdir()
    compose_file.write_text("services: {}\n", encoding="utf-8")
    clones = tmp_path / "clones"
    for repo in ("omnibase_infra", *PINS):
        (clones / repo).mkdir(parents=True)
    ledger = tmp_path / "hotpatch-ledger.yaml"
    ledger.write_text("schema: 1\nrows: []\n")
    plan = executor.ModelEffectsDeployPlan.model_validate_json(
        json.dumps(
            {
                "schema_version": "1",
                "ticket_id": "OMN-17991",
                "reason": "Verify approved effects-only rollout",
                "compose_project": "omnibase-infra",
                "service": "runtime-effects",
                "expected_container_id": CURRENT_ID,
                "expected_image_id": OLD_IMAGE,
                "candidate_image_id": NEW_IMAGE,
                "compose_files": [str(compose_file)],
                "compose_working_dir": str(compose_file.parent),
                "expected_compose_sha256": executor.compose_chain_sha256(
                    (compose_file,)
                ),
                "source_pins": PINS,
                "health_timeout_seconds": 1,
                "infra_source_sha": INFRA_SOURCE,
                "source_clones_root": str(clones),
                "hotpatch_ledger": str(ledger),
            }
        )
    )
    return plan, DockerBoundary(compose_file), tmp_path / "state"


def _execute(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path]
) -> dict[str, Any]:
    plan, docker, state = scenario
    return executor.run_deploy(
        plan,
        execute=True,
        state_root=state,
        runner=docker,
        sleep=docker.sleep,
        monotonic=lambda: docker.clock,
    )


def test_preview_has_no_mutations_or_state_files(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path]
) -> None:
    plan, docker, state = scenario
    result = executor.run_deploy(plan, state_root=state, runner=docker)
    assert result["status"] == "PREVIEW"
    assert result["compatibility_verified"] is False
    assert not state.exists()
    assert not any(call[1] in ("run", "tag") or "up" in call for call in docker.calls)


def test_success_only_replaces_effects_and_keeps_rollback(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path]
) -> None:
    plan, docker, state = scenario
    state.mkdir()
    registry = state / "registry.json"
    registry.write_text('{"old": true}\n')
    before_core = copy.deepcopy(docker.containers[CORE_ID])
    before_broker = copy.deepcopy(docker.containers[BROKER_ID])
    result = _execute(executor, scenario)
    assert result["status"] == "PASSED"
    assert docker.containers[CANDIDATE_ID]["Image"] == NEW_IMAGE
    assert docker.containers[CORE_ID] == before_core
    assert docker.containers[BROKER_ID] == before_broker
    assert registry.read_text() == '{"old": true}\n'
    assert (
        executor.compose_chain_sha256(plan.compose_files)
        == plan.expected_compose_sha256
    )
    assert not (state / ".deploy.lock").exists()
    assert Path(result["record_dir"]).joinpath("receipt.json").is_file()
    receipt = json.loads(
        Path(result["record_dir"]).joinpath("receipt.json").read_text()
    )
    assert receipt["status"] == "PASSED"
    assert receipt["actor"]["uid"] >= 0
    assert receipt["actor"]["user"] and receipt["actor"]["host"]
    assert receipt["started_at"] <= receipt["finished_at"]
    snapshot = Path(result["record_dir"]) / "admitted.compose.json"
    assert snapshot.stat().st_mode & 0o777 == 0o400
    frozen = json.loads(snapshot.read_text())
    assert frozen["networks"]["default"] == {
        "name": "omnibase-infra_default",
        "external": True,
    }
    up = next(call for call in docker.calls if "up" in call)
    assert str(snapshot) in up and str(plan.compose_files[0]) not in up
    assert any(
        call[:2] == ["docker", "tag"] and call[2] == OLD_IMAGE for call in docker.calls
    )
    assert sum("up" in call for call in docker.calls) == 1
    assert not any(
        any(
            word in call for word in ("prune", "down", "build", "produce", "seek", "rm")
        )
        for call in docker.calls
    )


@pytest.mark.parametrize(
    "mode",
    [
        "dependencies",
        "shared_runtime",
        "source_pins",
        "infra_source",
        "topics",
        "missing_topic",
        "race_after_probe",
        "hotpatch",
        "base_image",
        "lineage",
        "market_live_drift",
        "lock_stolen",
    ],
)
def test_refusal_never_recreates(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path], mode: str
) -> None:
    _, docker, _ = scenario
    docker.mode = mode
    result = _execute(executor, scenario)
    assert result["status"] == "REFUSED"
    assert not any("up" in call for call in docker.calls)


@pytest.mark.parametrize(
    "mode",
    [
        "candidate_unhealthy",
        "lost_ack",
        "missing_target",
        "interrupted",
        "market_after_drift",
        "target_stops_after_probe",
        "target_pauses_after_probe",
    ],
)
def test_failure_recreates_and_verifies_old_image(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path], mode: str
) -> None:
    _, docker, _ = scenario
    docker.mode = mode
    result = _execute(executor, scenario)
    assert result["status"] == "FAILED_ROLLED_BACK"
    assert docker.containers[ROLLBACK_ID]["Image"] == OLD_IMAGE
    assert docker.containers[ROLLBACK_ID]["State"]["Health"]["Status"] == "healthy"
    assert sum("up" in call for call in docker.calls) == 2


def test_rollback_failure_never_claims_recovery(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path]
) -> None:
    _, docker, _ = scenario
    docker.mode = "rollback_unhealthy"
    result = _execute(executor, scenario)
    assert result["status"] == "FAILED_ROLLBACK"
    assert result["rollback_verified"] is False


def test_collateral_change_cannot_pass_even_after_target_rollback(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path]
) -> None:
    _, docker, _ = scenario
    docker.mode = "collateral_change"
    result = _execute(executor, scenario)
    assert result["status"] == "FAILED_ROLLBACK"
    assert docker.containers[ROLLBACK_ID]["Image"] == OLD_IMAGE
    assert docker.containers[CORE_ID]["State"]["StartedAt"] == "foreign-restart"


@pytest.mark.parametrize(
    "mode", ["foreign_replacement", "foreign_after_probe", "changed_snapshot_after_up"]
)
def test_rollback_does_not_overwrite_foreign_state(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path], mode: str
) -> None:
    _, docker, _ = scenario
    docker.mode = mode
    result = _execute(executor, scenario)
    assert result["status"] == "FAILED_ROLLBACK"
    assert result["rollback_verified"] is False
    assert sum("up" in call for call in docker.calls) == 1


@pytest.mark.parametrize("mode", ["core_stops_after_probe", "changed_compose_after_up"])
def test_rollback_uses_frozen_config_but_never_hides_external_changes(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path], mode: str
) -> None:
    _, docker, _ = scenario
    docker.mode = mode
    result = _execute(executor, scenario)
    assert result["status"] == "FAILED_ROLLBACK"
    assert docker.containers[ROLLBACK_ID]["Image"] == OLD_IMAGE
    assert sum("up" in call for call in docker.calls) == 2


def test_changed_lock_owner_is_preserved(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path]
) -> None:
    _, docker, state = scenario
    docker.mode = "lock_stolen"
    assert _execute(executor, scenario)["status"] == "REFUSED"
    assert (state / ".deploy.lock" / "pid").read_text() == "9999999\n"


def test_unfenced_forward_process_holds_lock_and_withholds_rollback(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path]
) -> None:
    _, docker, state = scenario
    docker.mode = "unfenced"
    result = _execute(executor, scenario)
    assert result["status"] == "FAILED_UNFENCED"
    assert result["rollback_verified"] is False
    assert (state / ".deploy.lock" / "pid").is_file()
    assert sum("up" in call for call in docker.calls) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ports", [{"target": 8080, "published": "9090"}]),
        ("restart", "always"),
        ("cap_add", ["SYS_ADMIN"]),
        ("security_opt", ["seccomp=unconfined"]),
        ("healthcheck", {"test": ["CMD", "false"]}),
        ("mem_limit", "128m"),
    ],
)
def test_complete_compose_hash_refuses_unhandled_runtime_drift(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path], field: str, value: Any
) -> None:
    _, docker, _ = scenario
    docker.rendered["services"]["runtime-effects"][field] = value
    with pytest.raises(RuntimeError, match="Compose hash"):
        _execute(executor, scenario)
    assert not any(
        "up" in call or call[:2] == ["docker", "run"] for call in docker.calls
    )


def test_compose_version_skew_refuses_before_mutation(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path]
) -> None:
    _, docker, _ = scenario
    docker.mode = "compose_version"
    with pytest.raises(RuntimeError, match="version"):
        _execute(executor, scenario)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("NanoCpus", 1_000_000_000),
        ("CpuPeriod", 100000),
        ("CpuQuota", 10000),
        ("CpuRealtimePeriod", 1000),
        ("CpuRealtimeRuntime", 500),
        ("CpuShares", 512),
        ("CpusetCpus", "0-2"),
        ("CpusetMems", "0"),
        ("Memory", 1024**3),
        ("MemoryReservation", 1024**2),
        ("MemorySwap", -1),
        ("MemorySwappiness", 25),
        ("OomKillDisable", True),
        ("PidsLimit", 100),
        ("BlkioWeight", 500),
        ("RestartPolicy", {"Name": "always", "MaximumRetryCount": 0}),
    ],
)
def test_live_docker_update_drift_refuses_with_unchanged_compose_hash(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path], field: str, value: Any
) -> None:
    _, docker, _ = scenario
    docker.containers[CURRENT_ID]["HostConfig"][field] = value
    with pytest.raises(RuntimeError, match="mutable effects"):
        _execute(executor, scenario)
    assert not any(
        "up" in call or call[:2] == ["docker", "run"] for call in docker.calls
    )


def test_declared_mutable_resources_preserve_success(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path]
) -> None:
    _, docker, _ = scenario
    docker.rendered["services"]["runtime-effects"].update(
        deploy={
            "resources": {
                "limits": {"cpus": "1.5", "memory": "2g", "pids": 100},
                "reservations": {"memory": "128m"},
            }
        },
        restart="on-failure:3",
        cpu_shares=512,
    )
    docker.containers[CURRENT_ID]["Config"]["Labels"][
        "com.docker.compose.config-hash"
    ] = docker._config_hash(docker.rendered)
    docker.containers[CURRENT_ID]["HostConfig"].update(
        NanoCpus=1_500_000_000,
        Memory=2 * 1024**3,
        MemorySwap=4 * 1024**3,
        MemoryReservation=128 * 1024**2,
        PidsLimit=100,
        CpuShares=512,
        RestartPolicy={"Name": "on-failure", "MaximumRetryCount": 3},
    )
    assert _execute(executor, scenario)["status"] == "PASSED"


@pytest.mark.parametrize("deploy_pids", [-1, 0])
def test_nonpositive_deploy_pids_preserves_service_limit(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path], deploy_pids: int
) -> None:
    _, docker, _ = scenario
    docker.rendered["services"]["runtime-effects"].update(
        deploy={"resources": {"limits": {"pids": deploy_pids}}},
        pids_limit=80,
    )
    docker.containers[CURRENT_ID]["Config"]["Labels"][
        "com.docker.compose.config-hash"
    ] = docker._config_hash(docker.rendered)
    docker.containers[CURRENT_ID]["HostConfig"]["PidsLimit"] = 80
    assert _execute(executor, scenario)["status"] == "PASSED"


def test_timeout_kills_plugin_child_that_ignores_termination(
    executor: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "escaped-child"
    ready = tmp_path / "child-ready"
    child = (
        "import pathlib,signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(ready)!r}).write_text('ready'); "
        "time.sleep(3.5); "
        f"pathlib.Path({str(marker)!r}).write_text('unsafe surviving child')"
    )
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
    monkeypatch.setattr(executor, "_PROCESS_TERM_GRACE", 0.1)
    with pytest.raises(RuntimeError, match="process group was terminated"):
        executor._run_command([sys.executable, "-c", parent], None, 3)
    assert ready.is_file(), "the actual plugin-like child must have started"
    time.sleep(0.7)
    assert not marker.exists()


def test_existing_lock_is_never_stolen(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path]
) -> None:
    _, docker, state = scenario
    lock = state / ".deploy.lock"
    lock.mkdir(parents=True)
    marker = lock / "pid"
    marker.write_text("99999999\n")
    with pytest.raises(RuntimeError, match="lock"):
        _execute(executor, scenario)
    assert marker.read_text() == "99999999\n"
    assert not any(call[1] in ("run", "tag") or "up" in call for call in docker.calls)


def test_hotpatch_bypass_is_rejected_even_when_receipted(
    executor: Any,
    scenario: tuple[Any, DockerBoundary, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOTPATCH_PREFLIGHT_BYPASS", "# skip-token-allowed: receipt")
    with pytest.raises(RuntimeError, match="BYPASS"):
        _execute(executor, scenario)
    assert not scenario[1].calls


def test_hotpatch_unknown_source_refuses_host_head_fallback(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path]
) -> None:
    plan, docker, _ = scenario
    plan.hotpatch_ledger.write_text(
        "schema: 1\nrows:\n  - container: runtime-effects\n    source_repo: omniintelligence\n"
    )
    result = _execute(executor, scenario)
    assert result["status"] == "REFUSED"
    assert not any("up" in call or call[0] == "uv" for call in docker.calls)


def test_extra_live_environment_refuses_recreate(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path]
) -> None:
    _, docker, _ = scenario
    docker.containers[CURRENT_ID]["Config"]["Env"].append("REMOVED_SETTING=stale")
    result = _execute(executor, scenario)
    assert result["status"] == "REFUSED"
    assert not any("up" in call for call in docker.calls)


@pytest.mark.parametrize(
    "change",
    [
        "compose_bytes",
        "active_image",
        "working_dir",
        "environment",
        "dependency_health",
    ],
)
def test_drift_refuses_before_offline_probes(
    executor: Any, scenario: tuple[Any, DockerBoundary, Path], change: str
) -> None:
    plan, docker, _ = scenario
    if change == "compose_bytes":
        plan.compose_files[0].write_text("changed\n")
    elif change == "active_image":
        docker.containers[CURRENT_ID]["Image"] = NEW_IMAGE
    elif change == "working_dir":
        docker.containers[CURRENT_ID]["Config"]["Labels"][
            "com.docker.compose.project.working_dir"
        ] = "/different"
    elif change == "environment":
        docker.rendered["services"]["runtime-effects"]["environment"][
            "RUNTIME_PROFILE"
        ] = "different"
    else:
        docker.containers[BROKER_ID]["State"]["Health"]["Status"] = "unhealthy"
    with pytest.raises(RuntimeError):
        _execute(executor, scenario)
    assert not any(call[1] in ("run", "tag") or "up" in call for call in docker.calls)
