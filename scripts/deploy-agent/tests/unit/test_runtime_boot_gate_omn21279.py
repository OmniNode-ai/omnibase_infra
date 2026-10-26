# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A build that cannot resolve its boot contract must never replace dev."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import executor as executor_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import (
    EnumRuntimeLane,
    ModelRebuildRequested,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import DeployExecutor
from deploy_agent.job_state import JobStore

pytestmark = [pytest.mark.unit, pytest.mark.runtime_boot_gate]

# Scrubbed from the 2026-10-25 dev crash-loop (job 709248b9, revision
# 03ea9377): no host, credentials, endpoint or customer data retained.
BOOT_FAILURE = (
    "ProtocolConfigurationError: ONEX_CORE_041_INVALID_CONFIGURATION: "
    "BIFROST_BASE_CONTRACT_PATH is unset or blank; there is no default"
)


@pytest.fixture
def candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[DeployExecutor, list[str], Mock]:
    executor = DeployExecutor()
    order: list[str] = []
    monkeypatch.setattr(
        executor_mod, "_hold_build_context", lambda *a: contextlib.nullcontext()
    )
    monkeypatch.setattr(executor_mod, "_compose_env", lambda *a: {})
    monkeypatch.setattr(
        executor, "_compose_build", lambda *a, **kw: order.append("build")
    )
    monkeypatch.setattr(executor, "_build_dev_lane_only_services", Mock())
    monkeypatch.setattr(executor, "_compose_up", lambda *a, **kw: order.append("swap"))

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "run" in argv:
            order.append("probe")
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr=BOOT_FAILURE)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    runner = Mock(side_effect=run)
    monkeypatch.setattr(executor_mod, "_run", runner)
    return executor, order, runner


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_crash_loop_image_refused_before_any_swap(
    candidate: tuple[DeployExecutor, list[str], Mock], scope: Scope
) -> None:
    executor, order, _ = candidate
    with pytest.raises(RuntimeError, match="BIFROST_BASE_CONTRACT_PATH is unset"):
        executor.rebuild_scope(scope, [], Mock(), git_sha="03ea9377" + "0" * 32)
    assert order[-1] == "probe"
    assert "build" in order
    assert "swap" not in order  # Old containers are never stopped or recreated.


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_successful_candidates_all_probed_before_swap(
    candidate: tuple[DeployExecutor, list[str], Mock], scope: Scope
) -> None:
    executor, order, runner = candidate

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "run" in argv:
            order.append("probe")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    runner.side_effect = run
    executor.rebuild_scope(scope, [], Mock())
    assert order.count("probe") == 3
    assert max(i for i, step in enumerate(order) if step == "probe") < order.index(
        "swap"
    )
    probes = [call.args[0] for call in runner.call_args_list if "run" in call.args[0]]
    assert [argv[-3] for argv in probes] == [
        "omninode-runtime",
        "runtime-effects",
        "runtime-worker",
    ]
    for argv in probes:
        assert "--no-deps" in argv and "--rm" in argv
        assert "--build" not in argv and "--service-ports" not in argv
        assert argv[argv.index("--pull") + 1] == "never"
        assert argv[argv.index("--user") + 1] == "omniinfra"
        name = argv[argv.index("--name") + 1]
        assert any(
            call.args[0] == ["docker", "rm", "-f", name]
            for call in runner.call_args_list
        )


def test_candidate_probes_use_temporary_no_build_override(
    candidate: tuple[DeployExecutor, list[str], Mock],
) -> None:
    executor, _, runner = candidate
    lane_args = executor_mod._compose_file_args(EnumRuntimeLane.DEV)
    overrides: list[Path] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "run" in argv:
            assert argv[2 : 2 + len(lane_args)] == lane_args
            extra_args = argv[2 + len(lane_args) :]
            assert extra_args[0] == "-f"
            path = Path(extra_args[1])
            overrides.append(path)
            service = argv[-3]
            content = path.read_text()
            assert f"  {service}:\n" in content
            assert "build: !reset null" in content
            assert f"image: omnibase-infra-{service}" in content
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    runner.side_effect = run
    executor.rebuild_scope(Scope.RUNTIME, [], Mock())
    assert len(overrides) == 3
    assert all(not path.exists() for path in overrides)


def test_timeout_refuses_swap_marks_phase_failed_and_cleans_only_probe(
    candidate: tuple[DeployExecutor, list[str], Mock],
) -> None:
    executor, order, runner = candidate

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "run" in argv:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    runner.side_effect = run
    phase = Mock()
    with pytest.raises(
        RuntimeError, match=r"boot probe timed out.*running containers kept"
    ):
        executor.rebuild_scope(Scope.RUNTIME, [], phase)
    assert "swap" not in order
    phase.assert_called_with(Phase.RUNTIME, PhaseStatus.FAILED)
    probe = runner.call_args_list[0].args[0]
    assert runner.call_args_list[-1].args[0] == [
        "docker",
        "rm",
        "-f",
        probe[probe.index("--name") + 1],
    ]


def test_service_request_probes_only_requested_kernel(
    candidate: tuple[DeployExecutor, list[str], Mock],
) -> None:
    executor, order, runner = candidate
    with pytest.raises(RuntimeError, match="runtime-effects"):
        executor.rebuild_scope(Scope.RUNTIME, ["runtime-effects"], Mock())
    assert runner.call_args_list[0].args[0][-3] == "runtime-effects"
    assert "swap" not in order


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
@pytest.mark.parametrize("failed_service", ["runtime-effects", "runtime-worker"])
def test_later_kernel_failure_still_preserves_all_running_containers(
    candidate: tuple[DeployExecutor, list[str], Mock],
    scope: Scope,
    failed_service: str,
) -> None:
    executor, order, runner = candidate

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        failed = "run" in argv and argv[-3] == failed_service
        return subprocess.CompletedProcess(
            argv, int(failed), stdout="", stderr=BOOT_FAILURE if failed else ""
        )

    runner.side_effect = run
    with pytest.raises(RuntimeError, match=failed_service):
        executor.rebuild_scope(scope, [], Mock())
    assert "swap" not in order


@pytest.mark.parametrize("lane", [EnumRuntimeLane.PROD, EnumRuntimeLane.STABILITY_TEST])
def test_gate_is_dev_only(
    candidate: tuple[DeployExecutor, list[str], Mock], lane: EnumRuntimeLane
) -> None:
    executor, _, runner = candidate
    executor._preflight_runtime_boot([], Mock(), lane=lane)
    runner.assert_not_called()


@pytest.mark.parametrize(
    "base", [None, "", " ", "missing", "malformed", "valid", "legacy", "disabled"]
)
def test_candidate_script_uses_real_boot_contract_resolver(
    tmp_path: Path, base: str | None
) -> None:
    env = {**os.environ, "BIFROST_CONTRACT_PATH": str(tmp_path / "rendered.yaml")}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[4] / "src")
    env.pop("BIFROST_BASE_CONTRACT_PATH", None)
    program = executor_mod.RUNTIME_BOOT_PROBE
    if base == "disabled":
        env["BIFROST_CONTRACT_PATH"] = ""
    if base == "legacy":
        # The known-good pre-outage resolver has a zero-argument API.
        program = (
            "from pathlib import Path\n"
            "import os\n"
            "import omnibase_infra.runtime.render_bifrost_delegation_contract as renderer\n"
            "renderer._resolve_canonical_source_path = lambda: Path(os.environ['PROBE_LEGACY_BASE'])\n"
            + program
        )
    if base is not None:
        path = tmp_path / "base.yaml"
        if base in {"malformed", "valid", "legacy"}:
            path.write_text("[broken" if base == "malformed" else "backends: []\n")
        if base != "legacy":
            env["BIFROST_BASE_CONTRACT_PATH"] = str(path) if base.strip() else base
        env["PROBE_LEGACY_BASE"] = str(path)
    result = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert (result.returncode == 0) == (base in {"valid", "legacy", "disabled"}), (
        result.stderr
    )
    if base is None or not base.strip():
        assert "BIFROST_BASE_CONTRACT_PATH is unset or blank" in result.stderr
    assert not (tmp_path / "rendered.yaml").exists()


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
@pytest.mark.parametrize("has_rollback", [False, True])
def test_refused_candidate_records_failed_job_with_boot_reason(
    candidate: tuple[DeployExecutor, list[str], Mock],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: Scope,
    has_rollback: bool,
) -> None:
    real_executor, order, _ = candidate
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(agent_mod, "publish_result", lambda *a: False)
    monkeypatch.setattr(
        agent_mod,
        "capture_rollback_point",
        Mock(return_value=object() if has_rollback else None),
    )
    restore = Mock()
    monkeypatch.setattr(agent_mod, "restore_rollback_point", restore)
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=scope,
        runtime_lane=EnumRuntimeLane.DEV,
    )
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = JobStore(tmp_path)
    agent.job_store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
    executor = Mock(spec=DeployExecutor)
    executor.git_pull.return_value = "03ea9377" + "0" * 32
    executor.verify.return_value = []
    executor.run_lane_mirror.return_value = None
    executor.rebuild_scope.side_effect = real_executor.rebuild_scope
    for name in (
        "container_residue",
        "recreate_supervision",
        "verify_recreate",
        "oneshot_reruns",
        "deps_convergence",
        "compose_invocations",
        "health_checks",
    ):
        setattr(executor, name, [])
    executor.cloud_migration_failure = ""
    executor.sibling_source_refs = {}
    agent.executor = executor
    monkeypatch.setattr(agent, "_resolve_lab_overlay_sha", lambda *a, **kw: None)
    agent._run_deploy(cmd)
    job = agent.job_store.load(cmd.correlation_id)
    assert job is not None and job.status == "failed"
    assert BOOT_FAILURE in " ".join(job.errors)
    assert "swap" not in order
    executor.verify.assert_not_called()
    restore.assert_not_called()  # Refusal needs no destructive rollback.
