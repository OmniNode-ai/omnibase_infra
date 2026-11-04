# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise estimator image delivery through the deploy agent's build boundary."""

from __future__ import annotations

import json
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from deploy_agent import executor as execution
from deploy_agent.events import BuildSource, EnumRuntimeLane, Scope
from deploy_agent.executor import DeployExecutor

pytestmark = pytest.mark.unit
SHA = "a" * 40
OLD_SHA = "b" * 40
SOURCE = "ai.omninode.image.source-repo"
REVISION = "org.opencontainers.image.revision"
IMAGE = "omnibase-infra-cost-estimator-web:dev-lane"
SOURCE_REPOSITORY = "web-front-end"
SOURCE_URL = "https://example.invalid/web-front-end.git"


@pytest.fixture
def delivery(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(
        calls=[],
        labels=None,
        sha=SHA,
        fail=None,
        stamp=True,
        env={
            "COST_ESTIMATOR_WEB_SOURCE_REPOSITORY": SOURCE_REPOSITORY,
            "COST_ESTIMATOR_WEB_SOURCE_URL": SOURCE_URL,
        },
        disabled=(),
    )

    def run(
        argv: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        state.calls.append(argv)
        if state.fail and state.fail in argv:
            return subprocess.CompletedProcess(
                argv, 1, "sensitive output", "sensitive error"
            )
        if "rev-parse" in argv:
            return subprocess.CompletedProcess(argv, 0, state.sha + "\n", "")
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(
                argv, int(state.labels is None), json.dumps(state.labels), ""
            )
        if argv[:2] == ["docker", "build"] and state.stamp:
            state.labels = {SOURCE: SOURCE_REPOSITORY, REVISION: state.sha}
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(execution, "_run", run)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(execution, "_compose_env", lambda: dict(state.env))
    monkeypatch.setattr(
        execution,
        "lane_config_for",
        lambda lane: SimpleNamespace(
            disabled_services=state.disabled, compose_files=("base", "dev")
        ),
    )
    state.executor = DeployExecutor()
    state.executor._compose_build = Mock()
    return state


def build(state: SimpleNamespace, lane: EnumRuntimeLane = EnumRuntimeLane.DEV) -> None:
    state.executor._build_dev_lane_only_services(
        "c" * 40,
        lambda *args: None,
        build_source=BuildSource.WORKSPACE,
        lane=lane,
        git_ref="origin/dev",
    )


def builds(state: SimpleNamespace) -> list[list[str]]:
    return [argv for argv in state.calls if argv[:2] == ["docker", "build"]]


def test_missing_image_is_built_with_web_revision_before_recreate(
    delivery: SimpleNamespace,
) -> None:
    build(delivery)
    (argv,) = builds(delivery)
    assert f"{SOURCE}={SOURCE_REPOSITORY}" in argv
    assert f"{REVISION}={SHA}" in argv
    assert argv[argv.index("--tag") + 1] == IMAGE
    (clone,) = [c for c in delivery.calls if c[:2] == ["git", "clone"]]
    assert clone[clone.index("--branch") + 1] == "dev"
    assert SOURCE_URL in clone
    assert delivery.labels == {SOURCE: SOURCE_REPOSITORY, REVISION: SHA}
    delivery.executor._compose_build.assert_called_once()


def test_current_image_is_reused(delivery: SimpleNamespace) -> None:
    delivery.labels = {SOURCE: SOURCE_REPOSITORY, REVISION: SHA}
    build(delivery)
    assert not builds(delivery)
    assert any("rev-parse" in c for c in delivery.calls)


@pytest.mark.parametrize(
    "labels", [{SOURCE: SOURCE_REPOSITORY, REVISION: OLD_SHA}, {REVISION: SHA}, {}]
)
def test_stale_or_unstamped_image_is_rebuilt(
    delivery: SimpleNamespace, labels: dict[str, str]
) -> None:
    delivery.labels = labels
    build(delivery)
    assert len(builds(delivery)) == 1
    assert delivery.labels == {SOURCE: SOURCE_REPOSITORY, REVISION: SHA}


def test_next_web_merge_advances_the_image(delivery: SimpleNamespace) -> None:
    delivery.labels = {SOURCE: SOURCE_REPOSITORY, REVISION: SHA}
    delivery.sha = OLD_SHA
    build(delivery)
    assert f"{REVISION}={OLD_SHA}" in builds(delivery)[0]


def test_declared_image_override_is_the_build_target(delivery: SimpleNamespace) -> None:
    delivery.env["COST_ESTIMATOR_WEB_IMAGE"] = "estimator:isolated-lab"
    build(delivery)
    argv = builds(delivery)[0]
    assert argv[argv.index("--tag") + 1] == "estimator:isolated-lab"


@pytest.mark.parametrize(
    "name", ["COST_ESTIMATOR_WEB_SOURCE_REPOSITORY", "COST_ESTIMATOR_WEB_SOURCE_URL"]
)
def test_source_identity_is_explicit_and_has_no_public_default(
    delivery: SimpleNamespace,
    name: str,
) -> None:
    del delivery.env[name]
    with pytest.raises(RuntimeError, match=name):
        build(delivery)
    assert not delivery.calls


@pytest.mark.parametrize("lane", [EnumRuntimeLane.PROD, EnumRuntimeLane.STABILITY_TEST])
def test_non_dev_lane_does_no_estimator_work(
    delivery: SimpleNamespace, lane: EnumRuntimeLane
) -> None:
    build(delivery, lane)
    assert not delivery.calls
    delivery.executor._compose_build.assert_not_called()


def test_instance_without_the_page_does_not_build_it(delivery: SimpleNamespace) -> None:
    delivery.disabled = ("cost-estimator-web",)
    build(delivery)
    assert not delivery.calls
    delivery.executor._compose_build.assert_called_once()


@pytest.mark.parametrize("failure", ["clone", "build", "rev-parse"])
def test_source_or_build_failure_aborts_without_command_output(
    delivery: SimpleNamespace, failure: str
) -> None:
    delivery.fail = failure
    with pytest.raises(RuntimeError) as error:
        build(delivery)
    assert "sensitive" not in str(error.value)
    if failure != "build":
        assert not builds(delivery)


def test_invalid_revision_never_reaches_docker(delivery: SimpleNamespace) -> None:
    delivery.sha = "dev"
    with pytest.raises(RuntimeError, match="revision"):
        build(delivery)
    assert not builds(delivery)


def test_build_requires_label_readback(delivery: SimpleNamespace) -> None:
    delivery.stamp = False
    with pytest.raises(RuntimeError, match="revision"):
        build(delivery)
    assert len(builds(delivery)) == 1


def test_runtime_rebuild_builds_before_recreate(
    delivery: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "_hold_build_context", lambda purpose: nullcontext())
    monkeypatch.setattr(delivery.executor, "_preflight_runtime_boot", Mock())
    recreate = Mock(side_effect=RuntimeError("recreate reached"))
    monkeypatch.setattr(delivery.executor, "_compose_up", recreate)
    with pytest.raises(RuntimeError, match="recreate reached"):
        delivery.executor.rebuild_scope(
            Scope.RUNTIME,
            [],
            lambda *args: None,
            lane=EnumRuntimeLane.DEV,
            build_source=BuildSource.WORKSPACE,
        )
    assert delivery.labels == {SOURCE: SOURCE_REPOSITORY, REVISION: SHA}
    recreate.assert_called_once()


def test_build_failure_prevents_runtime_recreate(
    delivery: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "_hold_build_context", lambda purpose: nullcontext())
    recreate = Mock()
    monkeypatch.setattr(delivery.executor, "_compose_up", recreate)
    delivery.fail = "build"
    with pytest.raises(RuntimeError, match="exited"):
        delivery.executor.rebuild_scope(
            Scope.RUNTIME,
            [],
            lambda *args: None,
            lane=EnumRuntimeLane.DEV,
            build_source=BuildSource.WORKSPACE,
        )
    recreate.assert_not_called()


def test_dev_compose_passes_the_gate_settings_from_the_store() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    text = (root / "docker/docker-compose.dev-lane.yml").read_text()
    doc = yaml.safe_load(text.replace("!override", "").replace("!reset", ""))
    env = doc["services"]["cost-estimator-web"]["environment"]
    assert (
        env["FEATURE_ESTIMATOR_GATE_ENABLED"]
        == "${FEATURE_ESTIMATOR_GATE_ENABLED:-true}"
    )
    assert env["ESTIMATOR_GATE_PASSWORD"] == "${ESTIMATOR_GATE_PASSWORD:-}"


def test_compose_env_re_reads_the_materialized_gate_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = {"ESTIMATOR_GATE_PASSWORD": "synthetic-store-first"}
    monkeypatch.setattr(execution, "operator_store_overlay", lambda: dict(stored))
    monkeypatch.setattr(execution, "_load_runtime_policy_env", dict)
    monkeypatch.setattr(execution, "resolve_lane_overlay_dir", lambda env: None)
    monkeypatch.setattr(execution, "_lane_env_overlay", dict)
    monkeypatch.setattr(execution, "_dev_broker_policy_env", lambda env: {})
    monkeypatch.setattr(
        execution,
        "load_dev_boot_guard",
        lambda: SimpleNamespace(
            fold_compose_env=lambda policy, is_default: {},
        ),
    )
    assert (
        execution._compose_env()["ESTIMATOR_GATE_PASSWORD"]
        == stored["ESTIMATOR_GATE_PASSWORD"]
    )
    stored["ESTIMATOR_GATE_PASSWORD"] = "synthetic-store-next"
    assert (
        execution._compose_env()["ESTIMATOR_GATE_PASSWORD"]
        == stored["ESTIMATOR_GATE_PASSWORD"]
    )
