# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the deployed image-prune node, its retention table and installer."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from omnibase_core.runtime.runtime_local import RuntimeLocal
from omnibase_infra.handlers.handler_deploy_agent_image_prune import (
    HandlerDeployAgentImagePrune,
    build_plan,
    read_prune_policy,
)
from omnibase_infra.models.model_deploy_agent_image_prune_request import (
    ModelDeployAgentImagePruneRequest,
)
from omnibase_infra.models.model_deploy_agent_image_prune_result import (
    ModelDeployAgentImagePruneResult,
)

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = (
    ROOT / "src/omnibase_infra/nodes/node_deploy_agent_image_prune_effect/contract.yaml"
)
NOW = datetime(2026, 11, 4, 13, tzinfo=UTC)
IMAGE_ID = "sha256:" + "a" * 64


def make_request(**kwargs: str) -> ModelDeployAgentImagePruneRequest:
    return ModelDeployAgentImagePruneRequest(policy_path=str(CONTRACT), **kwargs)


def image(**changes: object) -> dict[str, object]:
    policy = read_prune_policy(CONTRACT)
    value: dict[str, object] = {
        "Id": IMAGE_ID,
        "Created": (NOW - timedelta(days=2)).isoformat(),
        "RepoTags": [],
        "RepoDigests": [],
        "Size": 1234,
        "Config": {"Labels": {policy.source_label: policy.source_value}},
    }
    value.update(changes)
    return value


def test_old_unreferenced_runtime_image_is_selected() -> None:
    plan = build_plan([image()], set(), policy=read_prune_policy(CONTRACT), now=NOW)
    assert plan.remove_image_ids == (IMAGE_ID,)
    assert plan.remove_nominal_bytes == 1234
    assert plan.cutoff_epoch == int((NOW - timedelta(hours=24)).timestamp())


@pytest.mark.parametrize(
    ("changes", "inuse"),
    [
        ({"RepoTags": ["runtime:rollback-dev"]}, set()),
        ({"RepoTags": ["runtime:latest"]}, set()),
        ({"RepoDigests": ["runtime@sha256:" + "b" * 64]}, set()),
        ({}, {IMAGE_ID}),
        ({"Created": NOW.isoformat()}, set()),
        ({"Created": (NOW - timedelta(hours=24)).isoformat()}, set()),
        ({"Created": "unreadable"}, set()),
        ({"Config": {"Labels": {}}}, set()),
        ({"Config": {"Labels": {"other": "unrelated"}}}, set()),
    ],
)
def test_references_rollback_age_and_other_images_are_kept(
    changes: dict[str, object], inuse: set[str]
) -> None:
    plan = build_plan(
        [image(**changes)], inuse, policy=read_prune_policy(CONTRACT), now=NOW
    )
    assert plan.remove_image_ids == ()
    assert IMAGE_ID in plan.kept_reasons


def docker_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log = tmp_path / "calls.jsonl"
    stub = tmp_path / "docker"
    stub.write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        "a=sys.argv[1:]\n"
        "with open(os.environ['CALL_LOG'], 'a') as f: f.write(json.dumps(a)+'\\n')\n"
        "if os.environ.get('FAIL_INVENTORY') and a[:2] == ['container', 'ls']: sys.exit(7)\n"
        "if os.environ.get('FAIL_INSPECT') and a[:2] == ['container', 'inspect']: sys.exit(7)\n"
        "if a[:2] == ['image', 'ls']: print(os.environ['IMAGE_ID'])\n"
        "elif a[:2] == ['image', 'inspect']: print(os.environ['IMAGE_JSON'])\n"
        "elif a[:2] == ['container', 'ls']:\n"
        " if os.environ.get('CONTAINER_IMAGE'): print('c1')\n"
        "elif a[:2] == ['container', 'inspect']: print(os.environ['CONTAINER_IMAGE'])\n"
        "elif a[:2] == ['image', 'prune']: print('Total reclaimed space: 0B')\n"
        "else: sys.exit('unexpected docker command: '+repr(a))\n"
    )
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("CALL_LOG", str(log))
    monkeypatch.setenv("IMAGE_ID", IMAGE_ID)
    monkeypatch.setenv(
        "IMAGE_JSON", json.dumps([image(Created="2020-01-01T00:00:00Z")])
    )
    return log


@pytest.mark.parametrize("execute", [False, True])
def test_handler_runs_only_scoped_dangling_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool
) -> None:
    log = docker_stub(tmp_path, monkeypatch)
    handler = HandlerDeployAgentImagePrune()
    request = make_request(
        operation="deploy_agent.images.prune" if execute else "deploy_agent.images.plan"
    )
    plan = asyncio.run(handler.handle(request))
    assert plan.success, plan.error_message
    assert plan.remove_image_ids == (IMAGE_ID,)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    prune = [call for call in calls if call[:2] == ["image", "prune"]]
    assert bool(prune) is execute
    if execute:
        assert prune == [
            [
                "image",
                "prune",
                "--force",
                "--filter",
                f"until={plan.cutoff_epoch}",
                "--filter",
                f"label={read_prune_policy(CONTRACT).source_label}={read_prune_policy(CONTRACT).source_value}",
            ]
        ]
    assert not any(
        call[0] in {"builder", "volume", "compose", "restart", "rm"} for call in calls
    )


@pytest.mark.parametrize("failure", ["FAIL_INVENTORY", "FAIL_INSPECT"])
def test_unreadable_inventory_refuses_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    log = docker_stub(tmp_path, monkeypatch)
    monkeypatch.setenv("CONTAINER_IMAGE", IMAGE_ID)
    monkeypatch.setenv(failure, "1")
    result = asyncio.run(
        HandlerDeployAgentImagePrune().handle(
            make_request(operation="deploy_agent.images.prune")
        )
    )
    assert not result.success
    assert result.error_message
    assert "prune" not in log.read_text()


def test_handler_keeps_image_referenced_by_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = docker_stub(tmp_path, monkeypatch)
    monkeypatch.setenv("CONTAINER_IMAGE", IMAGE_ID)
    result = asyncio.run(
        HandlerDeployAgentImagePrune().handle(
            make_request(operation="deploy_agent.images.prune")
        )
    )
    assert result.success, result.error_message
    assert result.remove_image_ids == ()
    assert "prune" not in log.read_text()


def test_timer_and_service_are_declared_with_node_installer() -> None:
    entries = yaml.safe_load((ROOT / "deploy/unit-drift-manifest.yaml").read_text())[
        "units"
    ]
    agent_hosts = next(
        item["hosts"] for item in entries if item["name"] == "deploy-agent-dev.service"
    )
    for suffix in ("service", "timer"):
        name = f"onex-deploy-agent-image-prune.{suffix}"
        entry = next(item for item in entries if item["name"] == name)
        assert entry["hosts"] == agent_hosts
        assert entry["installer_args"][:2] == ["node", CONTRACT.parent.name]
        assert (ROOT / entry["tracked"]).is_file()
    service = (
        ROOT / "scripts/deploy-agent/deploy/onex-deploy-agent-image-prune.service"
    ).read_text()
    assert "node node_deploy_agent_image_prune_effect" in service
    assert "prune.json" in service
    timer = (
        ROOT / "scripts/deploy-agent/deploy/onex-deploy-agent-image-prune.timer"
    ).read_text()
    assert "OnUnitActiveSec=1h" in timer
    assert (
        "onex-deploy-agent-image-prune.timer"
        in (ROOT / "scripts/deploy-agent/deploy/deploy-agent-dev.service").read_text()
    )


def test_installer_enables_only_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "systemctl.log"
    systemctl = tmp_path / "systemctl"
    systemctl.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$SYSTEMCTL_LOG"\n')
    systemctl.chmod(0o755)
    home = tmp_path / "home"
    repo = tmp_path / "branch with spaces"
    source = repo / "scripts/deploy-agent/deploy"
    source.mkdir(parents=True)
    (repo / ".venv/bin").mkdir(parents=True)
    (repo / ".venv/bin/onex").touch()
    handler = HandlerDeployAgentImagePrune()
    for name in read_prune_policy(CONTRACT).unit_names:
        shutil.copyfile(ROOT / "scripts/deploy-agent/deploy" / name, source / name)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("SYSTEMCTL_LOG", str(log))
    monkeypatch.setenv("HOME", str(home))
    result = asyncio.run(
        handler.handle(
            make_request(operation="deploy_agent.images.install", repository=str(repo))
        )
    )
    assert result.success, result.error_message
    assert len(result.installed_revision) == 64
    assert log.read_text().splitlines() == [
        "--user daemon-reload",
        "--user enable --now onex-deploy-agent-image-prune.timer",
    ]
    for name in read_prune_policy(CONTRACT).unit_names:
        assert (home / ".config/systemd/user" / name).read_bytes() == (
            source / name
        ).read_bytes()
    assert str(repo) in (home / ".config/onex/deploy-agent-image-prune.env").read_text()


@pytest.mark.parametrize(
    ("operation", "fail"),
    [
        ("deploy_agent.images.plan", False),
        ("deploy_agent.images.prune", False),
        ("deploy_agent.images.prune", True),
    ],
)
def test_actual_runtime_dispatches_contract_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, fail: bool
) -> None:
    log = docker_stub(tmp_path, monkeypatch)
    if fail:
        monkeypatch.setenv("FAIL_INVENTORY", "1")
    input_path = tmp_path / "request.json"
    input_path.write_text(make_request(operation=operation).model_dump_json())
    runtime = RuntimeLocal(
        CONTRACT,
        input_path=input_path,
        state_root=tmp_path / "state",
        backend_overrides={"event_bus": "inmemory"},
        host_handlers=True,
    )
    runtime.run()
    assert isinstance(runtime.handler_result, ModelDeployAgentImagePruneResult), (
        runtime.last_error
    )
    assert runtime.handler_result.success is not fail
    assert (runtime.exit_code != 0) is fail
    if not fail:
        assert runtime.handler_result.remove_image_ids == (IMAGE_ID,)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert any(call[:2] == ["image", "prune"] for call in calls) is (
        operation == "deploy_agent.images.prune" and not fail
    )


def test_no_old_images_needs_no_container_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = docker_stub(tmp_path, monkeypatch)
    monkeypatch.setenv(
        "IMAGE_JSON", json.dumps([image(Created=datetime.now(UTC).isoformat())])
    )
    monkeypatch.setenv("FAIL_INVENTORY", "1")
    result = asyncio.run(
        HandlerDeployAgentImagePrune().handle(
            make_request(operation="deploy_agent.images.prune")
        )
    )
    assert result.success, result.error_message
    assert result.remove_image_ids == ()
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert all(call[0] == "image" for call in calls)
    assert "prune" not in log.read_text()
