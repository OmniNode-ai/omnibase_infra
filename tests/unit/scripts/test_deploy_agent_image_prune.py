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

from omnibase_infra.handlers.deploy_agent_image_prune_effect.effect import (
    read_prune_policy,
)
from omnibase_infra.handlers.handler_deploy_agent_image_prune import (
    HandlerDeployAgentImagePrune,
    build_plan,
)
from omnibase_infra.models.model_deploy_agent_image_prune_request import (
    ModelDeployAgentImagePruneRequest,
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
    policy = read_prune_policy(CONTRACT)
    assert calls[0] == [
        "image",
        "ls",
        "--quiet",
        "--no-trunc",
        "--filter",
        "dangling=true",
        "--filter",
        f"label={policy.source_label}={policy.source_value}",
    ]
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
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any(call[:2] == ["image", "prune"] for call in calls)


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
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any(call[:2] == ["image", "prune"] for call in calls)


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
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any(call[:2] == ["image", "prune"] for call in calls)


@pytest.mark.parametrize(
    "created",
    ["not-a-date", None, "2020-01-01T00:00:00", "2020-01-01T00:00:00.123456789Z"],
)
def test_handler_handles_docker_creation_time_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, created: str | None
) -> None:
    log = docker_stub(tmp_path, monkeypatch)
    monkeypatch.setenv("IMAGE_JSON", json.dumps([image(Created=created)]))
    result = asyncio.run(
        HandlerDeployAgentImagePrune().handle(
            make_request(operation="deploy_agent.images.prune")
        )
    )
    assert result.success, result.error_message
    eligible = created == "2020-01-01T00:00:00.123456789Z"
    assert bool(result.remove_image_ids) is eligible
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert any(call[:2] == ["image", "prune"] for call in calls) is eligible


@pytest.mark.parametrize(
    "inspected", ["missing", "duplicate", "wrong-id", "not-a-list"]
)
def test_inventory_must_account_for_every_listed_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inspected: str
) -> None:
    log = docker_stub(tmp_path, monkeypatch)
    monkeypatch.setenv("IMAGE_ID", IMAGE_ID + "\nsha256:" + "b" * 64)
    if inspected == "duplicate":
        monkeypatch.setenv("IMAGE_JSON", json.dumps([image(), image()]))
    elif inspected == "wrong-id":
        monkeypatch.setenv(
            "IMAGE_JSON", json.dumps([image(), image(Id="sha256:" + "c" * 64)])
        )
    elif inspected == "not-a-list":
        monkeypatch.setenv("IMAGE_JSON", json.dumps({"error": "invalid inventory"}))
    # A successful but incomplete inspect must not authorize a daemon-wide prune.
    result = asyncio.run(
        HandlerDeployAgentImagePrune().handle(
            make_request(operation="deploy_agent.images.prune")
        )
    )
    assert not result.success
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any(call[:2] == ["image", "prune"] for call in calls)


def test_image_and_container_inspection_are_batched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnibase_infra.handlers.deploy_agent_image_prune_effect import effect

    policy = read_prune_policy(CONTRACT).model_copy(update={"inspect_batch_size": 2})
    image_ids = ["sha256:" + char * 64 for char in "abcde"]
    container_ids = ["c1", "c2", "c3", "c4", "c5"]
    calls: list[tuple[str, ...]] = []

    def docker(*args: str, policy: object) -> str:
        calls.append(args)
        if args[:2] == ("image", "ls"):
            return "\n".join(image_ids)
        if args[:2] == ("image", "inspect"):
            return json.dumps(
                [
                    image(Id=ident, Created="2020-01-01T00:00:00.123456789Z")
                    for ident in args[2:]
                ]
            )
        if args[:2] == ("container", "ls"):
            return "\n".join(container_ids)
        if args[:2] == ("container", "inspect"):
            return "\n".join(
                image_ids[container_ids.index(ident)] for ident in args[4:]
            )
        raise AssertionError(args)

    monkeypatch.setattr(effect, "_docker", docker)
    result = effect._prune(execute=True, policy=policy, planner=build_plan)
    assert result.success
    assert result.remove_image_ids == ()
    assert [call[2:] for call in calls if call[:2] == ("image", "inspect")] == [
        tuple(image_ids[:2]),
        tuple(image_ids[2:4]),
        tuple(image_ids[4:]),
    ]
    assert [call[4:] for call in calls if call[:2] == ("container", "inspect")] == [
        tuple(container_ids[:2]),
        tuple(container_ids[2:4]),
        tuple(container_ids[4:]),
    ]


def test_docker_timeout_refuses_prune(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from omnibase_infra.handlers.deploy_agent_image_prune_effect import effect

    calls: list[list[str]] = []

    def timeout(args: list[str], **kwargs: object) -> object:
        calls.append(args)
        assert kwargs["timeout"] == read_prune_policy(CONTRACT).docker_timeout_seconds
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(effect.subprocess, "run", timeout)
    result = asyncio.run(HandlerDeployAgentImagePrune().handle(make_request()))
    assert not result.success
    assert "timed out" in result.error_message
    assert len(calls) == 1
    assert calls[0][:3] == ["docker", "image", "ls"]


@pytest.mark.parametrize(
    "failure", ["missing-venv", "missing-unit", "systemctl", "newline"]
)
def test_installer_failure_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from omnibase_infra.handlers.deploy_agent_image_prune_effect import effect

    repo = tmp_path / ("bad\npath" if failure == "newline" else "branch")
    source = repo / "scripts/deploy-agent/deploy"
    source.mkdir(parents=True)
    if failure != "missing-venv":
        (repo / ".venv/bin").mkdir(parents=True)
        (repo / ".venv/bin/onex").touch()
    if failure != "missing-unit":
        for name in read_prune_policy(CONTRACT).unit_names:
            shutil.copyfile(ROOT / "scripts/deploy-agent/deploy" / name, source / name)
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    calls: list[list[str]] = []

    def fail(args: list[str], **kwargs: object) -> object:
        import subprocess

        calls.append(args)
        raise subprocess.CalledProcessError(7, args)

    monkeypatch.setattr(effect.subprocess, "run", fail)
    result = asyncio.run(
        HandlerDeployAgentImagePrune().handle(
            make_request(operation="deploy_agent.images.install", repository=str(repo))
        )
    )
    assert not result.success
    assert result.error_message
    if failure == "systemctl":
        assert calls == [["systemctl", "--user", "daemon-reload"]]
    else:
        assert not calls
        assert not home.exists()
