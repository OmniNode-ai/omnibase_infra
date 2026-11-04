# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""I/O boundary owned by the deploy-agent image-prune EFFECT contract (OMN-19998).

The host timer enters through the node's local-runtime command topic. Docker's
default dangling-only prune rechecks references during deletion; by-id removal
could instead untag an image retained for rollback after the inventory read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import yaml

from omnibase_infra.handlers.handler_deploy_agent_image_prune import build_plan
from omnibase_infra.models.model_deploy_agent_image_prune_policy import (
    ModelDeployAgentImagePrunePolicy,
)
from omnibase_infra.models.model_deploy_agent_image_prune_request import (
    ModelDeployAgentImagePruneRequest,
)
from omnibase_infra.models.model_deploy_agent_image_prune_result import (
    ModelDeployAgentImagePruneResult,
)

logger = logging.getLogger(__name__)


def read_prune_policy(path: Path) -> ModelDeployAgentImagePrunePolicy:
    """Read the explicitly supplied policy source, without choosing a host path."""
    return ModelDeployAgentImagePrunePolicy.model_validate(
        yaml.safe_load(path.read_text())["config"]["image_prune"]
    )


def _docker(*args: str, policy: ModelDeployAgentImagePrunePolicy) -> str:
    return subprocess.run(
        ["docker", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        timeout=policy.docker_timeout_seconds,
    ).stdout


def _prune(
    *, execute: bool, policy: ModelDeployAgentImagePrunePolicy
) -> ModelDeployAgentImagePruneResult:
    docker = partial(_docker, policy=policy)
    # Use the one source-label predicate declared by this node contract.
    label_filter = f"label={policy.source_label}={policy.source_value}"
    ids = sorted(
        set(
            docker(
                "image",
                "ls",
                "--quiet",
                "--no-trunc",
                "--filter",
                "dangling=true",
                "--filter",
                label_filter,
            ).split()
        )
    )
    images: list[dict[str, object]] = []
    batch = policy.inspect_batch_size
    for start in range(0, len(ids), batch):
        batch_ids = ids[start : start + batch]
        inspected = json.loads(docker("image", "inspect", *batch_ids))
        if not isinstance(inspected, list) or any(
            not isinstance(image, dict) or not isinstance(image.get("Id"), str)
            for image in inspected
        ):
            raise ValueError("Docker image inspection returned an invalid inventory")
        if sorted(image["Id"] for image in inspected) != batch_ids:
            raise ValueError(
                "Docker image inspection did not account for every listed image"
            )
        images.extend(inspected)
    now = datetime.now(UTC)
    plan = build_plan(images, set(), policy=policy, now=now)
    # No eligible image means no deletion; container inventory cannot add
    # a removal candidate. Avoid an unnecessary host-wide scan in that case.
    if plan.remove_image_ids:
        containers = docker("container", "ls", "--all", "--quiet").split()
        inuse: set[str] = set()
        for start in range(0, len(containers), batch):
            batch_containers = containers[start : start + batch]
            references = docker(
                "container",
                "inspect",
                "--format",
                "{{.Image}}",
                *batch_containers,
            ).split()
            if len(references) != len(batch_containers):
                raise ValueError(
                    "Docker container inspection returned an incomplete inventory"
                )
            inuse.update(references)
        plan = build_plan(images, inuse, policy=policy, now=now)
    logger.info(
        "Deploy agent image prune plan (bytes are nominal): %s",
        plan.model_dump_json(),
    )
    if execute and plan.remove_image_ids:
        output = docker(
            "image",
            "prune",
            "--force",
            "--filter",
            f"until={plan.cutoff_epoch}",
            "--filter",
            label_filter,
        )
        return plan.model_copy(update={"stdout": output})
    return plan


def _install(
    repository: str, *, policy: ModelDeployAgentImagePrunePolicy, policy_path: Path
) -> ModelDeployAgentImagePruneResult:
    repo = Path(repository).resolve()
    if "\n" in str(repo):
        raise ValueError("Source root must not contain a newline")
    if not (repo / ".venv/bin/onex").is_file():
        raise ValueError(
            "Run uv sync --frozen in the source checkout before installing"
        )
    source = repo / policy.unit_source_directory
    home = Path(policy.user_home_path).expanduser()
    unit_dir = home / policy.unit_directory
    environment_file = home / policy.environment_file
    # Read all source bytes before changing any installed artifact.
    units = {name: (source / name).read_bytes() for name in policy.unit_names}
    unit_dir.mkdir(parents=True, exist_ok=True)
    environment_file.parent.mkdir(parents=True, exist_ok=True)
    quoted_root = str(repo).replace("\\", "\\\\").replace('"', '\\"')
    environment_file.write_text(f'DEPLOY_AGENT_GC_REPO="{quoted_root}"\n')
    for name, content in units.items():
        (unit_dir / name).write_bytes(content)
        if (unit_dir / name).read_bytes() != content:
            raise OSError(f"Installed unit readback differs: {name}")
    for args in (
        ("daemon-reload",),
        ("enable", "--now", policy.unit_names[1]),
    ):
        subprocess.run(
            ["systemctl", "--user", *args],
            check=True,
            timeout=policy.docker_timeout_seconds,
        )
    revision = hashlib.sha256(
        b"".join(units.values())
        + policy_path.read_bytes()
        + Path(__file__).read_bytes()
        + Path(build_plan.__code__.co_filename).read_bytes()
    ).hexdigest()
    return ModelDeployAgentImagePruneResult(success=True, installed_revision=revision)


def run_request(
    request: ModelDeployAgentImagePruneRequest,
) -> ModelDeployAgentImagePruneResult:
    """Read policy and run host I/O within the owning effect package."""
    try:
        policy_path = Path(request.policy_path)
        policy = read_prune_policy(policy_path)
        if request.operation == "deploy_agent.images.install":
            return _install(request.repository, policy=policy, policy_path=policy_path)
        return _prune(
            execute=request.operation == "deploy_agent.images.prune", policy=policy
        )
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        KeyError,
        yaml.YAMLError,
    ) as exc:
        return ModelDeployAgentImagePruneResult(success=False, error_message=str(exc))
