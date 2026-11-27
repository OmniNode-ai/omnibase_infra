# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev runtime's pin file is written where its compose files read it (OMN-25901).

Integration of the real writer, real git clones and the real dev compose files: the
writer is run against a registry of real git repositories, and the file it writes is
at the host path each compose file mounts read-only and is the file each compose file's
``DOD_VERIFY_PRODUCT_CLONE_PIN_FILE`` names inside the container. The verifier's own
measure (``git rev-list --count HEAD..<pin>``) then reads a fast-forwarded clone as
fresh and a clone reset to an unrelated commit as stale.
"""

from __future__ import annotations

import json
import os
import posixpath
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.nodes.node_setup_local_provision_effect import (
    runtime_overlay_revision as writer,
)

pytestmark = [pytest.mark.integration]

REPO = Path(__file__).resolve().parents[3]
AGENT_SOURCE = REPO / "scripts/deploy-agent/deploy_agent/executor.py"
PIN_ENV = "DOD_VERIFY_PRODUCT_CLONE_PIN_FILE"
COMPOSE_HOSTS: tuple[tuple[str, str], ...] = (
    ("docker-compose.dev-lane.yml", "runtime-effects"),
    ("docker-compose.dev-tenant-fold.yml", "omninode-runtime"),
)


def _git(clone: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(clone),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()


def _compose_service(filename: str, service: str) -> dict[str, object]:
    """The service as compose text declares it; the ``!override`` tags are dropped."""
    text = (REPO / "docker" / filename).read_text(encoding="utf-8")
    text = text.replace("!override", "").replace("!reset", "")
    found = yaml.safe_load(text)["services"][service]
    assert isinstance(found, dict)
    return found


def _agent(root: Path) -> ModuleType:
    """A stand-in for the deploy agent module: its source location and compose clone."""
    agent = ModuleType("deploy_agent_stand_in")
    agent.__file__ = str(AGENT_SOURCE)
    vars(agent)["deploy_repo_dir"] = lambda: str(root / "omnibase_infra")
    return agent


def _registry(root: Path) -> tuple[ModuleType, dict[str, Path]]:
    execution = _agent(root)
    declared = writer.product_clone_pin_declaration(execution)
    names = declared["repositories"]
    assert isinstance(names, list)
    clones: dict[str, Path] = {}
    for name in names:
        clone = root / str(name)
        clone.mkdir(parents=True)
        _git(clone, "init", "-q")
        _git(clone, "commit", "-q", "--allow-empty", "-m", "first")
        clones[str(name)] = clone
    return execution, clones


def _behind(clone: Path, pinned_sha: str) -> str:
    return _git(clone, "rev-list", "--count", f"HEAD..{pinned_sha}")


@pytest.mark.parametrize(("filename", "service"), COMPOSE_HOSTS)
def test_written_pin_file_is_the_file_the_compose_service_reads(
    tmp_path: Path, filename: str, service: str
) -> None:
    execution, _ = _registry(tmp_path)
    written = writer.write_product_clone_pins(execution)

    declared = _compose_service(filename, service)
    environment = declared["environment"]
    volumes = declared["volumes"]
    assert isinstance(environment, dict)
    assert isinstance(volumes, list)
    container_file = str(environment[PIN_ENV])
    mount = next(
        str(v)
        for v in volumes
        if str(v).endswith(":ro") and ":/etc/onex/product-clone-pin:" in str(v)
    )
    host_dir, container_dir, _mode = mount.split(":")
    assert posixpath.dirname(container_file) == container_dir
    # ``docker/`` of the agent's compose clone is where the relative host path starts.
    compose_dir = tmp_path / "omnibase_infra" / "docker"
    assert (compose_dir / host_dir).resolve() == written.parent.resolve()
    assert written.name == posixpath.basename(container_file)


def test_a_fast_forwarded_clone_is_fresh_and_a_reset_clone_is_stale(
    tmp_path: Path,
) -> None:
    execution, clones = _registry(tmp_path)
    written = writer.write_product_clone_pins(execution)
    pins = {
        pin["repo_root"].removeprefix("/app/"): pin["pinned_sha"]
        for pin in json.loads(written.read_text(encoding="utf-8"))["pins"]
    }
    assert set(pins) == set(clones)
    for name, clone in clones.items():
        assert pins[name] == _git(clone, "rev-parse", "HEAD")

    ahead, reset = clones["omnimarket"], clones["omnimemory"]
    _git(ahead, "commit", "-q", "--allow-empty", "-m", "fast-forward after the pin")
    _git(reset, "checkout", "-q", "--orphan", "unrelated")
    _git(reset, "commit", "-q", "--allow-empty", "-m", "unrelated")

    assert _behind(ahead, pins["omnimarket"]) == "0"
    assert _behind(reset, pins["omnimemory"]) != "0"
