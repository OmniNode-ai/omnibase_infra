# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The overlay store root the deploy agent names is the variable the dev compose files mount (OMN-24216).

``runtime_overlay_revision.apply`` sets the environment variable its node contract declares as
``store_root_env`` to the rendered revision's overlays folder. The dev compose files mount the
overlay store root from a variable of their own. This is the seam between the two: the names must
match, and an environment the agent built must reach the mount source. The revision is rendered
from a real throwaway git repository by the module as the agent loads it (by path).
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from tests.ci._compose_render_omn21099 import load_compose

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
NODE_DIR = REPO_ROOT / "src/omnibase_infra/nodes/node_setup_local_provision_effect"
COMPOSE_FILES = (
    REPO_ROOT / "docker/docker-compose.dev-lane.yml",
    REPO_ROOT / "docker/docker-compose.dev-tenant-fold.yml",
)
STORE_ROOT_TARGET = "/etc/onex/skill-overlays-store"
EXECUTION = SimpleNamespace(
    __file__=str(REPO_ROOT / "scripts/deploy-agent/deploy_agent/executor.py")
)


def _module() -> ModuleType:
    """The module the way the standalone agent loads it: by path, not as a package import."""
    name = "omn24216_runtime_overlay_revision"
    spec = importlib.util.spec_from_file_location(
        name, NODE_DIR / "runtime_overlay_revision.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _declared() -> dict[str, str]:
    contract = yaml.safe_load((NODE_DIR / "contract.yaml").read_text(encoding="utf-8"))
    return {k: str(v) for k, v in contract["config"]["overlay_revision_render"].items()}


def _mount_specs(compose: Path) -> list[str]:
    document = load_compose(compose.name)
    return [
        volume
        for service in document["services"].values()
        for volume in service.get("volumes", [])
        if isinstance(volume, str) and f":{STORE_ROOT_TARGET}" in volume
    ]


def _interpolate(spec: str, environ: dict[str, str]) -> str:
    """The compose ``${VAR:-default}`` rule: the variable when set and non-empty, else the default."""

    def value(match: re.Match[str]) -> str:
        return environ.get(match.group(1)) or match.group(2) or ""

    return re.sub(r"\$\{(\w+)(?::-([^}]*))?\}", value, spec)


@pytest.mark.parametrize("compose", COMPOSE_FILES, ids=lambda path: path.name)
def test_the_contract_names_the_variable_the_compose_mount_reads(compose: Path) -> None:
    specs = _mount_specs(compose)
    assert specs, f"{compose.name} mounts nothing at {STORE_ROOT_TARGET}"
    for spec in specs:
        assert "${" + _declared()["store_root_env"] in spec


def test_an_environment_the_agent_built_reaches_the_mount_source(
    tmp_path: Path,
) -> None:
    declared = _declared()
    clone = tmp_path / "overlays-clone"
    folder = clone / declared["overlay_dir"]
    folder.mkdir(parents=True)
    (folder / "lane.yaml").write_text("mode: on_demand\n", encoding="utf-8")
    git = ["git", "-C", str(clone)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run(
        [*git, "-c", "user.email=lane@example.invalid", "-c", "user.name=lane"]
        + ["commit", "-q", "-m", "seed"],
        check=True,
    )
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()

    environ = {declared["clone_env"]: str(clone)}
    _module().apply(EXECUTION, tmp_path / "state", environ, head)

    for compose in COMPOSE_FILES:
        for spec in _mount_specs(compose):
            source = _interpolate(spec, environ).split(":")[0]
            assert source == str(
                tmp_path / "state" / declared["rendered_dir_name"] / head / "overlays"
            )
            assert (Path(source) / "lane.yaml").read_text(encoding="utf-8") == (
                "mode: on_demand\n"
            )


def test_without_a_revision_the_mount_source_is_the_host_copy_default() -> None:
    """The failing control: no revision means no override, so the mount keeps its default."""
    for compose in COMPOSE_FILES:
        for spec in _mount_specs(compose):
            source = _interpolate(spec, {}).split(":")[0]
            assert source.endswith("overlays/overlays")
