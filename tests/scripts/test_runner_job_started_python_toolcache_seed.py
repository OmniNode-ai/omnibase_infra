# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A fresh runner starts with CPython already in its Actions tool cache (OMN-19998).

setup-python looks for ``<tool cache>/Python/<x.y.z>/<arch>`` beside an
``<arch>.complete`` marker and downloads CPython from GitHub when it finds
none. A runner that cannot reach GitHub quickly times out on that download
before its first real step. The image now bakes that layout, and a pool opts
into it with ``AGENT_TOOLSDIRECTORY`` plus a named volume per runner.

The file is named by the ticket's acceptance falsifier commands. The first
design seeded the cache from ``runner-job-started.sh``; the canonical-file-shape
ratchet refused growth of that baselined script, so the seed is the image bake
plus a pool opt-in, and the behavioural check of the built layout is the
``runner-image-build-smoke`` step this file pins.

The last test is the seam between the package index the fleet config declares
and the one the runner hook wires into every job for pip and uv.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "runners" / "Dockerfile"
COMPOSES = [
    REPO_ROOT / "docker" / "docker-compose.runners-hp051-ci-runner.yml",
    REPO_ROOT / "docker" / "docker-compose.runners-hp053-ci-runner.yml",
]
SMOKE = REPO_ROOT / ".github" / "workflows" / "runner-image-build-smoke.yml"
HOOK = REPO_ROOT / "docker" / "runners" / "runner-job-started.sh"
FLEET = REPO_ROOT / "config" / "runner_fleet.yaml"
SMOKE_STEP = "Assert baked Actions tool cache holds CPython in the setup-python layout"


def _dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def _arg_default(name: str) -> str:
    match = re.search(rf'^ARG {name}="?([^"\n]+)"?$', _dockerfile(), re.MULTILINE)
    assert match is not None, f"the Dockerfile lost its default for ARG {name}"
    return match.group(1)


def _compose(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", yaml.safe_load(path.read_text(encoding="utf-8")))


def test_dockerfile_builds_the_actions_tool_cache_layout_after_the_runner_user_exists() -> (
    None
):
    text = _dockerfile()
    start = text.index("${OMNI_TOOLCACHE_DIR}/Python/")
    block_start = text.rindex("\nRUN ", 0, start)
    block_end = text.index("\n\n", start)
    block = text[block_start:block_end]

    assert text.index("useradd --uid 1001") < block_start, (
        "the bake chowns to the runner user, which must exist before it"
    )
    assert ".complete" in block, "setup-python ignores a version with no marker"
    assert "-path '*/lib/python*/EXTERNALLY-MANAGED' -type f -delete" in block
    assert "|| exit 1" in block, "a failed version must fail the build"
    assert 'chown -R runner:runner "${OMNI_TOOLCACHE_DIR}"' in block
    assert "_work" not in _arg_default("OMNI_TOOLCACHE_DIR"), (
        "every pool mounts over _work, which hides baked content"
    )


def test_baked_versions_cover_every_python_version_the_workflows_ask_for() -> None:
    baked = set(_arg_default("OMNI_TOOLCACHE_PYTHON_VERSIONS").split())
    pattern = re.compile(r"python-version:\s*[\"']?(3\.\d+)")
    files = sorted(
        [
            *(REPO_ROOT / ".github" / "workflows").glob("*.yml"),
            *(REPO_ROOT / ".github" / "actions").glob("*/action.yml"),
        ]
    )
    asked = {
        minor
        for path in files
        for minor in pattern.findall(path.read_text(encoding="utf-8"))
    }

    assert asked, "no workflow asked for a python-version; the scan is broken"
    assert asked <= baked, (
        f"workflows ask for Python {sorted(asked - baked)}, the image bakes {sorted(baked)}"
    )


@pytest.mark.parametrize("compose_path", COMPOSES, ids=lambda p: p.stem)
def test_pool_opts_into_the_baked_tool_cache_with_a_named_volume_per_runner(
    compose_path: Path,
) -> None:
    cache_dir = _arg_default("OMNI_TOOLCACHE_DIR")
    compose = _compose(compose_path)
    base_env = compose["x-runner-base"]["environment"]

    assert base_env["AGENT_TOOLSDIRECTORY"] == cache_dir
    declared = compose["volumes"]
    assert compose["services"], "the pool declares no runner"
    for name, service in compose["services"].items():
        volume = f"{name}-toolcache"
        assert f"{volume}:{cache_dir}" in service["volumes"], name
        assert volume in declared, f"{volume} is not declared as a named volume"
        for entry in service["volumes"]:
            if isinstance(entry, dict):
                target = str(entry.get("target", "")).rstrip("/")
                assert target != cache_dir, f"{name}: a mount hides the baked cache"
                assert not cache_dir.startswith(target + "/"), (
                    f"{name}: a mount at {target} would hide the baked tool cache"
                )
            elif f":{cache_dir}" in entry:
                assert entry.startswith(f"{volume}:"), (
                    f"{name}: {entry} puts something other than its named volume "
                    "over the baked tool cache"
                )


def test_smoke_workflow_asserts_the_baked_tool_cache() -> None:
    text = SMOKE.read_text(encoding="utf-8")
    assert SMOKE_STEP in text, "the image smoke no longer checks the baked tool cache"
    step = text[text.index(SMOKE_STEP) :]
    step = step[: step.index("\n      # ", 1)]
    assert ".complete" in step
    assert "EXTERNALLY-MANAGED" in step


def test_hook_default_pypi_index_matches_the_fleet_declaration() -> None:
    hook = HOOK.read_text(encoding="utf-8")
    index = re.search(r'_C1_PYPI_INDEX="\$\{OMNI_PYPI_CACHE_INDEX:-(.+?)\}"', hook)
    fallback = re.search(
        r'_C1_PYPI_FALLBACK="\$\{OMNI_PYPI_CACHE_FALLBACK:-(.+?)\}"', hook
    )
    declared = yaml.safe_load(FLEET.read_text(encoding="utf-8"))["pypi_cache"]

    assert index is not None, "the hook lost its package index default"
    assert fallback is not None, "the hook lost its fallback index default"
    assert index.group(1) == declared["simple_index_url"], (
        "the hook wires a different package index than the fleet config declares"
    )
    assert fallback.group(1) == declared["fallback_index_url"]
