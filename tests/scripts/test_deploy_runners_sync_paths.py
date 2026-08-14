# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""deploy-runners.sh's SYNC_PATHS + rsync_artifacts() must actually sync every
file docker/runners/Dockerfile COPYs (OMN-15142).

Discovered while rebuilding + force-recreating the omninode-pc runner fleet for
the OMN-14900 N+5 stability deploy-hop iteration: ``docker/runners/Dockerfile``
does ``COPY omni-curl /usr/local/bin/omni-curl``, but ``SYNC_PATHS`` never
listed ``docker/runners/omni-curl`` (or its source ``omni-curl.sh``), so a
rebuild against a fresh/empty deployment dir failed at the COPY layer:

    ERROR: failed to build: failed to solve: failed to compute cache key:
    failed to calculate checksum of ref ...: "/omni-curl": not found

Root cause is a two-list drift hazard, not just a missing entry:
``SYNC_PATHS`` is a declared bash array, but ``rsync_artifacts()`` does NOT
loop over it -- it hardcodes an independent, explicit filename list per
``rsync`` invocation. The two lists can silently diverge (as they did here).
These tests hold both lists to the Dockerfile's real COPY manifest and prove
they agree with each other, using ``scripts/ci/check_runner_host_artifact_freshness
.parse_sync_paths`` (the same extractor OMN-15114 already uses to avoid a
hand-maintained third copy) plus a real bash execution of the extracted
``rsync_artifacts()`` function against a stubbed ``rsync``/``ssh``, so a
regression is caught by executing the actual script logic, not by grepping
source text for a string.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runners.sh"
DOCKERFILE = REPO_ROOT / "docker" / "runners" / "Dockerfile"
SCRIPTS_CI = REPO_ROOT / "scripts" / "ci"
FRESHNESS_SCRIPT = SCRIPTS_CI / "check_runner_host_artifact_freshness.py"

# The two source files the Dockerfile COPYs that must round-trip through the
# rsync pipeline for a build against a fresh deployment dir to succeed.
_REQUIRED_OMNI_CURL_PATHS = (
    "docker/runners/omni-curl",
    "docker/runners/omni-curl.sh",
)


def _load_freshness_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "check_runner_host_artifact_freshness", FRESHNESS_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS_CI))
    try:
        spec.loader.exec_module(module)
    finally:
        if str(SCRIPTS_CI) in sys.path:
            sys.path.remove(str(SCRIPTS_CI))
    return module


def _script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _extract_function(name: str) -> str:
    """Return the source text of a single top-level bash function ``name()``."""
    text = _script_text()
    match = re.search(
        rf"^{re.escape(name)}\s*\(\)\s*\{{.*?\n\}}",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, (
        f"could not extract function {name}() from deploy-runners.sh"
    )
    return match.group(0)


def test_dockerfile_actually_copies_omni_curl() -> None:
    """Sanity check the fixture assumption: the Dockerfile really COPYs it."""
    dockerfile_text = DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY omni-curl /usr/local/bin/omni-curl" in dockerfile_text


@pytest.mark.parametrize("required_path", _REQUIRED_OMNI_CURL_PATHS)
def test_sync_paths_includes_omni_curl_artifacts(required_path: str) -> None:
    """SYNC_PATHS (the declared array) must list both omni-curl files."""
    module = _load_freshness_module()
    sync_paths = module.parse_sync_paths(_script_text())
    assert required_path in sync_paths, (
        f"SYNC_PATHS is missing {required_path!r} -- a build against a fresh "
        "deployment dir will fail at the Dockerfile COPY layer"
    )


def _run_rsync_artifacts() -> list[str]:
    """Execute the real ``rsync_artifacts()`` function from deploy-runners.sh
    against a stubbed ``rsync``/``ssh`` on PATH, and return the flat list of
    every argument passed to every ``rsync`` invocation.

    This proves the SECOND list (the hardcoded per-call rsync arguments,
    independent of SYNC_PATHS) actually syncs the omni-curl files too -- the
    two lists drifting apart is the exact defect class this ticket found.
    """
    harness = "\n".join(
        [
            "set -euo pipefail",
            f'REPO_ROOT="{REPO_ROOT}"',
            f'RUNNER_FLEET_CONFIG="{REPO_ROOT}/config/runner_fleet.yaml"',
            'RUNNER_HOST="dummy-host"',
            'RUNNER_HOST_DIR="/dummy/runners"',
            "DRY_RUN=false",
            "log() { :; }",
            "run_ssh() { :; }",  # mkdir -p call inside rsync_artifacts
            _extract_function("rsync_artifacts"),
            "rsync_artifacts",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="rsync-stub-") as stub_dir_name:
        stub_dir = Path(stub_dir_name)
        stub_rsync = stub_dir / "rsync"
        stub_rsync.write_text(
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n", encoding="utf-8"
        )
        stub_rsync.chmod(0o755)
        result = subprocess.run(
            ["bash", "-c", harness],
            capture_output=True,
            text=True,
            check=False,
            env={
                "PATH": f"{stub_dir}:/usr/bin:/bin",
            },
        )
    assert result.returncode == 0, (
        f"rsync_artifacts() harness failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    return result.stdout.splitlines()


@pytest.mark.parametrize("required_path", _REQUIRED_OMNI_CURL_PATHS)
def test_rsync_artifacts_actually_syncs_omni_curl(required_path: str) -> None:
    """The real rsync_artifacts() function must pass both omni-curl source
    paths to an rsync invocation -- not merely list them in SYNC_PATHS."""
    argv_lines = _run_rsync_artifacts()
    full_path = str(REPO_ROOT / required_path)
    assert any(full_path in line for line in argv_lines), (
        f"rsync_artifacts() never passed {full_path} to any rsync invocation "
        f"(argv captured: {argv_lines!r})"
    )
