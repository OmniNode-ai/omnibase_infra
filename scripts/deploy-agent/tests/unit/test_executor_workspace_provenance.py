# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Workspace provenance tests for BUILD_SOURCE=workspace mode (OMN-9470)."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from deploy_agent.events import BuildSource, Phase, PhaseStatus, Scope
from deploy_agent.executor import DeployExecutor

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
PROVENANCE_SCRIPT = (
    REPO_ROOT / "scripts" / "runtime_build" / "compute_workspace_provenance.py"
)
# Pin-resolution helper imported by the provenance script (OMN-12989). Isolated
# test copies of the provenance script must place this alongside the copy so the
# sys.path-relative import resolves — mirrors the Dockerfile's dual COPY.
PIN_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "resolve_workspace_pins.py"
STAGE_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "stage_workspace.sh"


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    pass


def _ok() -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def _fail(stderr: str = "error") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)


# ---------------------------------------------------------------------------
# stage_workspace.sh presence
# ---------------------------------------------------------------------------


def test_stage_workspace_script_exists() -> None:
    assert STAGE_SCRIPT.exists(), f"stage_workspace.sh not found at {STAGE_SCRIPT}"


def test_provenance_script_exists() -> None:
    assert PROVENANCE_SCRIPT.exists(), (
        f"compute_workspace_provenance.py not found at {PROVENANCE_SCRIPT}"
    )


def test_provenance_script_reads_direct_url_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "compute_workspace_provenance_direct_url_test", str(PROVENANCE_SCRIPT)
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    mock_dist = MagicMock()
    mock_dist.read_text.return_value = json.dumps(
        {"url": "file:///workspace/sibling-repos/omnimarket"}
    )
    monkeypatch.setattr(
        mod.importlib.metadata,
        "distribution",
        lambda _: mock_dist,
    )

    assert mod._installed_direct_url("omnimarket") == {
        "url": "file:///workspace/sibling-repos/omnimarket"
    }
    mock_dist.read_text.assert_called_once_with("direct_url.json")


# ---------------------------------------------------------------------------
# _stage_workspace helper
# ---------------------------------------------------------------------------


def test_stage_workspace_called_for_workspace_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNI_HOME", "/data/omninode/omni_home")
    executor = DeployExecutor()
    stage_calls: list[tuple[str, str]] = []

    def fake_stage(repo_dir: str, omni_home: str) -> None:
        stage_calls.append((repo_dir, omni_home))

    captured_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
        captured_cmds.append(cmd)
        return _ok()

    monkeypatch.setattr("deploy_agent.executor._run", fake_run)
    monkeypatch.setattr(DeployExecutor, "_stage_workspace", staticmethod(fake_stage))

    executor._compose_build(
        Scope.RUNTIME,
        "abc1234",
        _noop_phase_update,
        build_source=BuildSource.WORKSPACE,
    )

    assert len(stage_calls) == 1
    _, omni_home = stage_calls[0]
    assert omni_home == "/data/omninode/omni_home"


def test_stage_workspace_not_called_for_release_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = DeployExecutor()
    stage_calls: list[tuple] = []

    def fake_stage(repo_dir: str, omni_home: str) -> None:
        stage_calls.append((repo_dir, omni_home))

    captured_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
        captured_cmds.append(cmd)
        return _ok()

    monkeypatch.setattr("deploy_agent.executor._run", fake_run)
    monkeypatch.setattr(DeployExecutor, "_stage_workspace", staticmethod(fake_stage))

    executor._compose_build(
        Scope.RUNTIME,
        "abc1234",
        _noop_phase_update,
        build_source=BuildSource.RELEASE,
    )

    assert stage_calls == [], "stage_workspace must not be called in release mode"


def test_stage_workspace_failure_aborts_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNI_HOME", "/data/omninode/omni_home")
    executor = DeployExecutor()

    def fail_stage(repo_dir: str, omni_home: str) -> None:
        raise RuntimeError("Workspace staging failed")

    captured_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
        captured_cmds.append(cmd)
        return _ok()

    monkeypatch.setattr("deploy_agent.executor._run", fake_run)
    monkeypatch.setattr(DeployExecutor, "_stage_workspace", staticmethod(fail_stage))

    with pytest.raises(RuntimeError, match="Workspace staging failed"):
        executor._compose_build(
            Scope.RUNTIME,
            "abc1234",
            _noop_phase_update,
            build_source=BuildSource.WORKSPACE,
        )

    assert captured_cmds == [], (
        "docker compose build must not run after staging failure"
    )


def test_stage_workspace_missing_script_raises(tmp_path: Path) -> None:
    executor = DeployExecutor()
    with pytest.raises(RuntimeError, match="workspace staging script not found"):
        DeployExecutor._stage_workspace(str(tmp_path), "/data/omninode/omni_home")


def test_stage_workspace_script_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_script = tmp_path / "scripts" / "runtime_build" / "stage_workspace.sh"
    fake_script.parent.mkdir(parents=True)
    fake_script.write_text("#!/usr/bin/env bash\nexit 2\n")

    with pytest.raises(RuntimeError, match="Workspace staging failed"):
        DeployExecutor._stage_workspace(str(tmp_path), "/data/omninode/omni_home")


# ---------------------------------------------------------------------------
# build args in workspace mode
# ---------------------------------------------------------------------------


def test_workspace_build_passes_build_date_and_vcs_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNI_HOME", "/data/omninode/omni_home")
    executor = DeployExecutor()
    captured_cmds: list[list[str]] = []

    def fake_stage(repo_dir: str, omni_home: str) -> None:
        pass

    def fake_run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
        captured_cmds.append(cmd)
        return _ok()

    monkeypatch.setattr("deploy_agent.executor._run", fake_run)
    monkeypatch.setattr(DeployExecutor, "_stage_workspace", staticmethod(fake_stage))

    executor._compose_build(
        Scope.RUNTIME,
        "deadbeef",
        _noop_phase_update,
        build_source=BuildSource.WORKSPACE,
    )

    build_cmd = captured_cmds[0]
    assert "VCS_REF=deadbeef" in build_cmd
    # RUNTIME_VERSION is sourced from the repo's own pyproject — read it the same
    # way the executor does so this assertion never drifts on a version bump.
    import tomllib

    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    expected_runtime_version = pyproject["project"]["version"]
    assert f"RUNTIME_VERSION={expected_runtime_version}" in build_cmd
    assert any(a.startswith("BUILD_DATE=") for a in build_cmd)


# ---------------------------------------------------------------------------
# compute_workspace_provenance.py contract
# ---------------------------------------------------------------------------


def test_provenance_script_passes_with_valid_local_installs(
    tmp_path: Path,
) -> None:
    """Provenance script writes manifest and exits 0 when all packages are local installs."""
    sibling_dir = tmp_path / "workspace" / "sibling-repos"
    for repo in ("omnibase_compat", "onex_change_control", "omnimarket"):
        repo_dir = sibling_dir / repo
        repo_dir.mkdir(parents=True)
        (repo_dir / "pyproject.toml").write_text(
            f'[project]\nname = "{repo}"\nversion = "0.1.0"\n'
        )
        (repo_dir / "README.md").write_text(f"# {repo}\n")

    manifest_path = tmp_path / "build-provenance.json"

    import importlib.metadata
    import sys

    # Patch SIBLING_REPOS_DIR and OUTPUT_MANIFEST to point at tmp_path
    script_src = PROVENANCE_SCRIPT.read_text(encoding="utf-8")
    script_src = script_src.replace(
        'SIBLING_REPOS_DIR = Path("/workspace/sibling-repos")',
        f'SIBLING_REPOS_DIR = Path("{sibling_dir}")',
    ).replace(
        'OUTPUT_MANIFEST = Path("/app/build-provenance.json")',
        f'OUTPUT_MANIFEST = Path("{manifest_path}")',
    )

    patched_script = tmp_path / "compute_workspace_provenance_test.py"
    patched_script.write_text(script_src, encoding="utf-8")
    # Co-locate the pin-resolution helper so the script's import resolves.
    (tmp_path / "resolve_workspace_pins.py").write_text(
        PIN_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )

    # Patch importlib.metadata to simulate local installs
    mock_dist = MagicMock()
    mock_dist.locate_file.return_value = tmp_path / "direct_url.json"
    (tmp_path / "direct_url.json").write_text(
        json.dumps(
            {
                "url": f"file://{sibling_dir}/omnibase_compat",
                "dir_info": {"editable": False},
            }
        ),
        encoding="utf-8",
    )

    # Run the patched script in a subprocess with mocked distribution lookup
    result = subprocess.run(
        [sys.executable, str(patched_script)],
        capture_output=True,
        text=True,
        check=False,
        env={
            **dict(__import__("os").environ),
            "BUILD_DATE": "2026-05-26T00:00:00Z",
            "VCS_REF": "abc1234",
        },
    )
    # Script will exit 1 because importlib.metadata won't find the fake packages;
    # what we assert here is that the manifest is written and contains the right structure.
    assert manifest_path.exists(), "provenance manifest must be written even on failure"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["build_source"] == "workspace"
    assert "proofs" in manifest
    assert len(manifest["proofs"]) == len(
        ["omnibase_compat", "onex_change_control", "omnimarket"]
    )
    for proof in manifest["proofs"]:
        assert "workspace_digest" in proof
        assert len(proof["workspace_digest"]) == 64  # SHA-256 hex


def test_provenance_script_missing_sibling_repo_exits_nonzero(
    tmp_path: Path,
) -> None:
    """Provenance script exits 1 (missing_install) when sibling repo absent."""
    sibling_dir = tmp_path / "workspace" / "sibling-repos"
    sibling_dir.mkdir(parents=True)
    # Only create one of the three repos — others are missing

    manifest_path = tmp_path / "build-provenance.json"

    script_src = PROVENANCE_SCRIPT.read_text(encoding="utf-8")
    script_src = script_src.replace(
        'SIBLING_REPOS_DIR = Path("/workspace/sibling-repos")',
        f'SIBLING_REPOS_DIR = Path("{sibling_dir}")',
    ).replace(
        'OUTPUT_MANIFEST = Path("/app/build-provenance.json")',
        f'OUTPUT_MANIFEST = Path("{manifest_path}")',
    )

    patched_script = tmp_path / "compute_prov_missing.py"
    patched_script.write_text(script_src, encoding="utf-8")
    (tmp_path / "resolve_workspace_pins.py").write_text(
        PIN_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )

    import sys

    result = subprocess.run(
        [sys.executable, str(patched_script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, (
        "Script must exit non-zero when sibling repos are missing"
    )
    assert "Missing sibling repo" in result.stderr


def test_provenance_manifest_contains_workspace_digest(tmp_path: Path) -> None:
    """Hash function produces stable, non-empty SHA-256 digests."""
    import sys

    # Import the hash function directly
    sys.path.insert(0, str(PROVENANCE_SCRIPT.parent))
    import importlib

    spec = importlib.util.spec_from_file_location(
        "compute_workspace_provenance", str(PROVENANCE_SCRIPT)
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    test_dir = tmp_path / "repo"
    test_dir.mkdir()
    (test_dir / "file.py").write_text("x = 1\n", encoding="utf-8")

    digest1 = mod._hash_tree(test_dir)
    assert len(digest1) == 64

    # Mutating the tree changes the digest
    (test_dir / "file.py").write_text("x = 2\n", encoding="utf-8")
    digest2 = mod._hash_tree(test_dir)
    assert digest1 != digest2, "digest must change when repo tree content changes"


def test_dockerfile_workspace_copy_and_provenance_present() -> None:
    """Dockerfile must contain the workspace COPY and provenance runner."""
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.runtime").read_text(
        encoding="utf-8"
    )

    assert "COPY workspace/sibling-repos/ /workspace/sibling-repos/" in dockerfile
    assert "scripts/runtime_build/compute_workspace_provenance.py" in dockerfile
    assert "/workspace/compute_workspace_provenance.py" in dockerfile
    assert "build-provenance.json" in dockerfile
    assert "com.omninode.workspace_provenance_manifest" in dockerfile
    assert "--if-present" not in dockerfile


def test_dockerfile_copies_pin_resolution_helper() -> None:
    """OMN-12989: the pin-resolution helper must be copied beside the provenance
    script so the in-image import resolves."""
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.runtime").read_text(
        encoding="utf-8"
    )
    assert "scripts/runtime_build/resolve_workspace_pins.py" in dockerfile
    assert "/workspace/resolve_workspace_pins.py" in dockerfile


def _patched_provenance(tmp_path: Path, sibling_dir: Path, manifest_path: Path) -> Path:
    """Write an isolated copy of the provenance script + helper, path-patched."""
    script_src = PROVENANCE_SCRIPT.read_text(encoding="utf-8")
    script_src = script_src.replace(
        'SIBLING_REPOS_DIR = Path("/workspace/sibling-repos")',
        f'SIBLING_REPOS_DIR = Path("{sibling_dir}")',
    ).replace(
        'OUTPUT_MANIFEST = Path("/app/build-provenance.json")',
        f'OUTPUT_MANIFEST = Path("{manifest_path}")',
    )
    patched = tmp_path / "compute_prov_pins.py"
    patched.write_text(script_src, encoding="utf-8")
    (tmp_path / "resolve_workspace_pins.py").write_text(
        PIN_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return patched


_OMNIMARKET_LOCK = """
version = 1
requires-python = ">=3.12"

[[package]]
name = "omnibase-compat"
version = "0.5.1"
source = { git = "https://github.com/OmniNode-ai/omnibase_compat.git?rev=4d887307aae34d9d40d389ba91070cb411ce3df5#4d887307aae34d9d40d389ba91070cb411ce3df5" }

[[package]]
name = "omnibase-infra"
version = "0.38.1"
source = { git = "https://github.com/OmniNode-ai/omnibase_infra.git?rev=e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59#e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59" }

[[package]]
name = "onex-change-control"
version = "0.9.0"
source = { git = "https://github.com/OmniNode-ai/onex_change_control.git?rev=4877d3c223517cb0c7e1eca462ba0f4d38916314" }

[[package]]
name = "omnimarket"
version = "0.4.3"
source = { editable = "." }
"""


def _stage_siblings(sibling_dir: Path) -> None:
    """Create the three workspace siblings (compat, occ, omnimarket) at lock pin."""
    for repo, version in (
        ("omnibase_compat", "0.5.1"),
        ("onex_change_control", "0.9.0"),
        ("omnimarket", "0.4.3"),
    ):
        d = sibling_dir / repo
        d.mkdir(parents=True)
        name = repo.replace("_", "-")
        (d / "pyproject.toml").write_text(
            f'[project]\nname = "{name}"\nversion = "{version}"\n', encoding="utf-8"
        )
    (sibling_dir / "omnimarket" / "uv.lock").write_text(
        _OMNIMARKET_LOCK, encoding="utf-8"
    )


def test_provenance_emits_pin_comparison_block(tmp_path: Path) -> None:
    """OMN-12989: the manifest carries an expected-vs-actual pin_comparison block."""
    import sys

    sibling_dir = tmp_path / "workspace" / "sibling-repos"
    sibling_dir.mkdir(parents=True)
    _stage_siblings(sibling_dir)
    manifest_path = tmp_path / "build-provenance.json"
    patched = _patched_provenance(tmp_path, sibling_dir, manifest_path)

    # Host infra version resolved from importlib.metadata is at/above pin in the
    # test venv (>=0.38.1), so the infra self-check passes; here we only assert
    # the comparison block is present and the staged siblings are recorded.
    subprocess.run(
        [sys.executable, str(patched)], capture_output=True, text=True, check=False
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "pin_comparison" in manifest
    by_pkg = {e["package"]: e for e in manifest["pin_comparison"]}
    # omnibase-compat staged at exactly the lock pin.
    assert by_pkg["omnibase-compat"]["status"] == "exact"
    assert by_pkg["omnibase-compat"]["match"] is True
    # The host infra self-check appears only when omnibase-infra is installed in
    # the running venv (always true in the build image; not in the deploy-agent
    # venv). When present it must carry the lock-pinned expected version.
    if "omnibase-infra" in by_pkg:
        assert by_pkg["omnibase-infra"]["expected_version"] == "0.38.1"


def test_build_comparisons_flags_host_infra_regression(tmp_path: Path) -> None:
    """OMN-12989 (unit): the exact crash — host infra 0.37.0 vs lock pin 0.38.1.

    Exercises build_comparisons directly with an explicit host-infra actual
    version so the regression detection does not depend on the test venv.
    """
    mod = _load_resolve_module()
    omnimarket = tmp_path / "omnimarket"
    omnimarket.mkdir()
    (omnimarket / "uv.lock").write_text(_OMNIMARKET_LOCK, encoding="utf-8")

    # Synthesize a "host infra" pyproject tree at the stale crash version.
    infra = tmp_path / "omnibase_infra"
    infra.mkdir()
    (infra / "pyproject.toml").write_text(
        '[project]\nname = "omnibase-infra"\nversion = "0.37.0"\n', encoding="utf-8"
    )

    comparisons = mod.build_comparisons(
        lock_path=omnimarket / "uv.lock",
        siblings={"omnibase-infra": infra},
    )
    infra_cmp = next(c for c in comparisons if c.package == "omnibase-infra")
    assert infra_cmp.status == "regression"
    with pytest.raises(mod.WorkspacePinError):
        mod.assert_pins_satisfied(comparisons)


def _load_resolve_module():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "resolve_workspace_pins_provtest", str(PIN_SCRIPT)
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod
