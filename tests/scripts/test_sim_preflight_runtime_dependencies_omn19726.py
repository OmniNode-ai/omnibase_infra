# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the private sim-preflight runtime dependency staging transform."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = (
    _REPO_ROOT
    / "scripts"
    / "runtime_build"
    / "prepare_sim_preflight_runtime_dependencies.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "prepare_sim_preflight_runtime_dependencies", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _canonical_dockerfile() -> bytes:
    return (_REPO_ROOT / "docker" / "Dockerfile.runtime").read_bytes()


def _market_lock(hash_value: str | None = None) -> str:
    sha256 = hash_value or _MODULE.PYRAGE_SHA256
    return (
        "[[package]]\n"
        'name = "pyrage"\n'
        'version = "1.4.0"\n'
        "wheels = [\n"
        '  { url = "https://files.pythonhosted.org/packages/pyrage/'
        f'{_MODULE.PYRAGE_WHEEL}", hash = "sha256:{sha256}" }},\n'
        "]\n"
    )


def _write_context(context: Path, *, lock_hash: str | None = None) -> Path:
    docker_dir = context / "docker"
    market_dir = context / "workspace" / "sibling-repos" / "omnimarket"
    docker_dir.mkdir(parents=True)
    market_dir.mkdir(parents=True)
    dockerfile = docker_dir / "Dockerfile.runtime"
    dockerfile.write_bytes(_canonical_dockerfile())
    (market_dir / "uv.lock").write_text(_market_lock(lock_hash), encoding="utf-8")
    return dockerfile


def test_recipe_preserves_active_sources_entrypoint_and_security_floors() -> None:
    canonical = _canonical_dockerfile().decode("utf-8")

    staged = _MODULE._recipe(canonical)

    for source_ref in (
        '"omnibase-core @ file:///workspace/sibling-repos/omnibase_core"',
        '"omnibase-compat @ file:///workspace/sibling-repos/omnibase_compat"',
        '"omnimarket @ file:///workspace/sibling-repos/omnimarket"',
    ):
        assert source_ref in staged
    assert "compute_workspace_provenance.py" in staged
    assert '"omninode-claude>=0.25.1,<1.0.0"' not in staged
    assert '"omninode-memory>=0.18.2,<1.0.0"' not in staged
    assert '"omninode-intelligence>=0.24.0,<1.0.0"' not in staged
    assert '    "adaptive-classifier>=0.1.2" \\' not in staged
    assert '"qdrant-client>=1.7.0,<1.17.0"' not in staged
    assert staged.count('"qdrant-client>=1.18.0,<1.20.0"') == 1

    for security_floor in (
        '"protobuf>=5.29.6"',
        '"setuptools>=78.1.1"',
        '"cryptography>=50.0.0"',
        '"aiohttp>=3.14.3"',
        '"pyasn1>=0.6.4"',
        '"transformers>=5.5.0"',
    ):
        assert security_floor in staged

    entrypoint = 'ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint-runtime.sh"]'
    assert staged.count(entrypoint) == 1
    profile_copy = staged.rfind(
        "COPY --chown=omniinfra:omniinfra docker/sim-preflight-local-profile.json"
    )
    assert profile_copy < staged.rfind(entrypoint)


def test_recipe_runs_final_strip_and_full_gate_after_all_dependency_mutations() -> None:
    staged = _MODULE._recipe(_canonical_dockerfile().decode("utf-8"))
    strip = "/app/.venv/bin/python -m omnibase_infra.runtime.strip_runtime_entry_points"
    strip_positions = [
        index for index in range(len(staged)) if staged.startswith(strip, index)
    ]
    last_install = staged.rfind("uv-with-retry pip install")
    provenance = staged.rfind(
        "/app/.venv/bin/python /workspace/compute_workspace_provenance.py"
    )
    pip_check = staged.rfind("uv pip check --python /app/.venv/bin/python")
    torch_check = staged.index("# Verify torch is CPU-only")

    assert len(strip_positions) == 2
    assert strip_positions[0] < last_install < strip_positions[1]
    assert strip_positions[1] < provenance < pip_check < torch_check
    for final_floor in (
        "m.version('omninode-memory') == '0.18.0'",
        "m.version('pyrage') == '1.4.0'",
        "Version(m.version('cryptography')) >= Version('50.0.1')",
        "Version(m.version('aiohttp')) >= Version('3.14.3')",
        "Version(m.version('pyasn1')) >= Version('0.6.4')",
        "Version(m.version('protobuf')) >= Version('5.29.6')",
        "Version(m.version('setuptools')) >= Version('78.1.1')",
        "Version(m.version('transformers')) >= Version('5.5.0')",
    ):
        assert final_floor in staged


def test_recipe_refuses_missing_or_duplicate_transformation_anchors() -> None:
    canonical = _canonical_dockerfile().decode("utf-8")
    plugin_anchor = '"omninode-claude>=0.25.1,<1.0.0"'

    with pytest.raises(ValueError, match="anchor absent or nonunique"):
        _MODULE._recipe(canonical.replace(plugin_anchor, '"changed-plugin"'))
    with pytest.raises(ValueError, match="anchor absent or nonunique"):
        _MODULE._recipe(canonical + "\n# Security: upgrade protobuf")
    with pytest.raises(ValueError, match="anchor absent or nonunique"):
        _MODULE._recipe(canonical.replace("# Security: upgrade protobuf", "# changed"))


def test_locked_pyrage_url_requires_exact_version_filename_registry_and_hash(
    tmp_path: Path,
) -> None:
    market = tmp_path / "omnimarket"
    market.mkdir()
    lock = market / "uv.lock"
    lock.write_text(_market_lock(), encoding="utf-8")

    assert _MODULE._locked_pyrage_url(market).endswith(_MODULE.PYRAGE_WHEEL)

    lock.write_text(_market_lock("0" * 64), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from approved digest"):
        _MODULE._locked_pyrage_url(market)

    lock.write_text(
        _market_lock().replace(
            "https://files.pythonhosted.org", "https://example.test"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected registry"):
        _MODULE._locked_pyrage_url(market)


def test_prepare_stages_only_hash_verified_locked_wheel_and_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel_bytes = b"synthetic test wheel bytes"
    wheel_sha256 = hashlib.sha256(wheel_bytes).hexdigest()
    monkeypatch.setattr(_MODULE, "PYRAGE_SHA256", wheel_sha256)
    context = tmp_path / "context"
    dockerfile = _write_context(context, lock_hash=wheel_sha256)

    manifest = _MODULE.prepare(context, wheel_bytes=wheel_bytes)

    assert dockerfile.read_text(encoding="utf-8") == _MODULE._recipe(
        _canonical_dockerfile().decode("utf-8")
    )
    artifacts = context / "docker" / "sim-preflight-artifacts"
    assert (artifacts / _MODULE.PYRAGE_WHEEL).read_bytes() == wheel_bytes
    assert (
        json.loads(
            (context / "docker" / "sim-preflight-local-profile.json").read_text(
                encoding="utf-8"
            )
        )
        == manifest
    )
    assert manifest["pyrage_wheel_sha256"] == wheel_sha256
    assert (
        manifest["canonical_dockerfile_sha256"] == _MODULE.CANONICAL_DOCKERFILE_SHA256
    )
    assert (
        manifest["staged_dockerfile_sha256"]
        == hashlib.sha256(dockerfile.read_bytes()).hexdigest()
    )


def test_prepare_refuses_bad_wheel_without_mutating_context(tmp_path: Path) -> None:
    context = tmp_path / "context"
    dockerfile = _write_context(context)
    original = dockerfile.read_bytes()

    with pytest.raises(ValueError, match="differs from Market lock hash"):
        _MODULE.prepare(context, wheel_bytes=b"wrong wheel bytes")

    assert dockerfile.read_bytes() == original
    assert not (context / "docker" / "sim-preflight-artifacts").exists()
    assert not (context / "docker" / "sim-preflight-local-profile.json").exists()
