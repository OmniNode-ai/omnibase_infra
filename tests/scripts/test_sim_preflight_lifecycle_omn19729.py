# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Static safety contracts for the disposable sim lifecycle entrypoint."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from scripts.runtime_build.prepare_sim_preflight_workspace import prepare, tree_digest
from scripts.runtime_build.verify_sim_preflight_image_provenance import (
    verify as verify_image,
)
from scripts.runtime_build.verify_sim_preflight_runtime_source_pins import verify

ROOT = Path(__file__).resolve().parents[2]
START = ROOT / "scripts" / "runtime_build" / "start_sim_preflight_clone.sh"
PINS = ROOT / "docker" / "sim-preflight-runtime-source-pins.json"
BUILDER = ROOT / "scripts" / "runtime_build" / "build_sim_preflight_runtime_image.sh"


def test_start_refuses_to_bypass_the_two_preflight_verifiers() -> None:
    raw = START.read_text(encoding="utf-8")
    assert 'readonly PROJECT="omnibase-infra-sim-preflight"' in raw
    assert "verify_sim_preflight_migration_profile.py" in raw
    assert "verify_sim_preflight_runtime_source_pins.py" in raw
    assert "--migrations-dir" in raw
    assert "SIM_PREFLIGHT_ENV_FILE must have mode 600" in raw
    assert "refusing to restart an existing sim-preflight project" in raw
    assert "ps -a -q" in raw
    assert "refusing to reuse existing sim-preflight volumes" in raw
    assert " up -d --no-build" in raw
    assert "--force-recreate" not in raw
    assert " down " not in raw


def test_graph_overlay_is_optional_and_requires_private_mount_inputs() -> None:
    raw = START.read_text(encoding="utf-8")
    assert 'case "${SIM_PREFLIGHT_GRAPH_OVERLAY:-false}" in' in raw
    assert "SIM_PREFLIGHT_GRAPH_RUNTIME_CONFIG_FILE" in raw
    assert "SIM_PREFLIGHT_GRAPH_GATEWAY_KEYMAP_FILE" in raw
    assert "SIM_PREFLIGHT_GRAPH_TERMINAL_PRIVATE_KEY_FILE" in raw
    assert (
        'compose_files+=(-f "${ROOT}/docker/docker-compose.sim-preflight-graph.yml")'
        in raw
    )
    assert '"${compose_files[@]}"' in raw


def test_isolated_first_launch_stages_auth_before_gateway_or_runtime() -> None:
    raw = START.read_text(encoding="utf-8")
    assert 'case "${SIM_PREFLIGHT_ISOLATED_AUTH:-false}" in' in raw
    assert "scripts.runtime_build.verify_sim_preflight_isolation" in raw
    assert "docker-compose.sim-preflight-isolated.yml" in raw
    assert "docker-compose.sim-preflight-auth.yml" in raw
    assert "start_services=(keycloak cloud-migration redpanda valkey)" in raw
    assert 'up -d --no-build "${start_services[@]}"' in raw


def test_start_rejects_invalid_rendered_compose_without_up_or_secret_output(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "docker-calls.log"
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'printf \'%s\\n\' "$*" >> "$CALL_LOG"\n'
        "if [[ \"$*\" == *'config --format json'* ]]; then\n"
        "  printf '%s\\n' '{\"sensitive_value\":\"rendered-secret-marker\"}'\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'if [[ "$*" == *summarize_sim_preflight_compose.py* ]]; then\n'
        "  cat >/dev/null\n"
        "  printf '%s\\n' 'validator-secret-marker' >&2\n"
        "  exit 1\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    private_env = tmp_path / "private.env"
    private_env.write_text("TEST_VALUE=private\n", encoding="utf-8")
    private_env.chmod(0o600)
    env = os.environ.copy()
    env.pop("COMPOSE_PROJECT_NAME", None)
    env.update(
        {
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
            "CALL_LOG": str(call_log),
            "SIM_PREFLIGHT_ENV_FILE": str(private_env),
            "SIM_202_RUNTIME_IMAGE": "sim:test",
            "SIM_PREFLIGHT_IMAGE_PROVENANCE_PATH": str(
                tmp_path / "unused-provenance.json"
            ),
            "SIM_PREFLIGHT_GRAPH_OVERLAY": "false",
        }
    )

    result = subprocess.run(
        [str(START)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert result.returncode == 65
    assert "sim-preflight compose config validation failed" in result.stderr
    assert "rendered-secret-marker" not in result.stdout + result.stderr
    assert "validator-secret-marker" not in result.stdout + result.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert any("config --format json" in call for call in calls)
    assert not any("up" in call.split() for call in calls)
    assert not any("ps" in call.split() for call in calls)


def test_runtime_source_pins_are_exact_and_fail_closed(tmp_path: Path) -> None:
    assert verify(PINS) == {
        "omnibase_core": "1afbad2135ae5ed2c26d36864ec1a15de99b46d5",
        "omnibase_infra": "f9366cb9d13ef930af5625433b0537083b4efd0a",
    }
    malformed = tmp_path / "pins.json"
    malformed.write_text(
        json.dumps(
            {
                "profile": "sim-preflight-runtime-source-pins-v1",
                "sources": {"omnibase_core": "not-a-revision"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="core and infra"):
        verify(malformed)


def test_builder_archives_sources_without_mutating_any_worktree() -> None:
    raw = BUILDER.read_text(encoding="utf-8")
    assert 'git -C "$source" archive "$head"' in raw
    assert "prepare_sim_preflight_workspace.py" in raw
    assert "git checkout" not in raw
    assert "git reset" not in raw
    assert "git clean" not in raw
    assert "ln -s" not in raw
    assert "io.omninode.sim-preflight.infra_snapshot_sha256" in raw
    assert "io.omninode.sim-preflight.core_snapshot_sha256" in raw


def test_snapshot_digest_changes_with_bytes_and_ignores_generated_workspace(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "runtime.py"
    source.write_text("one", encoding="utf-8")
    first = tree_digest(tmp_path, exclude_workspace=True)
    source.write_text("two", encoding="utf-8")
    assert tree_digest(tmp_path, exclude_workspace=True) != first
    second = tree_digest(tmp_path, exclude_workspace=True)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "metadata.json").write_text("generated")
    assert tree_digest(tmp_path, exclude_workspace=True) == second


def test_image_provenance_rejects_tampered_image_id_and_labels(tmp_path: Path) -> None:
    head = "a" * 40
    digest = "b" * 64
    raw = {
        "profile": "sim-preflight-image-provenance-v2",
        "dependency_profile": "sim-preflight-runtime-active-base-v1",
        "image_ref": "sim:test",
        "image_id": "sha256:" + "c" * 64,
        **{
            name: head if name.endswith("head") else digest
            for name in (
                "infra_head",
                "infra_snapshot_sha256",
                "core_head",
                "core_snapshot_sha256",
                "market_head",
                "market_snapshot_sha256",
                "compat_head",
                "compat_snapshot_sha256",
                "canonical_dockerfile_sha256",
                "staged_dockerfile_sha256",
                "pyrage_wheel_sha256",
            )
        },
    }
    provenance = tmp_path / "provenance.json"
    pins = tmp_path / "pins.json"
    provenance.write_text(json.dumps(raw), encoding="utf-8")
    pins.write_text(
        json.dumps({"sources": {"omnibase_core": head, "omnibase_infra": head}})
    )
    labels = {
        "io.omninode.sim-preflight." + key: value
        for key, value in raw.items()
        if key.endswith(("head", "sha256"))
    }
    labels.update(
        {
            "io.omninode.sim-preflight.dependency_profile": raw["dependency_profile"],
            "com.omninode.build_source": "workspace",
            "com.omninode.promotion_class": "stability-candidate",
            "com.omninode.non_main_lineage": "true",
        }
    )
    image = {"Id": raw["image_id"], "Config": {"Labels": labels}}
    with patch("subprocess.run") as run:
        run.return_value.stdout = json.dumps([image])
        verify_image(provenance, "sim:test", pins)
        image["Id"] = "sha256:" + "d" * 64
        run.return_value.stdout = json.dumps([image])
        with pytest.raises(ValueError, match="image id"):
            verify_image(provenance, "sim:test", pins)
        image["Id"] = raw["image_id"]
        labels["io.omninode.sim-preflight.market_snapshot_sha256"] = "e" * 64
        run.return_value.stdout = json.dumps([image])
        with pytest.raises(ValueError, match="labels"):
            verify_image(provenance, "sim:test", pins)
        labels["io.omninode.sim-preflight.market_snapshot_sha256"] = digest
        labels["io.omninode.sim-preflight.staged_dockerfile_sha256"] = "e" * 64
        run.return_value.stdout = json.dumps([image])
        with pytest.raises(ValueError, match="labels"):
            verify_image(provenance, "sim:test", pins)
        labels["io.omninode.sim-preflight.staged_dockerfile_sha256"] = digest
        labels["com.omninode.non_main_lineage"] = "false"
        run.return_value.stdout = json.dumps([image])
        with pytest.raises(ValueError, match="non-production"):
            verify_image(provenance, "sim:test", pins)


def test_prepare_uses_canonical_pin_comparison_without_touching_sources(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    context = tmp_path / "context"
    staged = context / "workspace" / "sibling-repos"
    staged.mkdir(parents=True)
    names = (
        "omnibase_infra",
        "omnibase_core",
        "omnibase_spi",
        "omnibase_compat",
        "omnimarket",
    )
    lock = "".join(
        f'[[package]]\nname = "{name.replace("_", "-")}"\nversion = "1.0.0"\n'
        for name in names
    )
    heads: dict[str, str] = {}
    for name in names:
        repo = source_root / name
        repo.mkdir()
        (repo / "pyproject.toml").write_text(
            f'[project]\nname = "{name.replace("_", "-")}"\nversion = "1.0.0"\n',
            encoding="utf-8",
        )
        if name == "omnimarket":
            (repo / "uv.lock").write_text(lock, encoding="utf-8")
        subprocess.run(
            ["git", "init", "-q", str(repo)],
            check=True,
            env=scrub_git_location_env(os.environ),
        )
        subprocess.run(
            ["git", "-C", str(repo), "add", "."],
            check=True,
            env=scrub_git_location_env(os.environ),
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.test",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "source",
            ],
            check=True,
            env=scrub_git_location_env(os.environ),
        )
        heads[name] = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            env=scrub_git_location_env(os.environ),
        ).stdout.strip()
        if name in ("omnibase_core", "omnibase_compat", "omnimarket"):
            shutil.copytree(repo, staged / name, ignore=shutil.ignore_patterns(".git"))
    (source_root / "omnimarket" / "tests").mkdir()
    (source_root / "omnimarket" / "tests" / "pending.py").write_text("assert True\n")
    shutil.copy(
        source_root / "omnibase_infra" / "pyproject.toml", context / "pyproject.toml"
    )
    before = {
        name: subprocess.run(
            ["git", "-C", str(source_root / name), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            env=scrub_git_location_env(os.environ),
        ).stdout
        for name in names
    }
    digests = prepare(
        context,
        source_root,
        source_root / "omnibase_infra",
        source_root / "omnimarket",
        source_root / "omnibase_core",
        heads,
    )
    comparison = json.loads(
        (context / "workspace" / "sibling-pin-comparison.json").read_text()
    )
    assert comparison["drift_count"] == 0
    assert len(comparison["comparisons"]) == 5
    assert all(
        row["actual_git_sha"] == heads[row["package"].replace("-", "_")]
        for row in comparison["comparisons"]
    )
    assert len(digests["core_snapshot_sha256"]) == 64
    for name in names:
        after = subprocess.run(
            ["git", "-C", str(source_root / name), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            env=scrub_git_location_env(os.environ),
        ).stdout
        assert after == before[name]
    (source_root / "omnimarket" / "src").mkdir()
    (source_root / "omnimarket" / "src" / "pending.py").write_text(
        "runtime_change = True\n"
    )
    with pytest.raises(ValueError, match="uncommitted non-test files"):
        prepare(
            context,
            source_root,
            source_root / "omnibase_infra",
            source_root / "omnimarket",
            source_root / "omnibase_core",
            heads,
        )
