# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20173: exercise the tracked-config gate and its positive control."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts" / "check_no_coding_plan_endpoint.py"
_URL = "https://api.z.ai/api/coding/paas/v4"


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
        env=scrub_git_location_env(os.environ),
    )


def _tree(root: Path, filename: str, content: str) -> None:
    subprocess.run(
        ["git", "init", "-q", str(root)],
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    subprocess.run(
        ["git", "-C", str(root), "add", "--", filename],
        check=True,
        env=scrub_git_location_env(os.environ),
    )


def test_clean_tree_passes(tmp_path: Path) -> None:
    _tree(tmp_path, "config.yml", "url: https://api.z.ai/api/paas/v4\n")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr


def test_compose_default_fails(tmp_path: Path) -> None:
    _tree(tmp_path, "docker/compose.yml", f"LLM_GLM_URL: ${{LLM_GLM_URL:-{_URL}}}\n")
    result = _run(tmp_path)
    assert result.returncode == 1, result.stderr
    assert "docker/compose.yml:1" in result.stderr


def test_yaml_comment_passes(tmp_path: Path) -> None:
    _tree(tmp_path, "config.yaml", f"# {_URL}\nurl: https://example.com/v1 # {_URL}\n")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "filename",
    [
        "config.json",
        "config.toml",
        ".env",
        ".env.local",
        "service.env.template",
        "compose.override",
    ],
)
def test_other_config_formats_fail(tmp_path: Path, filename: str) -> None:
    _tree(tmp_path, filename, f'url="{_URL}"\n')
    assert _run(tmp_path).returncode == 1


@pytest.mark.parametrize(
    "value",
    [
        "http://other.example/api/coding/",
        "api.z.ai/api/coding",
        "https://other.example/api/coding",
        "https://other.example/api/coding/paas/v4",
    ],
)
def test_nested_yaml_scalar_fails(tmp_path: Path, value: str) -> None:
    _tree(tmp_path, "config.yaml", f"outer:\n  urls:\n    - >-\n      {value}\n")
    assert _run(tmp_path).returncode == 1


def test_hostless_path_and_url_key_pass(tmp_path: Path) -> None:
    _tree(tmp_path, "config.yaml", f'"{_URL}": /api/coding/paas/v4\n')
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "filename", ["tests/fixture.yaml", "docs/example.yml", "example.md", "source.py"]
)
def test_excluded_files_pass(tmp_path: Path, filename: str) -> None:
    _tree(tmp_path, filename, f"url: {_URL}\n")
    assert _run(tmp_path).returncode == 0


def test_untracked_config_is_not_scanned(tmp_path: Path) -> None:
    _tree(tmp_path, "config.yaml", "url: /v1\n")
    (tmp_path / "scratch.yaml").write_text(f"url: {_URL}\n")
    assert _run(tmp_path).returncode == 0


def test_real_repo_tree_passes() -> None:
    result = _run(_ROOT)
    assert result.returncode == 0, result.stderr


def test_gate_is_wired_to_precommit_and_ci() -> None:
    config = yaml.safe_load((_ROOT / ".pre-commit-config.yaml").read_text())
    hooks = {hook["id"]: hook for repo in config["repos"] for hook in repo["hooks"]}
    hook = hooks["no-coding-plan-endpoint"]
    assert hook["entry"] == "uv run python scripts/check_no_coding_plan_endpoint.py"
    assert hook["pass_filenames"] is False
    assert hook["always_run"] is True
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text()
    assert hook["entry"] in workflow
    assert "tests/unit/scripts/test_check_no_coding_plan_endpoint.py" in workflow
