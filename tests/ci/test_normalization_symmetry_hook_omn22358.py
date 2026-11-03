# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-22358: the normalization-symmetry hook is core's hook id and runs at pre-commit.

The repository used to declare a local ``normalization-symmetry`` hook whose entry
named ``omnibase_core.validators.normalization-symmetry``. A hyphen cannot be a
module name, the module never existed, and the hook was manual-stage and listed in
the pre-commit.ci ``skip:`` list, so it never ran and never failed. The hook now
comes from the omnibase_core hook repository by id, at the pre-commit stage, and
is not skipped. The canary runs the check that hook runs over a planted tree.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / ".pre-commit-config.yaml"
CORE_REPO = "https://github.com/OmniNode-ai/omnibase_core"
HOOK_ID = "normalization-symmetry"
CHECK_MODULE = (
    "omnibase_core.nodes.node_normalization_symmetry_check_compute"
    ".runtime_normalization_symmetry_check"
)
HYPHENATED_VALIDATOR_MODULE = re.compile(r"omnibase_core\.validators\.[\w.]*-")

ASYMMETRIC_PRODUCER = """\
from omnibase_core.topics import TopicBase


class _Producer:
    def emit(self) -> None:
        topic = TopicBase.SESSION_STARTED.strip()
        self._bus.send(topic, {"key": "value"})
"""
ASYMMETRIC_CONSUMER = """\
from omnibase_core.topics import TopicBase


class _Consumer:
    def register(self) -> None:
        self._bus.subscribe([TopicBase.SESSION_STARTED])
"""
SYMMETRIC_CONSUMER = """\
from omnibase_core.topics import TopicBase


class _Consumer:
    def register(self) -> None:
        self._bus.subscribe([TopicBase.SESSION_STARTED.strip()])
"""


def _config() -> dict[str, object]:
    loaded = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _hooks() -> list[tuple[str, dict[str, object]]]:
    repos = _config()["repos"]
    assert isinstance(repos, list)
    found: list[tuple[str, dict[str, object]]] = []
    for repo in repos:
        for hook in repo.get("hooks") or []:
            found.append((str(repo["repo"]), hook))
    return found


def _declarations() -> list[tuple[str, dict[str, object]]]:
    return [(repo, hook) for repo, hook in _hooks() if hook.get("id") == HOOK_ID]


def test_normalization_symmetry_hook_names_no_hyphenated_validator_module() -> None:
    offenders = [
        str(hook.get("entry", ""))
        for _, hook in _hooks()
        if HYPHENATED_VALIDATOR_MODULE.search(str(hook.get("entry", "")))
    ]
    assert offenders == [], f"entry names a module that cannot exist: {offenders}"


def test_normalization_symmetry_hook_is_core_hook_id_with_no_local_entry() -> None:
    declarations = _declarations()
    assert [repo for repo, _ in declarations] == [CORE_REPO], (
        f"{HOOK_ID} must be declared once, under {CORE_REPO}: {declarations}"
    )
    hook = declarations[0][1]
    for key in ("entry", "language", "files", "exclude", "pass_filenames"):
        assert key not in hook, f"core's {key} must not be overridden: {hook}"


def test_normalization_symmetry_hook_is_not_manual_only() -> None:
    for _, hook in _declarations():
        stages = hook.get("stages")
        assert stages is None or (
            isinstance(stages, list) and "pre-commit" in stages
        ), f"{HOOK_ID} must run at pre-commit, got stages={stages}"


def test_normalization_symmetry_hook_is_not_skipped_in_pre_commit_ci() -> None:
    ci = _config().get("ci")
    assert isinstance(ci, dict)
    assert HOOK_ID not in (ci.get("skip") or [])


def _run_check(tree: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", CHECK_MODULE, str(tree)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_normalization_symmetry_hook_check_refuses_a_canary_asymmetry(
    tmp_path: Path,
) -> None:
    (tmp_path / "producer.py").write_text(ASYMMETRIC_PRODUCER, encoding="utf-8")
    (tmp_path / "consumer.py").write_text(ASYMMETRIC_CONSUMER, encoding="utf-8")
    result = _run_check(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stderr.strip()


def test_normalization_symmetry_hook_check_passes_a_symmetric_tree(
    tmp_path: Path,
) -> None:
    """Positive control: the canary above fails because of the asymmetry alone."""
    (tmp_path / "producer.py").write_text(ASYMMETRIC_PRODUCER, encoding="utf-8")
    (tmp_path / "consumer.py").write_text(SYMMETRIC_CONSUMER, encoding="utf-8")
    result = _run_check(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
