# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Verdict parity: scripts/validate_no_env_fallbacks.py vs the core check node (OMN-20568).

The infra script and ``omnibase_core.nodes.node_no_env_fallbacks_check_compute``
must give the same exit code and the same finding set ``{(path, line, text)}``
over a fixture corpus and over this repository's own tree. The corpus mixes the
cases the script's own unit tests cover with the drift cases the other copies of
the gate were written for, and lives in ``tests/fixtures/validator_parity``.

The golden verdict is the script's recorded output over the corpus. While the
script exists this test proves the golden is the script's verdict; the switch PR
deletes the script and keeps the node-versus-golden half as the regression test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_core.nodes.node_no_env_fallbacks_check_compute import (
    matcher_env_fallbacks,
)
from omnibase_core.nodes.node_no_env_fallbacks_check_compute.matcher_env_fallbacks import (
    find_env_fallback_violations,
)
from omnibase_core.nodes.node_no_env_fallbacks_check_compute.runtime_no_env_fallbacks_check import (
    main as node_main,
)
from scripts.validate_no_env_fallbacks import run_on_files

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "validator_parity" / "no_env_fallbacks"

Finding = tuple[str, int, str]


def _corpus() -> list[tuple[str, str]]:
    raw = json.loads((FIXTURES / "corpus.json").read_text(encoding="utf-8"))
    return [(entry["path"], entry["source"]) for entry in raw]


def _golden() -> set[Finding]:
    raw = json.loads((FIXTURES / "golden_findings.json").read_text(encoding="utf-8"))
    return {(p, int(n), t) for p, n, t in raw}


def _write_corpus(root: Path) -> list[Path]:
    paths: list[Path] = []
    for rel, source in _corpus():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
        paths.append(Path(rel))
    return paths


def _script_findings(files: list[Path], root: Path) -> set[Finding]:
    return {
        (path, lineno, text.strip()) for path, lineno, text in run_on_files(files, root)
    }


def _node_findings(files: list[Path], root: Path) -> set[Finding]:
    found: set[Finding] = set()
    for rel in files:
        source = (root / rel).read_text(encoding="utf-8")
        for finding in find_env_fallback_violations(str(rel), source):
            path, lineno, text = finding.message.split(":", 2)
            found.add((path, int(lineno), text.strip()))
    return found


@pytest.mark.unit
def test_node_parity_golden_is_the_scripts_verdict(tmp_path: Path) -> None:
    files = _write_corpus(tmp_path)
    golden = _golden()
    assert golden, "corpus must contain violating inputs"
    assert _script_findings(files, tmp_path) == golden


@pytest.mark.unit
def test_node_parity_corpus_findings_match_script_and_golden(tmp_path: Path) -> None:
    files = _write_corpus(tmp_path)
    node = _node_findings(files, tmp_path)
    assert node == _script_findings(files, tmp_path)
    assert node == _golden()


@pytest.mark.unit
@pytest.mark.parametrize("rel", [path for path, _ in _corpus()])
def test_node_parity_corpus_exit_code_matches_script_per_file(
    rel: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_corpus(tmp_path)
    monkeypatch.chdir(tmp_path)
    script_exit = 1 if run_on_files([Path(rel)], tmp_path) else 0
    node_exit = node_main([rel])
    capsys.readouterr()
    assert node_exit == script_exit, f"{rel}: node={node_exit} script={script_exit}"


@pytest.mark.unit
def test_node_parity_repo_tree_findings_and_exit_code_match_script(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The src and scripts trees, the scope the CI backstop scanned."""
    monkeypatch.chdir(REPO_ROOT)
    files = [
        p
        for p in sorted(
            list((REPO_ROOT / "src").rglob("*"))
            + list((REPO_ROOT / "scripts").rglob("*"))
        )
        if p.suffix in {".py", ".sh", ".bash"} and p.is_file()
    ]
    rel_files = [p.relative_to(REPO_ROOT) for p in files]
    script = _script_findings(rel_files, REPO_ROOT)
    node = _node_findings(rel_files, REPO_ROOT)
    assert node == script
    assert node_main([str(p) for p in rel_files]) == (1 if script else 0)
    capsys.readouterr()


@pytest.mark.unit
def test_node_parity_comparison_detects_a_broken_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parity check has teeth: a node that drops a pattern is reported as differing."""
    files = _write_corpus(tmp_path)
    monkeypatch.setattr(
        matcher_env_fallbacks,
        "_PYTHON_FALLBACK_PATTERNS",
        matcher_env_fallbacks._PYTHON_FALLBACK_PATTERNS[1:],
    )
    assert _node_findings(files, tmp_path) != _golden()
