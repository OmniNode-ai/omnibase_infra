# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Verdict parity: the removed validate_no_env_fallbacks script vs the core check node (OMN-20568).

The infra script and ``omnibase_core.nodes.node_no_env_fallbacks_check_compute``
must give the same exit code and the same finding set ``{(path, line, text)}``
over a fixture corpus and over this repository's own tree. The corpus mixes the
cases the script's own unit tests cover with the drift cases the other copies of
the gate were written for, and lives in ``tests/fixtures/validator_parity``.

The golden verdict (``golden_findings.json``) is the deleted script's recorded
output over the corpus, proven equal to the script's live output by this test at
commit 36d335d9a before the script was removed. The tree verdict is the script's
at the PR base: zero findings. The node is the regression-tested implementation.
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


def _node_findings(files: list[Path], root: Path) -> set[Finding]:
    found: set[Finding] = set()
    for rel in files:
        source = (root / rel).read_text(encoding="utf-8")
        for finding in find_env_fallback_violations(str(rel), source):
            path, lineno, text = finding.message.split(":", 2)
            found.add((path, int(lineno), text.strip()))
    return found


@pytest.mark.unit
def test_node_parity_corpus_findings_match_golden(tmp_path: Path) -> None:
    files = _write_corpus(tmp_path)
    golden = _golden()
    assert golden, "corpus must contain violating inputs"
    assert _node_findings(files, tmp_path) == golden


@pytest.mark.unit
@pytest.mark.parametrize("rel", [path for path, _ in _corpus()])
def test_node_parity_corpus_exit_code_matches_golden_per_file(
    rel: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_corpus(tmp_path)
    monkeypatch.chdir(tmp_path)
    expected_exit = 1 if any(path == rel for path, _, _ in _golden()) else 0
    node_exit = node_main([rel])
    capsys.readouterr()
    assert node_exit == expected_exit, f"{rel}: node={node_exit} golden={expected_exit}"


@pytest.mark.unit
def test_node_parity_repo_tree_is_clean_as_the_script_was(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The script found nothing in src and scripts at the base; the node must not either."""
    monkeypatch.chdir(REPO_ROOT)
    rel_files = [
        p.relative_to(REPO_ROOT)
        for p in sorted(
            list((REPO_ROOT / "src").rglob("*"))
            + list((REPO_ROOT / "scripts").rglob("*"))
        )
        if p.suffix in {".py", ".sh", ".bash"} and p.is_file()
    ]
    assert rel_files
    assert _node_findings(rel_files, REPO_ROOT) == set()
    assert node_main([str(p) for p in rel_files]) == 0
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
