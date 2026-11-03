# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Falsifiers of the staged-diff ``topic-naming-lint`` hook (OMN-22323).

A violation planted in a changed file is refused with the exit code and the output the
whole-tree scan gives. A violation in a file the commit does not touch is not judged by
the staged entry (each finding is a function of one file) and is caught by the whole-tree
mode, which a call with no file argument, a staged validator input, CI and
``pre-commit run --all-files`` all run.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from omnibase_infra.handlers import staged_commit_hook_scans as staged

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
BAD_TOPIC = "onex.evt.nosuchproducer.thing-happened.v1"
BAD_PYTHON = f"TOPIC = {BAD_TOPIC!r}\n"
MIXED_CONSUMER = (
    "TOPICS = ('tenant-a.onex.evt.platform.node-registration.v1', "
    "'onex.evt.platform.node-registration.v1')\n"
    "consumer.subscribe(topics=TOPICS)\n"
)
GOOD_PYTHON = "TOPIC = 'onex.evt.platform.node-registration.v1'\n"
BAD_CONTRACT = f"name: node_planted\nevent_bus:\n  publish_topics:\n    - {BAD_TOPIC}\n"
GOOD_CONTRACT = (
    "name: node_clean\n"
    "event_bus:\n"
    "  publish_topics:\n"
    "    - onex.evt.platform.node-registration.v1\n"
)


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in (
        "scripts/validation/lint_topic_names.py",
        "scripts/validation/topic_naming_baseline.txt",
        "src/omnibase_infra/nodes/node_contract_validate_compute/contract.yaml",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / name, target)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def _run(
    root: Path, capsys: pytest.CaptureFixture[str], paths: list[str]
) -> tuple[int, str, str]:
    code = staged.topic_naming_lint(root, paths)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_a_bad_topic_in_a_changed_python_file_is_refused_like_the_whole_tree_scan(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(
        tree,
        {
            "src/omnibase_infra/bad.py": BAD_PYTHON,
            "src/omnibase_infra/good.py": GOOD_PYTHON,
        },
    )
    code, out, err = _run(tree, capsys, ["src/omnibase_infra/bad.py"])
    assert code == 1
    assert "bad.py:1:" in err
    assert (code, out, err) == _run(tree, capsys, [])


def test_a_mixed_consumer_declaration_in_a_changed_file_is_refused_like_the_whole_tree_scan(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tree, {"src/omnibase_infra/consumer.py": MIXED_CONSUMER})
    code, out, err = _run(tree, capsys, ["src/omnibase_infra/consumer.py"])
    assert code == 1
    assert "mixed consumer topic namespace" in err
    assert (code, out, err) == _run(tree, capsys, [])


def test_a_bad_topic_in_a_changed_contract_is_refused_like_the_whole_tree_scan(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(
        tree,
        {
            "src/omnibase_infra/nodes/node_planted/contract.yaml": BAD_CONTRACT,
            "src/omnibase_infra/nodes/node_clean/contract.yaml": GOOD_CONTRACT,
        },
    )
    code, out, err = _run(
        tree, capsys, ["src/omnibase_infra/nodes/node_planted/contract.yaml"]
    )
    assert code == 1
    assert "node_planted/contract.yaml" in err
    assert (code, out, err) == _run(tree, capsys, [])


def test_a_bad_topic_in_an_unchanged_file_is_caught_by_the_whole_tree_mode(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(
        tree,
        {
            "src/omnibase_infra/bad.py": BAD_PYTHON,
            "src/omnibase_infra/good.py": GOOD_PYTHON,
        },
    )
    code, out, _err = _run(tree, capsys, ["src/omnibase_infra/good.py"])
    assert code == 0
    assert out.count("OK: no topic naming violations found under") == 2
    code, _out, err = _run(tree, capsys, [])
    assert code == 1
    assert "bad.py:1:" in err


@pytest.mark.parametrize(
    "validator_input",
    [
        "scripts/validation/lint_topic_names.py",
        "scripts/validation/topic_naming_baseline.txt",
        "src/omnibase_infra/nodes/node_contract_validate_compute/contract.yaml",
        "src/omnibase_infra/runtime/service_dispatch_result_applier.py",
        "src/omnibase_infra/event_bus/topic_constants.py",
    ],
)
def test_a_staged_validator_input_judges_the_whole_tree(
    tree: Path, capsys: pytest.CaptureFixture[str], validator_input: str
) -> None:
    _write(tree, {"src/omnibase_infra/bad.py": BAD_PYTHON})
    code, _out, err = _run(tree, capsys, [validator_input])
    assert code == 1
    assert "bad.py:1:" in err


def test_staged_files_outside_the_scanned_roots_are_not_judged(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(
        tree,
        {"src/omnibase_infra/bad.py": BAD_PYTHON, "tests/test_thing.py": BAD_PYTHON},
    )
    code, out, _err = _run(tree, capsys, ["tests/test_thing.py"])
    assert code == 0
    assert out.count("OK: no topic naming violations found under") == 2
