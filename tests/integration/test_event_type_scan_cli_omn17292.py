# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Exercise filesystem discovery, AST scanning and CLI verdicts together."""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from omnibase_infra.validators import no_literal_event_type_in_tests as validator

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("literal_site", [False, True])
def test_cli_scan_work_budget_and_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    literal_site: bool,
) -> None:
    text = "class TestManyCases:\n" + "".join(
        f"    def test_case_{i}(self):\n"
        + "".join(f"        value_{j} = {j}\n" for j in range(30))
        for i in range(100)
    )
    if literal_site:
        text += (
            "    def test_envelope(self):\n"
            '        ModelEventEnvelope(event_type="onex.evt.producer.created.v1")\n'
        )
    nested = tmp_path / "nested"
    nested.mkdir()
    module = nested / "test_many_cases.py"
    module.write_text(text, encoding="utf-8")
    # Discovery must ignore malformed files outside the test-module pattern.
    (tmp_path / "helper.py").write_text("def broken(:\n", encoding="utf-8")
    walk = ast.walk
    node_count = sum(1 for _ in walk(ast.parse(text)))
    visits = 0

    def counted_walk(node: ast.AST) -> Iterator[ast.AST]:
        nonlocal visits
        for child in walk(node):
            visits += 1
            yield child

    monkeypatch.setattr(ast, "walk", counted_walk)
    exit_code = validator.main([str(tmp_path), "--min-test-files", "1"])
    output = capsys.readouterr()
    assert exit_code == int(literal_site)
    assert output.out == ""
    if literal_site:
        assert f"{module}:{len(text.splitlines())}" in output.err
        assert "is a TOPIC" in output.err
    else:
        assert "OK: 1 test modules scanned" in output.err
    assert visits <= 2 * node_count, (visits, node_count)


def _run_cli(
    root: Path, *, min_test_files: int = 1
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "omnibase_infra.validators.no_literal_event_type_in_tests",
            str(root),
            "--min-test-files",
            str(min_test_files),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_package_cli_scans_aliases_then_refuses_indirect_topics_and_parse_errors(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    alias_module = nested / "test_alias.py"
    alias_module.write_text(
        'ModelEventEnvelope(event_type="producer.created")\n', encoding="utf-8"
    )
    clean = _run_cli(tmp_path)
    assert clean.returncode == 0, clean.stderr
    assert "OK: 1 test modules scanned" in clean.stderr

    indirect = nested / "test_indirect.py"
    indirect.write_text(
        "def test_envelope():\n"
        '    topic = "onex.evt.producer.created.v1"\n'
        "    ModelEventEnvelope(event_type=topic)\n",
        encoding="utf-8",
    )
    golden = tmp_path / "test_golden_chain_probe.py"
    golden.write_text(
        'data = {"event_\\x74ype": "onex.evt.producer.created.v1"}\n',
        encoding="utf-8",
    )
    malformed = tmp_path / "test_malformed.py"
    malformed.write_text("def broken(:\n", encoding="utf-8")

    refused = _run_cli(tmp_path)
    assert refused.returncode == 1, refused.stderr
    assert refused.stdout == ""
    assert f"{indirect}:3" in refused.stderr
    assert f"{golden}:1" in refused.stderr
    assert f"{malformed}:1" in refused.stderr
    assert "module does not parse" in refused.stderr
    assert str(alias_module) not in refused.stderr


def test_package_cli_refuses_an_empty_scan(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, min_test_files=validator.DEFAULT_MIN_TEST_FILES)
    assert result.returncode == 1, result.stderr
    assert "FAIL (vacuity guard): only 0 test modules discovered" in result.stderr
