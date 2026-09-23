# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The thin caller of the rolling ledger row grammar (OMN-19256, OMN-16728).

The grammar itself -- which row types the rolling work ledger admits, and the
fields each requires -- is decided by a committed module in the omni_home
registry, where its own tests run its real bytes. This repo carries only the
caller, so these tests pin the CALLING CONTRACT with a stub module standing in
for the real one:

1. A ledger named ROLLING_WORK_LEDGER.md consults the module before the lock
   (``refusal_for_payload``) and inside it (``refusal_for_state``), and a
   refusal from either writes nothing and exits 65.
2. The module being ABSENT refuses every append to a governed ledger. The
   operator ruling allows no warn-only mode, and a clone at a revision without
   the module would otherwise switch the grammar off for every lane.
3. Any other ledger never consults the module.
4. ``--print-grammar`` prints the module's JSON, and exits 2 without it.
5. The editor verb (``-- COMMAND``) is judged by the same module.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ledger_lock.py"

STUB = """
import json

GOVERNED_LEDGER_NAMES = frozenset({"ROLLING_WORK_LEDGER.md"})


def refusal_for_payload(payload, ledger_name):
    if "| BADTYPE |" in payload:
        return "stub grammar refused BADTYPE in " + ledger_name
    return None


def refusal_for_state(payload, existing, ledger_name):
    if "| DUPE |" in payload and "| DUPE |" in existing:
        return "stub grammar refused a second DUPE"
    return None


def grammar_json():
    return json.dumps({"types": {"STATUS": {}}})
"""


def _load():
    spec = importlib.util.spec_from_file_location("ledger_lock_grammar_caller", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ll = _load()


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "shared" / "ledger_grammar.py"
    path.parent.mkdir()
    path.write_text(STUB, encoding="utf-8")
    monkeypatch.setattr(ll, "_LEDGER_GRAMMAR_PATH", path)
    monkeypatch.setenv(ll.NOW_OVERRIDE_ENV, _now())
    return path


@pytest.fixture
def governed(tmp_path: Path) -> Path:
    path = tmp_path / "tracking" / "ROLLING_WORK_LEDGER.md"
    path.parent.mkdir()
    path.write_text("## log\n", encoding="utf-8")
    return path


@pytest.mark.unit
def test_a_refused_payload_writes_nothing_and_exits_65(
    stub: Path, governed: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = governed.read_text(encoding="utf-8")
    assert (
        ll.main([str(governed), "--append", f"{_now()} | BADTYPE | lane=x | y"]) == 65
    )
    assert governed.read_text(encoding="utf-8") == before
    assert "stub grammar refused BADTYPE" in capsys.readouterr().err


@pytest.mark.unit
def test_an_admitted_payload_lands(stub: Path, governed: Path) -> None:
    row = f"{_now()} | STATUS | lane=x | fine"
    assert ll.main([str(governed), "--append", row]) == 0
    assert row in governed.read_text(encoding="utf-8")


@pytest.mark.unit
def test_the_state_half_runs_inside_the_lock(stub: Path, governed: Path) -> None:
    assert ll.main([str(governed), "--append", f"{_now()} | DUPE | lane=x | one"]) == 0
    assert ll.main([str(governed), "--append", f"{_now()} | DUPE | lane=x | two"]) == 65
    assert "| two" not in governed.read_text(encoding="utf-8")


@pytest.mark.unit
def test_an_absent_module_fails_closed_on_a_governed_ledger(
    tmp_path: Path,
    governed: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(ll, "_LEDGER_GRAMMAR_PATH", tmp_path / "absent.py")
    monkeypatch.setenv(ll.NOW_OVERRIDE_ENV, _now())
    before = governed.read_text(encoding="utf-8")
    assert ll.main([str(governed), "--append", f"{_now()} | STATUS | lane=x | y"]) == 65
    assert governed.read_text(encoding="utf-8") == before
    assert "fails closed" in capsys.readouterr().err


@pytest.mark.unit
def test_another_ledger_never_consults_the_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ll, "_LEDGER_GRAMMAR_PATH", tmp_path / "absent.py")
    monkeypatch.setenv(ll.NOW_OVERRIDE_ENV, _now())
    other = tmp_path / "OTHER_LEDGER.md"
    assert ll.main([str(other), "--append", f"{_now()} | BADTYPE | lane=x | y"]) == 0


@pytest.mark.unit
def test_print_grammar_prints_the_module_json(
    stub: Path, governed: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ll.main([str(governed), "--print-grammar"]) == 0
    assert '"STATUS"' in capsys.readouterr().out


@pytest.mark.unit
def test_print_grammar_without_the_module_exits_2(
    tmp_path: Path, governed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ll, "_LEDGER_GRAMMAR_PATH", tmp_path / "absent.py")
    assert ll.main([str(governed), "--print-grammar"]) == 2


@pytest.mark.unit
def test_the_editor_verb_is_judged_by_the_same_module(
    stub: Path, governed: Path
) -> None:
    before = governed.read_text(encoding="utf-8")
    script = (
        "import sys; open(sys.argv[1], 'a').write("
        f"'{_now()} | BADTYPE | lane=x | via the editor\\n')"
    )
    rc = ll.main([str(governed), "--", sys.executable, "-c", script, str(governed)])
    assert rc == 65
    assert governed.read_text(encoding="utf-8") == before
