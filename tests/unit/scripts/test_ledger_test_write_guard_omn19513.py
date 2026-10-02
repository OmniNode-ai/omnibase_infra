# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""scripts/ledger_lock.py refuses ledger writes under a test runner (OMN-19513).

The falsifiers reproduce the leak of 2026-10-01 (fixture rows, lane=alpha ticket=OMN-1, reached
the ledger of record from a test run): a fixture append to the canonical ledger must exit 79
naming the guard and leave the bytes unchanged; the same append to a scratch ledger must succeed;
and with the refusal switched off the canonical append lands, so the test fails without the guard.

"Canonical" is judged against the temporary directory, so each test narrows the guard's temporary
root to ``tmp_path/scratch`` and puts the "canonical" ledger beside it: every byte these tests
could write, even on a regression, stays under ``tmp_path``.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.handlers import handler_ledger_write_guard as guard

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "ledger_lock.py"


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ledger_lock_guard_under_test", _SCRIPT
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_script()
REFUSED = guard.EXIT_TEST_WRITE_REFUSED


def _row() -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{stamp} | STATUS | lane=alpha | ticket=OMN-1 | a fixture row"


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(guard, "_temp_root", lambda: scratch.resolve())
    omni_home = tmp_path / "omni_home"
    omni_home.mkdir()
    canonical = omni_home / "ROLLING_WORK_LEDGER.md"
    canonical.write_text("## Work ledger\n", encoding="utf-8")
    test_ledger = scratch / "ROLLING_WORK_LEDGER.md"
    test_ledger.write_text("## Work ledger\n", encoding="utf-8")
    monkeypatch.setenv("OMNI_HOME", str(omni_home))
    # The script loads the guard by path; hand it this module object so the narrowed temp root
    # applies to it. The row grammar lives in the omni_home clone, which a unit test does not
    # have: it is a different gate and these tests judge only the write guard.
    monkeypatch.setattr(MOD, "load_write_guard", lambda: guard)
    monkeypatch.setattr(MOD, "validate_grammar_payload", lambda payload, ledger: None)
    monkeypatch.setattr(MOD, "validate_grammar_state", lambda payload, ledger: None)
    return {"canonical": canonical, "test": test_ledger, "scratch": scratch}


def _run(ledger: Path, *extra: str) -> int:
    return int(MOD.main([str(ledger), *extra]))


def test_an_append_to_the_canonical_ledger_exits_79_and_leaves_bytes_unchanged(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    before = world["canonical"].read_bytes()
    code = _run(world["canonical"], "--append", _row())
    err = capsys.readouterr().err
    assert code == REFUSED
    assert guard.GUARD_NAME in err and str(world["canonical"]) in err
    assert world["canonical"].read_bytes() == before
    assert not (world["canonical"].parent / MOD.DEFAULT_LOCK_DIRNAME).exists()


def test_the_same_append_to_a_scratch_ledger_succeeds(world: dict[str, Path]) -> None:
    assert _run(world["test"], "--append", _row()) == 0
    assert "a fixture row" in world["test"].read_text(encoding="utf-8")


def test_the_editor_command_on_the_canonical_ledger_is_refused_before_it_runs(
    world: dict[str, Path],
) -> None:
    marker = world["scratch"] / "ran"
    before = world["canonical"].read_bytes()
    code = int(MOD.main([str(world["canonical"]), "--", "touch", str(marker)]))
    assert code == REFUSED
    assert not marker.exists()
    assert world["canonical"].read_bytes() == before


def test_the_refusal_switched_off_lets_the_canonical_append_land(
    world: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Falsifier: without the refusal the fixture row reaches the canonical file."""
    monkeypatch.setattr(guard, "check_file", lambda path, environ: None)
    assert _run(world["canonical"], "--append", _row()) == 0
    assert "a fixture row" in world["canonical"].read_text(encoding="utf-8")


def test_a_subprocess_of_a_test_is_refused_through_the_inherited_signal(
    world: dict[str, Path],
) -> None:
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env["ONEX_TEST_CONTEXT"] = "pytest"
    # The child's temp root is the scratch directory, so the canonical ledger beside it is
    # outside it, and a regression still writes only under tmp_path.
    env["TMPDIR"] = str(world["scratch"])
    env.pop("OMNI_HOME", None)
    before = world["canonical"].read_bytes()
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), str(world["canonical"]), "--append", _row()],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == REFUSED, proc.stderr
    assert guard.GUARD_NAME in proc.stderr
    assert world["canonical"].read_bytes() == before


def test_a_canonical_topic_and_a_shared_dsn_are_refused() -> None:
    topic = "onex.cmd.omnimarket.work-ledger-append-requested.v1"
    with pytest.raises(guard.LedgerTestWriteRefusedError):
        guard.check_topic(topic, os.environ)
    with pytest.raises(guard.LedgerTestWriteRefusedError):
        guard.check_dsn(
            "postgresql://u@db.example.internal:5432/omnibase_infra", os.environ
        )
    guard.check_topic("onex.evt.test.scratch.v1", os.environ)
    guard.check_dsn("postgresql://u@localhost:5432/scratch", os.environ)


def test_a_ledger_under_a_real_omni_home_is_canonical_wherever_the_temp_dir_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    omni_home = tmp_path / "registry"
    omni_home.mkdir()
    monkeypatch.setattr(guard, "_temp_root", lambda: tmp_path.resolve())
    monkeypatch.setenv("OMNI_HOME", str(omni_home))
    assert not guard.file_is_canonical(omni_home / "ROLLING_WORK_LEDGER.md", os.environ)
    monkeypatch.setattr(guard, "_temp_root", lambda: (tmp_path / "elsewhere").resolve())
    assert guard.file_is_canonical(omni_home / "ROLLING_WORK_LEDGER.md", os.environ)


def test_the_suite_gives_each_test_a_scratch_ledger_and_no_bus_route() -> None:
    ledger = Path(os.environ["ONEX_LEDGER_PATH"])
    assert not guard.file_is_canonical(ledger, os.environ)
    assert "ONEX_LEDGER_WRITE_VIA" not in os.environ
    assert os.environ["ONEX_TEST_CONTEXT"] == "pytest"
