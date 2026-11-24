# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pre-merge pool driver reads and writes the work ledger over the bus (OMN-26507).

The driver took ``--ledger`` (default ``ONEX_LEDGER_PATH``) and read the file for the HOLD rows
that lease a lab host surface and for the RELEASE rows that decide the INCONCLUSIVE streak alarm.
A file copy is a second ledger (operator ruling 2026-11-05), so the driver now asks the ledger
reader node (reader ``prepr-runtime-pool``, query ``rows``, kinds HOLD and RELEASE) through the
command ``ONEX_LEDGER_READER_COMMAND`` names, and writes the alarm row through the bus append
command ``ONEX_LEDGER_BUS_APPEND_COMMAND`` names. Each test runs with ``ONEX_LEDGER_PATH`` unset and
no ledger file on disk; the interpreter that answers the reader is the stub of
``tests/scripts/ledger_reader_stub.py``.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.helpers.prepr_pool_overlay import pool_overlay_env
from tests.scripts.ledger_reader_stub import LedgerReaderStub

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_PY = REPO_ROOT / "scripts" / "runtime_build" / "prepr_runtime_pool.py"

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("pool_overlay_env")]

NOW = dt.datetime(2026, 11, 23, 6, 0, tzinfo=dt.UTC)
HOLD_ROW = (
    "2026-11-23T05:00:00Z | HOLD | lane=peer | id=h1 | surface=local-slot | "
    "until=2026-11-23T08:00:00Z | proof in flight"
)

# The files of the OMN-26507 group whose ledger file reader or writer moved to the bus. The ledger
# file census records every file that still holds one in config/ledger_file_census_ratchet.yaml (a
# hook keeps that table equal to the census node's own count), so a moved file has no row there.
LAUNCH_SCRIPT = "scripts/deploy-agent/deploy/deploy-agent-launch.sh"
MOVED_FILES = ("scripts/runtime_build/prepr_runtime_pool.py", LAUNCH_SCRIPT)
CENSUS_RATCHET = REPO_ROOT / "config" / "ledger_file_census_ratchet.yaml"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pool = _load("prepr_pool_ledger_bus_omn26507", POOL_PY)


@pytest.fixture
def bus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LedgerReaderStub:
    """A reader stub in the environment.

    The suite's isolation fixture (tests/conftest.py) points the ledger path variable at a scratch
    file that does not exist: code that opens it raises, code that asks the bus does not.
    """
    monkeypatch.chdir(tmp_path)
    stub = LedgerReaderStub(tmp_path / "ledger-reader")
    for name, value in stub.env().items():
        monkeypatch.setenv(name, value)
    (stub.directory / "rows.txt").write_text(f"{HOLD_ROW}\n", encoding="utf-8")
    return stub


def test_ledger_bus_reader_rows_are_the_live_surface_holds(
    bus: LedgerReaderStub, tmp_path: Path
) -> None:
    assert not list(tmp_path.glob("**/*LEDGER*.md"))
    lines = pool._ledger_lines()
    assert pool.live_surface_holds(lines, NOW) == {
        "local-slot": ("h1", "peer", "2026-11-23T08:00:00Z")
    }
    (call,) = bus.calls()
    assert "--reader prepr-runtime-pool" in call
    assert "--query rows" in call
    assert "--kind HOLD" in call and "--kind RELEASE" in call
    assert re.search(r"--window-hours \d+", call)


@pytest.mark.parametrize("code", [1, 3, 69])
def test_ledger_bus_refusal_is_an_unread_ledger_never_an_empty_one(
    bus: LedgerReaderStub, code: int
) -> None:
    bus.refuse_with(code)
    with pytest.raises(OSError, match=str(code)):
        pool._ledger_lines()


def test_ledger_bus_status_asks_the_reader_node_once(
    bus: LedgerReaderStub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "pool.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "slot": {
                    "compose_project": "omnibase-infra-local",
                    "ports": [8085, 8086],
                    "lease_dir": ".onex-prepr-pool/lease",
                    "work_root_prefix": "prove-pool-",
                    "model_endpoint": "http://192.0.2.10:8000",
                    "max_load_ratio": 1.0,
                },
                "hosts": [
                    {
                        "name": "local-slot",
                        "ssh_target": "192.0.2.55",
                        "surface": "local-slot",
                        "os": "linux",
                        "status": "pool",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class Offline:
        def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
            return 7, "fixed probe failure"

        def put(self, host: Any, local: Path, remote: str) -> int:
            return 8

    assert pool.main(["--config", str(config), "status"], transport=Offline()) == 0
    assert len(bus.calls()) == 1
    assert "OFFLINE" in capsys.readouterr().out


def test_ledger_bus_driver_takes_no_ledger_file_argument() -> None:
    with pytest.raises(SystemExit):
        pool.build_parser().parse_args(["--ledger", "/x/ledger.md", "status"])


def _census_rows() -> set[str]:
    table = yaml.safe_load(CENSUS_RATCHET.read_text(encoding="utf-8"))["files"]
    return {path for rows in table.values() for path in rows}


def test_no_ledger_file_path_census_table_is_readable() -> None:
    rows = _census_rows()
    assert rows, "the census table parsed to nothing, so the check below is vacuous"
    assert all((REPO_ROOT / path).is_file() for path in rows)


@pytest.mark.parametrize("relative", MOVED_FILES)
def test_no_ledger_file_path_in_a_moved_file(relative: str) -> None:
    assert relative not in _census_rows()
    assert (REPO_ROOT / relative).is_file()


def _launch_ledger_block() -> str:
    """The launch script's own dev ledger transport block, as written in the file."""
    text = (REPO_ROOT / LAUNCH_SCRIPT).read_text(encoding="utf-8")
    start = text.index('if [[ "${DEPLOY_AGENT_ALLOWED_LANES:-}" == "dev"')
    return text[start : text.index("\nfi\n", start) + 4]


def test_ledger_bus_deploy_agent_launch_hands_over_the_append_command_only(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry"
    done = subprocess.run(
        ["bash", "-c", f"{_launch_ledger_block()}\nenv"],
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": os.environ["PATH"],
            "DEPLOY_AGENT_ALLOWED_LANES": "dev",
            "ONEX_LEDGER_WRITE_VIA": "bus",
            "UNIT_DRIFT_REGISTRY_ROOT": str(registry),
        },
    )
    handed_over = dict(
        line.split("=", 1) for line in done.stdout.splitlines() if "=" in line
    )
    command = handed_over["ONEX_LEDGER_BUS_APPEND_COMMAND"]
    assert "-m omnimarket.work_ledger_bus.cli append --bus-lane dev" in command
    assert not [name for name in handed_over if name.startswith("ONEX_LEDGER_P")]
    assert not list(registry.glob("**/*"))
