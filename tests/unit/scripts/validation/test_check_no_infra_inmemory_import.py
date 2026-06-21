"""Tests for the OMN-13419 single-canonical in-memory bus import gate."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "validation" / "check_no_infra_inmemory_import.sh"


def _run_gate(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    script_path = tmp_path / "scripts" / "validation" / SCRIPT.name
    script_path.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, script_path)

    return subprocess.run(
        ["bash", str(script_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )


def test_gate_rejects_direct_module_import(tmp_path: Path) -> None:
    module = tmp_path / "src" / "omnibase_infra" / "nodes" / "bad_import.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "import omnibase_infra.event_bus.event_bus_inmemory as infra_bus\n",
        encoding="utf-8",
    )

    result = _run_gate(tmp_path)

    assert result.returncode == 1
    assert "Single-canonical in-memory bus" in result.stdout
    assert "bad_import.py" in result.stdout


def test_gate_allows_allowlisted_adapter_import(tmp_path: Path) -> None:
    adapter = (
        tmp_path / "src" / "omnibase_infra" / "event_bus" / "event_bus_inmemory.py"
    )
    adapter.parent.mkdir(parents=True)
    adapter.write_text(
        "import omnibase_infra.event_bus.event_bus_inmemory as infra_bus\n",
        encoding="utf-8",
    )

    result = _run_gate(tmp_path)

    assert result.returncode == 0
    assert "OK: no disallowed imports" in result.stdout
