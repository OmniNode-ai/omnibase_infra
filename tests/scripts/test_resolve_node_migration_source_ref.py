# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "resolve_node_migration_source_ref.py"
)


def _run(tmp_path: Path, body: str | None) -> subprocess.CompletedProcess[str]:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps({"pull_request": {"body": body}}), encoding="utf-8"
    )
    output_path = tmp_path / "github_output.txt"
    env = {
        **os.environ,
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_OUTPUT": str(output_path),
    }
    return subprocess.run(
        ["python3", str(SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_defaults_to_dev_without_metadata(tmp_path: Path) -> None:
    result = _run(tmp_path, "Refs OMN-15038")

    assert result.returncode == 0
    assert result.stdout.strip() == "dev"
    assert (tmp_path / "github_output.txt").read_text(encoding="utf-8") == "ref=dev\n"


def test_reads_explicit_omnimarket_source_ref(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "Refs OMN-15038\nOmnimarket-Source-Ref: jonah/omn-15038-drop-unwired-routing-columns",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "jonah/omn-15038-drop-unwired-routing-columns"


def test_rejects_unsafe_ref(tmp_path: Path) -> None:
    result = _run(tmp_path, "Omnimarket-Source-Ref: ../dev")

    assert result.returncode == 1
    assert "invalid omnimarket source ref" in result.stderr
