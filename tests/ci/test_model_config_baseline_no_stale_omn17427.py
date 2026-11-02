# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The hardcoded model config baseline carries no stale entry (OMN-17427).

An entry is keyed by ``sha1(line.strip())`` of a line of the file it names. An
entry whose hash matches no line of that file at the current tree is stale: the
finding it baselined was fixed or moved, and the entry now only widens what the
baseline admits. The baseline header says entries only leave, so a stale entry
is removed rather than kept.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_BASELINE = "config/hardcoded_model_config_baseline.yaml"


def _line_key(line: str) -> str:
    digest = hashlib.sha1(line.strip().encode("utf-8"), usedforsecurity=False)
    return digest.hexdigest()


def _entries(baseline: Path) -> list[dict[str, Any]]:
    loaded = yaml.safe_load(baseline.read_text("utf-8"))
    entries: list[dict[str, Any]] = loaded["entries"]
    return entries


def stale_entries(root: Path, baseline: Path) -> list[str]:
    """Return one description per entry whose line is absent from its file."""
    line_keys: dict[str, set[str]] = {}
    stale: list[str] = []
    for entry in _entries(baseline):
        path = entry["path"]
        if path not in line_keys:
            target = root / path
            line_keys[path] = (
                {_line_key(line) for line in target.read_text("utf-8").splitlines()}
                if target.is_file()
                else set()
            )
        if entry["content_sha1"] not in line_keys[path]:
            stale.append(f"{path} family {entry['family']} {entry['content_sha1']}")
    return stale


def test_baseline_carries_no_stale_entry() -> None:
    """Every baseline entry names a line that exists in the file it names."""
    stale = stale_entries(ROOT, ROOT / MODEL_CONFIG_BASELINE)
    assert stale == [], f"{len(stale)} stale baseline entries: {stale}"


def test_stale_entry_is_detected(tmp_path: Path) -> None:
    """Negative control: an entry for a line the file does not hold is refused."""
    (tmp_path / "mod.py").write_text('MODEL = "kept"\n', "utf-8")
    kept = _line_key('MODEL = "kept"')
    gone = _line_key('MODEL = "gone"')
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "entries": [
                    {"path": "mod.py", "family": "O", "content_sha1": kept},
                    {"path": "mod.py", "family": "O", "content_sha1": gone},
                    {"path": "missing.py", "family": "O", "content_sha1": kept},
                ],
            }
        ),
        "utf-8",
    )
    assert stale_entries(tmp_path, baseline) == [
        f"mod.py family O {gone}",
        f"missing.py family O {kept}",
    ]
