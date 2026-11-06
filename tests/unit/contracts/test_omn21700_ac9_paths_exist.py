# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The validator-clean check of OMN-21700 names only files that exist (OMN-22617).

Core's handler-shape validator skips a path it cannot read and exits 0, so a check that names a
file that is not there reports clean for a file it never read. This reads the check's own command
and requires every ``src/`` path it names to exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT = _ROOT / "contracts" / "OMN-21700.yaml"
_CHECK_ID = "dod-omn21700-ac9-validator-clean"


def _named_source_paths() -> list[str]:
    document = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    items = [i for i in document["dod_evidence"] if i["id"] == _CHECK_ID]
    assert len(items) == 1, f"{_CHECK_ID} must be one evidence item"
    command = " ".join(str(c["check_value"]) for c in items[0]["checks"])
    return [t for t in command.split() if t.startswith("src/")]


def test_the_validator_clean_check_names_handler_files() -> None:
    assert _named_source_paths(), "the check names no src path at all"


def test_every_source_path_the_validator_clean_check_names_exists() -> None:
    missing = [p for p in _named_source_paths() if not (_ROOT / p).is_file()]
    assert missing == [], f"the check names files that are not in the tree: {missing}"
