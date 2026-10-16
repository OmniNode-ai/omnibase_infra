# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The OMN-20074 AC11 evidence pins the committed retired-corpus header counts.

A pull request that adds node contracts regenerates
``selection-oracle-no-occ-v2.header.json``; the AC11 evidence check in
``contracts/OMN-20074.yaml`` hardcodes the same counts and must move with it,
or every later OMN-20074 pull request inherits a failing check. Both files are
read as committed blobs at HEAD, so the check holds for the commit itself.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = "contracts/OMN-20074.yaml"
HEADER = "tests/fixtures/dispatch_parity/selection-oracle-no-occ-v2.header.json"
EVIDENCE_ID = "dod-dispatch-parity-declared-closure-ac11"


def _tracked(path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"HEAD:{path}"],
        cwd=REPO_ROOT,
        env=scrub_git_location_env(os.environ),
        text=True,
    )


def _ac11_command() -> str:
    contract = yaml.safe_load(_tracked(CONTRACT))
    (item,) = [i for i in contract["dod_evidence"] if i["id"] == EVIDENCE_ID]
    (command,) = [c for c in item["checks"] if c["check_type"] == "command"]
    return str(command["check_value"])


def _pinned(command: str, key: str) -> Any:
    match = re.search(
        rf"assert retired\['header'\]\['corpus'\]\['{key}'\] == (.+)$",
        command,
        re.MULTILINE,
    )
    assert match, f"AC11 evidence no longer pins retired {key}"
    return ast.literal_eval(match.group(1).strip())


def test_ac11_evidence_pins_the_committed_retired_header() -> None:
    corpus = json.loads(_tracked(HEADER))["corpus"]
    command = _ac11_command()
    assert _pinned(command, "contracts_discovered") == corpus["contracts_discovered"]
    assert _pinned(command, "contracts_by_package") == corpus["contracts_by_package"]


def test_ac11_statement_names_the_committed_retired_count() -> None:
    corpus = json.loads(_tracked(HEADER))["corpus"]
    contract = yaml.safe_load(_tracked(CONTRACT))
    statements = [
        acceptance["statement"]
        for requirement in contract["requirements"]
        for acceptance in requirement.get("acceptance", [])
        if acceptance["id"] == "AC11"
    ]
    assert statements
    assert (
        f"captured {corpus['contracts_discovered']}-contract retired" in statements[0]
    )
