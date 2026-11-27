# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26430: the deploy probe of ``contracts/OMN-26430.yaml`` greps a symbol that exists.

The post-deploy probe reads the live runtime container, which a hosted runner does
not have, so it runs only in the local Done gate and binds no acceptance criterion.
Because nothing on a pull request ever runs it, a probe naming a symbol that never
existed in source (``serialize_owned_database``) could not pass on any build and no
hosted check noticed. This test reads the probe's grep pattern and file path from the
contract and requires the pattern to occur in that source file, both as committed at
``HEAD``. The hosted control copies the new contract and tests over the older
commit's tree, so a working-tree read cannot tell the two apart; the committed read
can: at that older commit the committed contract names ``serialize_owned_database``, which is
in no committed source file, and the test fails.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT = "contracts/OMN-26430.yaml"
_CONTAINER_SRC = "/app/"


def _committed(path: str) -> str:
    """Return ``path`` as committed at HEAD, never the working tree."""
    return subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=_ROOT,
        env=scrub_git_location_env(os.environ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _probe() -> dict[str, Any]:
    document = yaml.safe_load(_committed(_CONTRACT))
    (item,) = [
        i
        for i in document["dod_evidence"]
        if any(str(c["check_value"]).startswith("docker exec ") for c in i["checks"])
    ]
    return item


def _pattern_and_source() -> tuple[str, str]:
    (check,) = _probe()["checks"]
    argv = shlex.split(str(check["check_value"]))
    # docker exec <container> grep -q <pattern> <path in the container>
    assert argv[:2] == ["docker", "exec"], argv
    assert argv[3:5] == ["grep", "-q"], argv
    pattern, container_path = argv[5], argv[6]
    assert container_path.startswith(_CONTAINER_SRC), container_path
    return pattern, container_path.removeprefix(_CONTAINER_SRC)


def test_probe_is_local_done_gate_and_binds_no_criterion() -> None:
    probe = _probe()
    assert probe.get("execution_scope") == "local_done_gate"
    assert not probe.get("binds_ac")


def test_probe_symbol_occurs_in_the_source_file_it_greps() -> None:
    pattern, source = _pattern_and_source()
    assert pattern in _committed(source), (pattern, source)
