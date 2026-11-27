# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26430: the deploy probe of ``contracts/OMN-26430.yaml`` greps a symbol that exists.

The post-deploy probe reads the live runtime container, which a hosted runner does
not have, so it runs only in the local Done gate and binds no acceptance criterion.
Because nothing on a pull request ever runs it, a probe naming a symbol that never
existed in source (``serialize_owned_database``) could not pass on any build and no
hosted check noticed. This test reads the probe's grep pattern and file path from the
contract and requires the pattern to occur in that source file in this checkout.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT = _ROOT / "contracts" / "OMN-26430.yaml"
_PROBE_ID = "dod-omn26430-deploy-owned-database-gate-live"
_CONTAINER_SRC = "/app/"


def _probe() -> dict[str, Any]:
    document = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    (item,) = [i for i in document["dod_evidence"] if i["id"] == _PROBE_ID]
    return item


def _pattern_and_source() -> tuple[str, Path]:
    (check,) = _probe()["checks"]
    argv = shlex.split(str(check["check_value"]))
    # docker exec <container> grep -q <pattern> <path in the container>
    assert argv[:2] == ["docker", "exec"], argv
    assert argv[3:5] == ["grep", "-q"], argv
    pattern, container_path = argv[5], argv[6]
    assert container_path.startswith(_CONTAINER_SRC), container_path
    return pattern, _ROOT / container_path.removeprefix(_CONTAINER_SRC)


def test_probe_is_local_done_gate_and_binds_no_criterion() -> None:
    probe = _probe()
    assert probe.get("execution_scope") == "local_done_gate"
    assert not probe.get("binds_ac")


def test_probe_symbol_occurs_in_the_source_file_it_greps() -> None:
    pattern, source = _pattern_and_source()
    assert source.is_file(), source
    assert pattern in source.read_text(encoding="utf-8"), (pattern, str(source))
