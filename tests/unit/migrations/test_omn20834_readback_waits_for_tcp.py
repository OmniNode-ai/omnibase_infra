# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The OMN-20834 readback waits for the real PostgreSQL server, not the init one.

The postgres image's first boot runs a temporary socket-only server and then
restarts it. The contract's original probe asked pg_isready over the socket,
got an answer from that temporary server, and its psql then failed against the
restart (observed 2026-10-20). It also fell through to psql when the server
never answered. The readback now lives in a script that checks readiness over
TCP, which the init server does not listen on, and fails when it never answers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "validation" / "check_generation_cost_null_readback.sh"
_CONTRACT = _REPO / "contracts" / "OMN-20834.yaml"
_ITEM = "dod-omn20834-postgres-null-readback"


def _code_lines() -> list[str]:
    if not _SCRIPT.exists():
        pytest.fail(f"{_SCRIPT} is missing")
    return [
        line.strip()
        for line in _SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@pytest.mark.unit
def test_every_readiness_check_and_psql_use_tcp() -> None:
    lines = _code_lines()
    probes = [line for line in lines if "pg_isready" in line or "psql " in line]
    assert any("pg_isready" in line for line in probes)
    assert any("psql " in line for line in probes)
    for line in probes:
        assert "-h 127.0.0.1" in line, f"socket connection, not TCP: {line}"


@pytest.mark.unit
def test_a_server_that_never_answers_fails_the_run() -> None:
    text = "\n".join(_code_lines())
    assert 'if [ "$ready" -ne 1 ]; then' in text
    assert text.index("exit 1") < text.index("docker exec -i")


@pytest.mark.unit
def test_the_contract_runs_the_script() -> None:
    contract = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    item = next(i for i in contract["dod_evidence"] if i["id"] == _ITEM)
    commands = [str(check["check_value"]).strip() for check in item["checks"]]
    assert commands == ["bash scripts/validation/check_generation_cost_null_readback.sh"]
