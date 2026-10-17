# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The real entrypoint stops retrying a boot that cannot succeed (OMN-18550).

No fakes: this starts ``python -m omnibase_infra.runtime.entrypoint_preflight
onex-runtime`` as a fresh process, the way the container's restart policy
relaunches it, against a contracts directory whose ``runtime_config.yaml`` is
malformed YAML. The real kernel loads that contract and exits 1. The test
relaunches it as the restart policy would and asserts what the container would
show: three counted failures with their cause, then a parked process that no
longer boots, and the durable ledger holding one event per failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

_BOUND = 3
_RUN_TIMEOUT_SECONDS = 120.0


def _env(state_dir: Path, contracts_dir: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", str(state_dir)),
        "ONEX_STATE_DIR": str(state_dir),
        "ONEX_CONTRACTS_DIR": str(contracts_dir),
        "ONEX_GROUP_ID": "omn18550-itest",
        "HOSTNAME": "omn18550-container",
        "RUNTIME_SOURCE_HASH": "omn18550-hash",
        "ONEX_BOOT_FAILURE_MAX_CONSECUTIVE": str(_BOUND),
    }


def _launch(env: dict[str, str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "omnibase_infra.runtime.entrypoint_preflight",
            "onex-runtime",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_a_malformed_contract_is_counted_then_the_entrypoint_stops_booting(
    tmp_path: Path,
) -> None:
    contracts_dir = tmp_path / "contracts"
    (contracts_dir / "runtime").mkdir(parents=True)
    (contracts_dir / "runtime" / "runtime_config.yaml").write_text(
        'runtime: [unclosed\n  name: "deliberately malformed\n', encoding="utf-8"
    )
    state_dir = tmp_path / "state"
    env = _env(state_dir, contracts_dir)

    for attempt in range(1, _BOUND):
        proc = _launch(env)
        _, stderr = proc.communicate(timeout=_RUN_TIMEOUT_SECONDS)
        assert proc.returncode == 1, stderr
        assert f"BOOT FAILED {attempt}/{_BOUND}" in stderr
        assert "Failed to parse runtime config YAML" in stderr

    # The last allowed boot fails, records the exhausted event, and parks.
    parked = _launch(env)
    assert parked.stderr is not None
    seen: list[str] = []
    deadline = time.monotonic() + _RUN_TIMEOUT_SECONDS
    try:
        while time.monotonic() < deadline:
            line = parked.stderr.readline()
            if not line:
                break
            seen.append(line)
            if "not booting again" in line:
                break
        assert any(f"BOOT FAILED {_BOUND}/{_BOUND}" in x for x in seen), seen
        assert any("RETRIES EXHAUSTED" in x for x in seen), seen
        assert parked.poll() is None, "an exhausted entrypoint must idle, not exit"
    finally:
        parked.terminate()
        parked.wait(timeout=30)

    # A relaunch by the restart policy neither boots nor counts again.
    relaunched = _launch(env)
    assert relaunched.stderr is not None
    try:
        line = relaunched.stderr.readline()
        assert "RETRIES EXHAUSTED" in line, line
    finally:
        relaunched.terminate()
        relaunched.wait(timeout=30)

    events = [
        json.loads(x)
        for x in (state_dir / "boot_failures" / "events.jsonl").read_text().splitlines()
    ]
    assert [e["consecutive_failures"] for e in events] == [1, 2, 3]
    assert [e["retries_exhausted"] for e in events] == [False, False, True]
    assert all(e["exit_code"] == 1 for e in events)
    assert "Failed to parse runtime config YAML" in events[-1]["cause"]


__all__: list[str] = []
