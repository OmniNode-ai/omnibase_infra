# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A boot failure's cause is read from a real supervised process exit (OMN-26316).

Unlike the unit tests, which run the supervisor in the test process, this starts a separate
Python process whose kernel logs the dispatch engine's own record from the engine's own module
and then exits 1, and reads the boot-failed event that process published.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_CHILD = textwrap.dedent(
    """
    import json, logging, sys
    from omnibase_infra.runtime import entrypoint_preflight, message_dispatch_engine

    seen = []
    entrypoint_preflight._publish_boot_failed = lambda event, _env: seen.append(event)
    entrypoint_preflight.run_preflight = lambda _env: 0
    kernel_error = sys.argv[1]

    def handoff(_argv):
        log = logging.getLogger("omnibase_infra.runtime.service_kernel")
        if kernel_error:
            log.error(kernel_error)
        record = log.makeRecord(
            log.name, logging.ERROR, message_dispatch_engine.__file__, 1785,
            "Dispatch completed with errors", (), None,
        )
        log.handle(record)
        raise SystemExit(1)

    entrypoint_preflight.handoff = handoff
    env = {
        "ONEX_STATE_DIR": sys.argv[2],
        "ONEX_GROUP_ID": "onex-runtime-main",
        "RUNTIME_SOURCE_HASH": "abc123",
        "HOSTNAME": "c1",
        "ONEX_BOOT_FAILURE_MAX_CONSECUTIVE": "5",
    }
    code = entrypoint_preflight._supervised_start(["onex-runtime"], env)
    print(json.dumps({"code": code, "causes": [e.cause for e in seen]}))
    """
)


def _run_child(kernel_error: str, state_dir: Path) -> dict[str, object]:
    done = subprocess.run(
        [sys.executable, "-c", _CHILD, kernel_error, str(state_dir)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_real_process_exit_after_a_dispatch_error_records_no_kernel_error(
    tmp_path: Path,
) -> None:
    result = _run_child("", tmp_path)

    assert result["code"] == 1
    assert result["causes"] == [
        "no kernel error was logged before the exit; last dispatch error: "
        "Dispatch completed with errors"
    ]


def test_a_real_process_keeps_the_kernel_error_as_the_cause(tmp_path: Path) -> None:
    result = _run_child("Topic provisioning failed", tmp_path)

    assert result["code"] == 1
    assert result["causes"] == ["Topic provisioning failed"]
