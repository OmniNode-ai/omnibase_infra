# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise kernel startup and real OS file limits in an isolated process."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration

_BOOT_WITH_LOW_NOFILE = """
import json
import resource
from contextlib import ExitStack

from omnibase_infra.runtime import service_kernel

_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
before = resource.getrlimit(resource.RLIMIT_NOFILE)


async def exercise_boot():
    # Keep the real main(), limit adjustment and asyncio event loop. Replace
    # only the broker-backed bootstrap with real file acquisition at boot.
    soft, current_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    with ExitStack() as stack:
        files = [stack.enter_context(open('/dev/null', 'rb')) for _ in range(512)]
        print(json.dumps({
            'before': before,
            'boot_limits': [soft, current_hard],
            'opened': len(files),
            'infinity': resource.RLIM_INFINITY,
        }), flush=True)
    return 0


service_kernel.bootstrap = exercise_boot
service_kernel.main()
"""


@pytest.mark.skipif(sys.platform == "win32", reason="resource is Unix-only")
def test_kernel_boot_opens_files_beyond_inherited_soft_limit() -> None:
    import resource

    parent_limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    completed = subprocess.run(
        [sys.executable, "-c", _BOOT_WITH_LOW_NOFILE],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    readback = json.loads(completed.stdout.splitlines()[-1])
    before_soft, before_hard = readback["before"]
    boot_soft, boot_hard = readback["boot_limits"]
    assert before_soft == 256
    assert boot_hard == before_hard
    assert boot_soft == (1_048_576 if boot_hard == readback["infinity"] else boot_hard)
    assert readback["opened"] == 512
    assert "RLIMIT_NOFILE soft limit raised from 256" in completed.stderr
    assert resource.getrlimit(resource.RLIMIT_NOFILE) == parent_limits
