# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime dependency installs must preserve exhausted uv failures."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("failures", [0, 1, 3])
def test_runtime_uv_retry_preserves_install_status(
    dockerfile_path: Path, tmp_path: Path, failures: int
) -> None:
    dockerfile = dockerfile_path.read_text(encoding="utf-8")
    script = dockerfile.split("cat > /usr/local/bin/uv-with-retry <<'EOF'\n", 1)[
        1
    ].split("\nEOF", 1)[0]
    wrapper = tmp_path / "uv-with-retry"
    wrapper.write_text(script, encoding="utf-8")
    # Simulate an index/install failure without network access or retry delays.
    uv = tmp_path / "uv"
    uv.write_text(
        '#!/bin/sh\ncount=0\n[ ! -f "$UV_TEST_CALLS" ] || '
        'count=$(cat "$UV_TEST_CALLS")\ncount=$((count + 1))\n'
        'echo "$count" > "$UV_TEST_CALLS"\n'
        '[ "$count" -gt "$UV_TEST_FAILURES" ] || exit 23\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    sleep = tmp_path / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    sleep.chmod(0o755)
    calls = tmp_path / "calls"
    result = subprocess.run(
        ["/bin/sh", str(wrapper), "pip", "install", "example-plugin"],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "UV_RETRY_ATTEMPTS": "3",
            "UV_TEST_CALLS": str(calls),
            "UV_TEST_FAILURES": str(failures),
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert int(calls.read_text(encoding="utf-8")) == min(failures + 1, 3)
    assert result.returncode == (23 if failures == 3 else 0), result.stderr
    if failures:
        assert "failed with exit 23" in result.stderr
    if failures == 3:
        assert "failed after 3 attempt(s)" in result.stderr
