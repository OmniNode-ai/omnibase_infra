# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dispatch the census adapter through its contract, bus and typed handler."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("first_exit", [0, 30, 8])
def test_census_bus_adapter_retains_sample_failure_until_completion(
    tmp_path: Path, first_exit: int
) -> None:
    """A clean follow-up must not erase an earlier alarm or publication failure."""
    collector = tmp_path / "lane-census-check.sh"
    collector.write_text(
        """set -eu
root="$(dirname "$0")"
count=0
if [ -f "$root/samples" ]; then
    read -r count < "$root/samples"
fi
count=$((count + 1))
printf '%s\\n' "$count" > "$root/samples"
printf '%s\\n' "$*" >> "$root/arguments"
if [ "$count" -eq 1 ]; then
    printf '{"pending_oneshots":[{"remaining_seconds":0.01}]}' > "$ONEX_CENSUS_PLAN_OUT"
else
    printf '{"pending_oneshots":[]}' > "$ONEX_CENSUS_PLAN_OUT"
fi
printf '{"sample":%s}\\n' "$count"
if [ "$count" -eq 1 ]; then
    exit """
        + str(first_exit)
        + """
fi
""",
        encoding="utf-8",
    )
    contract = (
        _REPO / "src/omnibase_infra/nodes/node_census_oneshot_effect/contract.yaml"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnibase_infra.nodes.node_census_oneshot_effect",
            str(tmp_path),
            str(contract),
            "--memory",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == first_exit, result.stderr
    assert json.loads(result.stdout) == {"sample": 2}
    assert (tmp_path / "samples").read_text().strip() == "2"
    assert (tmp_path / "arguments").read_text().splitlines() == [
        "--memory --json",
        "--json",
    ]
    assert "automatically sampling again" in result.stderr
