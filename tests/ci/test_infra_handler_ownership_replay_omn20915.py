# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The infra node handler allowlist holds no entry for a deleted handler (OMN-20915).

The recording is the real CI response of the ownership gate on the pull request
that deleted the moved and dead GitHub source-control nodes: it named sixteen
allowlist entries as stale. This test replays that recording against the
committed allowlist, so the entries named by the real gate cannot come back, and
adds the general rule the gate enforces: every allowlisted path exists.

The parity-red-on-base baseline entry and the hand-flip proof of the deleted
node_pr_state_projection_compute stay: the flip-bundle gate refuses a baseline
entry removed by deleting its proof (shrink-only), so only the allowlist and the
shape baseline lose their entries here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = ROOT / "scripts" / "ci" / "infra-node-allowlist.txt"


def _allowlisted_paths() -> list[str]:
    paths: list[str] = []
    for raw in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        entry = raw.split("#", 1)[0].strip()
        if entry:
            paths.append(entry)
    return paths


@pytest.mark.live_contact(
    "tests/ci/fixtures/infra_handler_ownership_gate_response_omn20915.json"
)
def test_the_entries_the_real_gate_named_stale_are_gone(
    recorded_response: dict[str, object],
) -> None:
    response = recorded_response["response"]
    assert isinstance(response, dict)
    stale = [
        line.removeprefix("  - ")
        for line in response["stderr_lines"]
        if line.startswith("  - ")
    ]
    assert len(stale) == 16, "the recording lost the entries it exists to replay"
    allowed = set(_allowlisted_paths())
    assert allowed, "the allowlist parsed to nothing, so the check below is vacuous"
    assert sorted(set(stale) & allowed) == []


def test_every_allowlisted_handler_path_exists() -> None:
    missing = [path for path in _allowlisted_paths() if not (ROOT / path).exists()]
    assert missing == []
