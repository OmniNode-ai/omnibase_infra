# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21041 -- the flip proof under scripts/ci is the bytes the real gate verified.

The runner-fleet-maintain orchestrator and the runner health snapshot effect left
the frozen non-canonical handler-shape baseline with an adequacy receipt and a
hand-flip proof each. The omnibase_core ratchet and flip-bundle gate judged those
exact bytes at the pinned core ref; the recorded run below is that judgement,
captured with provenance. This test replays it against the committed files, so an
edit to a receipt, a proof or the baseline fails here until the real gate is run
again and the recording refreshed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_NODES = (
    "omnibase_infra.nodes.node_runner_fleet_maintain_orchestrator",
    "omnibase_infra.nodes.node_runner_health_snapshot_effect",
)


@pytest.mark.live_contact(
    "tests/ci/fixtures/runner_fleet_maintain_flip_gate_omn21041.json"
)
def test_flip_proof_is_the_bytes_the_real_gate_verified(
    recorded_response: dict[str, object],
) -> None:
    recorded = recorded_response["response"]
    assert isinstance(recorded, dict)

    assert "new=0 unproven_flips=0" in str(recorded["ratchet"])
    nodes = recorded["flip_bundle_nodes"]
    assert isinstance(nodes, dict)
    assert sorted(nodes) == sorted(_NODES)
    assert all("all seven flip-bundle assertions hold" in v for v in nodes.values())
    assert "2 flip bundle(s) coherent" in str(recorded["flip_bundle_total"])

    digests = recorded["artifact_sha256"]
    assert isinstance(digests, dict)
    for relative, digest in digests.items():
        actual = hashlib.sha256((_ROOT / relative).read_bytes()).hexdigest()
        assert actual == digest, f"{relative} changed after the real gate verified it"
    for node in _NODES:
        assert f"scripts/ci/adequacy_receipts/{node}.json" in digests
        assert f"scripts/ci/adequacy_receipts/{node}.handflip.json" in digests


def test_flipped_nodes_are_out_of_the_frozen_baseline() -> None:
    baseline = (_ROOT / "scripts/ci/canonical_handler_shape_baseline.py").read_text(
        encoding="utf-8"
    )
    for node in _NODES:
        assert f'"{node}"' not in baseline
