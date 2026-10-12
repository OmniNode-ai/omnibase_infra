# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay of the receipt that held omnibase_infra#4809 (OMN-19566).

The lab pool proved omnibase_infra#4809 (an omnimarket pin advance) PASS on
a lab host, and the verifier still refused the receipt as MISSING_MANDATORY_CHECK
for changed_path_live and forwarder_refused_topic: checks no pool run of a pin
advance can produce. The recording is that receipt, its bus event and the PR's
changed paths. Replayed through the workflow's own pin command
(``validate_lab_proof_profiles.py --pin --changed-files``), the PR's diff waives
exactly those two and the real receipt carries every check that is left; with no
diff, the incident reproduces.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = REPO_ROOT / "scripts" / "ci" / "validate_lab_proof_profiles.py"
INFRA = "OmniNode-ai/omnibase_infra"


def _pin(tmp_path: Path, changed: list[str] | None) -> dict[str, list[str]]:
    argv = [sys.executable, str(VALIDATOR), "--pin", INFRA, "runtime_image"]
    if changed is not None:
        listing = tmp_path / "changed.txt"
        listing.write_text("".join(f"{p}\n" for p in changed), encoding="utf-8")
        argv += ["--changed-files", str(listing)]
    done = subprocess.run(
        argv, cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return cast("dict[str, list[str]]", json.loads(done.stdout))


@pytest.mark.live_contact(
    "tests/ci/fixtures/lab_proof_receipt_omnibase_infra_4809_omn19566.json"
)
def test_the_held_pin_advance_receipt_carries_every_check_its_diff_demands(
    tmp_path: Path, recorded_response: dict[str, object]
) -> None:
    response = cast("dict[str, object]", recorded_response["response"])
    receipt = cast("dict[str, object]", response["receipt"])
    changed = cast("list[str]", response["changed_paths"])
    rows = cast("list[dict[str, object]]", receipt["checks"])
    passing = {str(row["name"]) for row in rows if row.get("ok", True) is True}
    assert receipt["result"] == "PASS"

    # the incident: judged with no diff, the two surface checks are missing
    unread = _pin(tmp_path, None)
    missing = sorted(set(unread["mandatory_checks"]) - passing)
    assert missing == response["missing_mandatory_checks"]
    assert response["verifier_token"] == "MISSING_MANDATORY_CHECK"

    # judged against its own diff, nothing the profile still demands is missing
    pinned = _pin(tmp_path, changed)
    assert sorted(pinned["inapplicable_checks"]) == [
        "changed_path_live",
        "forwarder_refused_topic",
    ]
    assert set(pinned["mandatory_checks"]) <= passing
