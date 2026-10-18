# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: the autoclose diagnose step probes no change-control repository.

The ``diagnose`` step of ``evidence-autoclose-sweep.yml`` lists the repositories
whose pull requests the sweep's credential must be able to read, and calls
``gh api`` on each. Its list named ``onex_change_control``, a repository this
repository is being taken off. The list holds only repositories the sweep reads
a pull request from today; the sweep's own checkout of the change-control
contracts stays until companions stop being minted, so it is not asserted here.

Falsifier: restore ``OmniNode-ai/onex_change_control`` to the probe list and
``test_probe_list_names_no_change_control_repo`` names it.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/evidence-autoclose-sweep.yml"

_PROBE_LOOP = re.compile(
    r"echo \"== credential capability ==\"\n"
    r"\s*for repo in \\\n(?P<repos>(?:\s*OmniNode-ai/\S+(?: \\)?\n)+)\s*do\n"
)


def _probe_repos() -> list[str]:
    match = _PROBE_LOOP.search(WORKFLOW.read_text(encoding="utf-8"))
    assert match, "credential capability loop not found in the diagnose step"
    return [
        token
        for token in match.group("repos").replace("\\", " ").split()
        if token.startswith("OmniNode-ai/")
    ]


def test_probe_list_names_no_change_control_repo() -> None:
    repos = _probe_repos()
    assert not [r for r in repos if "change_control" in r or "change-control" in r], (
        f"diagnose probes a change-control repository: {repos}"
    )


def test_probe_list_keeps_the_repositories_the_sweep_reads() -> None:
    """Positive control: the regex found the loop and the other probes stay."""
    repos = _probe_repos()
    assert len(repos) == 4
    assert {
        "OmniNode-ai/omnibase_infra",
        "OmniNode-ai/omnimarket",
        "OmniNode-ai/omnibase_core",
    } <= set(repos)


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/autoclose_diagnose_probe_repos_omn17427.json"
)
def test_probe_list_holds_the_repositories_recorded_from_the_canonical_clones(
    recorded_response: dict[str, Any],
) -> None:
    """The probe list keeps every recorded repository and adds no unrecorded public one.

    The recording holds the default-branch head of three fleet repositories, read
    from their canonical clones. Each is a repository the sweep reads pull
    requests from, so each must stay in the list; the one repository not
    recorded here is private and is covered by the count in the test above.
    """
    recording = recorded_response["response"]
    lines = "".join(
        f"{repo}\t{row['default_branch']}\t{row['head']}\n"
        for repo, row in recording.items()
    )
    assert (
        hashlib.sha256(lines.encode("utf-8")).hexdigest()
        == recorded_response["_provenance"]["source_sha256"]
    )
    assert set(recording) <= set(_probe_repos())
    assert all(len(row["head"]) == 40 for row in recording.values())
