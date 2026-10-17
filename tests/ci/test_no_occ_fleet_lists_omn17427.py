# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: omnibase_infra's fleet tooling no longer targets onex_change_control.

The change-control repository is being retired (operator RULING
2026-10-14T05:30:13Z: everything off it). These repo lists and defaults named it
as one more repository to clone, mirror, sweep, audit or tag:

* ``scripts/pull-all.sh`` and ``scripts/lib/merge_sweep_common.sh`` ``REPOS``
* ``scripts/audit-merge-method-on-main.sh`` and
  ``scripts/verify-merge-queue-no-bypass.sh`` ``REPOS``
* ``docker/runners/git-mirror-refresh.sh`` ``MIRROR_REPOS``
* ``docker/runners/runner-job-started.sh`` ``_C2B_REWRITE_REPOS`` default
* ``scripts/runtime_build/sibling_clone_manifest.sh`` ``SIBLING_EXTRA_TRACKED_REPOS``
* ``scripts/enforcement_parity_manifest.yaml`` repos and
  ``scripts/version-matrix.yaml`` packages

Falsifier: put the repository back in any one of them and the matching test names it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
OCC = "onex_change_control"


def _bash_array(relpath: str, name: str) -> list[str]:
    text = (ROOT / relpath).read_text(encoding="utf-8")
    if re.search(rf"^\s*{name}=\(\)", text, re.MULTILINE):
        return []
    match = re.search(rf"^\s*{name}=\(\n(.*?)^\s*\)", text, re.MULTILINE | re.DOTALL)
    assert match, f"{relpath}: array {name} not found"
    items: list[str] = []
    for line in match.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            items.extend(tok.strip('"') for tok in line.split())
    return items


def _bash_default(relpath: str, pattern: str) -> str:
    text = (ROOT / relpath).read_text(encoding="utf-8")
    match = re.search(pattern, text, re.MULTILINE)
    assert match, f"{relpath}: {pattern} not found"
    return match.group(1)


@pytest.mark.parametrize(
    ("relpath", "name"),
    [
        ("scripts/pull-all.sh", "REPOS"),
        ("scripts/lib/merge_sweep_common.sh", "REPOS"),
        ("scripts/audit-merge-method-on-main.sh", "REPOS"),
        ("scripts/verify-merge-queue-no-bypass.sh", "REPOS"),
        ("docker/runners/git-mirror-refresh.sh", "MIRROR_REPOS"),
        (
            "scripts/runtime_build/sibling_clone_manifest.sh",
            "SIBLING_EXTRA_TRACKED_REPOS",
        ),
    ],
)
def test_bash_repo_array_omits_change_control(relpath: str, name: str) -> None:
    assert OCC not in _bash_array(relpath, name), f"{relpath} {name} still names {OCC}"


@pytest.mark.parametrize(
    ("relpath", "pattern"),
    [
        (
            "docker/runners/runner-job-started.sh",
            r'^_C2B_REWRITE_REPOS="\$\{OMNI_GIT_MIRROR_REWRITE_REPOS:-([^}]*)\}"',
        ),
    ],
)
def test_repo_default_omits_change_control(relpath: str, pattern: str) -> None:
    value = _bash_default(relpath, pattern)
    assert OCC not in re.split(r"[,\s]+", value), f"{relpath} default still names {OCC}"


def test_enforcement_parity_manifest_omits_change_control() -> None:
    data = yaml.safe_load(
        (ROOT / "scripts/enforcement_parity_manifest.yaml").read_text(encoding="utf-8")
    )
    repos = data.get("repos", data)
    assert OCC not in repos


def test_version_matrix_omits_change_control() -> None:
    data = yaml.safe_load(
        (ROOT / "scripts/version-matrix.yaml").read_text(encoding="utf-8")
    )
    names = {entry["name"] for entry in data["repos"]}
    assert "onex-change-control" not in data["packages"]
    assert OCC not in names
