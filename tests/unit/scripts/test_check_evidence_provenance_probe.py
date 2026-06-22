# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-13030: evidence provenance-probe gate.

A deployed-SHA claim in a handoff/OCC/evidence doc must carry an adjacent
`docker exec <container> cat /app/build-provenance.json` probe citation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "check_evidence_provenance_probe.py"


def _load():
    spec = importlib.util.spec_from_file_location("evidence_probe", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
def test_claim_without_probe_is_rejected(tmp_path: Path) -> None:
    mod = _load()
    doc = tmp_path / "handoff.md"
    doc.write_text(
        "# Deploy\n\nThe runtime is now running sha abc1234 on the dev lane.\n",
        encoding="utf-8",
    )
    violations = mod.check_file(doc)
    assert violations
    assert "provenance-probe" in violations[0]


@pytest.mark.unit
def test_claim_with_adjacent_probe_passes(tmp_path: Path) -> None:
    mod = _load()
    doc = tmp_path / "handoff.md"
    doc.write_text(
        "# Deploy\n\n"
        "Deployed sha abc1234 to dev.\n\n"
        "```bash\n"
        "docker exec omninode-runtime cat /app/build-provenance.json\n"
        "```\n",
        encoding="utf-8",
    )
    assert mod.check_file(doc) == []


@pytest.mark.unit
def test_doc_without_any_claim_passes(tmp_path: Path) -> None:
    mod = _load()
    doc = tmp_path / "notes.md"
    doc.write_text("# Notes\n\nThis doc makes no deploy claims.\n", encoding="utf-8")
    assert mod.check_file(doc) == []


@pytest.mark.unit
def test_remote_ssh_docker_exec_probe_accepted(tmp_path: Path) -> None:
    mod = _load()
    doc = tmp_path / "handoff.md"
    doc.write_text(
        "Promoted revision deadbeef.\n\n"
        "ssh jonah@host docker exec runtime cat /app/build-provenance.json\n",
        encoding="utf-8",
    )
    assert mod.check_file(doc) == []
