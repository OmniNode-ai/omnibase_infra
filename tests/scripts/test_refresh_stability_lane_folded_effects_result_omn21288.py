# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Static result-contract coverage for folded effects verification (OMN-21288)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "refresh_stability_lane.sh"


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


@pytest.mark.unit
def test_folded_effects_result_is_documented_under_exit_two() -> None:
    header = _text().split("set -euo pipefail", 1)[0]
    exit_two = header.split("#   2  ", 1)[1].split("#   64 ", 1)[0]
    assert "FAILED_FOLDED_EFFECTS (OMN-21288)" in exit_two
    assert "health gate passed" in exit_two
    assert "no rollback" in exit_two
    assert "preflight tags kept" in exit_two
    assert "rollback anchors" in exit_two
    assert "log lines and receipt result" in exit_two


@pytest.mark.unit
def test_folded_effects_result_has_an_explicit_exit_two_arm() -> None:
    result_case = _text().split('case "${RESULT}" in', 1)[1].split("esac", 1)[0]
    assert "FAILED_FOLDED_EFFECTS) exit 2 ;;" in result_case
    assert result_case.index("FAILED_FOLDED_EFFECTS) exit 2 ;;") < result_case.index(
        "*) exit 2 ;;"
    )


@pytest.mark.unit
def test_preflight_tag_pruning_remains_guarded_by_success() -> None:
    match = re.search(
        r'    if \[\[ "\$\{RESULT\}" == SUCCESS \]\]; then\n'
        r"(?P<body>        # Prune old preflight tags.*?)(?=\n    fi)",
        _text(),
        re.DOTALL,
    )
    assert match is not None, "Preflight tags must only be pruned on SUCCESS."
    body = match.group("body")
    assert 'for svc in "${ROLLBACK_SERVICES[@]}"; do' in body
    assert "for t in ${old_tags}; do" in body
    assert 'docker rmi "${image_tag}:${t}"' in body
