# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The contracts_and_di literal_path rule finds nothing in the two files this slice cleared (OMN-21623)."""

from pathlib import Path

import pytest

from omnibase_core.validation.validator_contracts_and_di import (
    ValidatorContractsAndDi,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

CLEARED_FILES = (
    "scripts/deploy-agent/deploy_agent/unit_drift.py",
    "scripts/validation/check_claudemd_references.py",
)


def _literal_path_issues(path: Path) -> list[str]:
    result = ValidatorContractsAndDi().validate_file(path)
    return [issue.message for issue in result.issues if "literal_path" in str(issue)]


def test_rule_flags_a_planted_path_literal(tmp_path: Path) -> None:
    """Positive control: the same call reports a literal path, so a zero below means clean."""
    planted = tmp_path / "planted.py"
    planted.write_text('STATE_FILE = "/var/lib/example/state.json"\n', encoding="utf-8")
    assert len(_literal_path_issues(planted)) == 1


@pytest.mark.parametrize("relative", CLEARED_FILES)
def test_cleared_file_has_no_literal_path_finding(relative: str) -> None:
    path = REPO_ROOT / relative
    assert path.is_file()
    assert _literal_path_issues(path) == []
