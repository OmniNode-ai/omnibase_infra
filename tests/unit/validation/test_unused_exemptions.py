# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Prove unused exemptions fail against raw findings from the correct validator."""

from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_infra.validation import infra_validators as validators
from scripts import validate

pytestmark = pytest.mark.unit

FINDING = "sample.py:10: Function 'handle' has 6 parameters"
USED: validators.ExemptionPattern = {
    "file_pattern": r"sample\.py",
    "method_pattern": "Function 'handle'",
    "violation_pattern": r"has \d+ parameters",
}
UNUSED: validators.ExemptionPattern = {
    "file_pattern": r"missing\.py",
    "violation_pattern": r"has \d+ parameters",
}


def test_reports_only_unused_entry() -> None:
    assert validators.find_unused_exemptions([FINDING], [USED, UNUSED]) == [UNUSED]


def test_all_used_and_overlapping_entries_pass() -> None:
    overlapping: validators.ExemptionPattern = {"file_pattern": r"sample\.py"}
    assert validators.find_unused_exemptions([FINDING], [USED, overlapping]) == []


@pytest.mark.parametrize(
    "pattern",
    [
        {**USED, "class_pattern": "Class 'Missing'"},
        {**USED, "method_pattern": "Function 'missing'"},
        {**USED, "violation_pattern": "has 8 parameters"},
    ],
)
def test_every_regex_field_must_match(pattern: validators.ExemptionPattern) -> None:
    assert validators.find_unused_exemptions([FINDING], [pattern]) == [pattern]
    assert validators._filter_exempted_errors([FINDING], [pattern]) == [FINDING]


@pytest.mark.parametrize(
    "section", ["pattern_exemptions", "architecture_exemptions", "union_exemptions"]
)
@pytest.mark.parametrize("stale", [False, True])
def test_gate_and_cli_use_unfiltered_findings(section: str, stale: bool) -> None:
    exemptions = {
        "pattern_exemptions": [USED],
        "architecture_exemptions": [USED],
        "union_exemptions": [USED],
    }
    if stale:
        exemptions[section].append(UNUSED)
    raw = validators.ModelValidationResult[None](is_valid=False, errors=[FINDING])
    with (
        patch.object(validators, "_load_exemptions_yaml", return_value=exemptions),
        patch.object(
            validators, "validate_infra_patterns", return_value=raw
        ) as patterns,
        patch.object(
            validators, "validate_infra_architecture", return_value=raw
        ) as arch,
        patch.object(
            validators, "validate_infra_union_usage", return_value=raw
        ) as unions,
    ):
        result = validators.validate_infra_unused_exemptions("synthetic")
        for validator in (patterns, arch, unions):
            validator.assert_called_once_with("synthetic", apply_exemptions=False)
        assert result.is_valid is not stale
        if stale:
            assert len(result.errors) == 1
            assert f"{section}[2]" in result.errors[0]
            assert "missing" in result.errors[0]
        else:
            assert result.errors == []
        with patch.object(
            validators, "validate_infra_unused_exemptions", return_value=result
        ):
            assert validate.run_unused_exemptions() is not stale


def test_findings_do_not_cross_validator_sections() -> None:
    empty = validators.ModelValidationResult[None](is_valid=True, errors=[])
    raw = validators.ModelValidationResult[None](is_valid=False, errors=[FINDING])
    with (
        patch.object(
            validators,
            "_load_exemptions_yaml",
            return_value={
                "pattern_exemptions": [],
                "architecture_exemptions": [USED],
                "union_exemptions": [],
            },
        ),
        patch.object(validators, "validate_infra_patterns", return_value=raw),
        patch.object(validators, "validate_infra_architecture", return_value=empty),
        patch.object(validators, "validate_infra_union_usage", return_value=empty),
    ):
        result = validators.validate_infra_unused_exemptions()
    assert not result.is_valid
    assert "architecture_exemptions[1]" in result.errors[0]


@pytest.mark.parametrize("kind", ["architecture", "patterns"])
def test_validator_can_return_raw_or_filtered_findings(kind: str) -> None:
    raw = validators.ModelValidationResult[None](is_valid=False, errors=[FINDING])
    function = (
        validators.validate_infra_architecture
        if kind == "architecture"
        else validators.validate_infra_patterns
    )
    getter = (
        "get_architecture_exemptions"
        if kind == "architecture"
        else "get_pattern_exemptions"
    )
    with (
        patch.object(validators, f"validate_{kind}", return_value=raw),
        patch.object(validators, getter, return_value=[USED]),
    ):
        assert function("synthetic").errors == []
        assert function("synthetic", apply_exemptions=False).errors == [FINDING]


def test_union_validator_can_return_raw_or_filtered_findings(tmp_path: Path) -> None:
    with (
        patch.object(
            validators,
            "_count_non_optional_unions",
            return_value=(0, 0, 0, 0, [FINDING]),
        ),
        patch.object(validators, "get_union_exemptions", return_value=[USED]),
    ):
        assert validators.validate_infra_union_usage(tmp_path).errors == []
        assert validators.validate_infra_union_usage(
            tmp_path, apply_exemptions=False
        ).errors == [FINDING]


@pytest.mark.parametrize("stale", [False, True])
def test_cli_exit_code(stale: bool) -> None:
    result = validators.ModelValidationResult[None](
        is_valid=not stale, errors=["Unused planted entry"] if stale else []
    )
    with (
        patch.object(
            validators, "validate_infra_unused_exemptions", return_value=result
        ),
        patch("sys.argv", ["validate.py", "unused_exemptions"]),
    ):
        assert validate.main() == (1 if stale else 0)
