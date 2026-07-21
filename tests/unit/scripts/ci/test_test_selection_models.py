# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
import pytest
from pydantic import ValidationError

from scripts.ci.test_selection_models import (
    EnumFullSuiteReason,
    ModelTestSelection,
)

pytestmark = pytest.mark.unit


def test_full_suite_selection_serializes_with_reason() -> None:
    selection = ModelTestSelection(
        selected_paths=["tests/"],
        split_count=15,
        is_full_suite=True,
        full_suite_reason=EnumFullSuiteReason.SHARED_MODULE,
        matrix=list(range(1, 16)),
    )
    payload = selection.model_dump(mode="json")
    assert payload == {
        "selected_paths": ["tests/"],
        "split_count": 15,
        "is_full_suite": True,
        "full_suite_reason": "shared_module",
        "matrix": list(range(1, 16)),
    }


def test_smart_selection_disallows_full_suite_reason() -> None:
    with pytest.raises(ValidationError):
        ModelTestSelection(
            selected_paths=["tests/unit/nodes/"],
            split_count=2,
            is_full_suite=False,
            full_suite_reason=EnumFullSuiteReason.SHARED_MODULE,
            matrix=[1, 2],
        )


def test_matrix_length_matches_split_count() -> None:
    with pytest.raises(ValidationError):
        ModelTestSelection(
            selected_paths=["tests/unit/nodes/"],
            split_count=3,
            is_full_suite=False,
            full_suite_reason=None,
            matrix=[1, 2],  # length mismatch
        )


def test_split_count_bounded_by_15() -> None:
    """Infra max split count is 15, not 40."""
    with pytest.raises(ValidationError):
        ModelTestSelection(
            selected_paths=["tests/"],
            split_count=40,  # core uses 40; infra max is 15
            is_full_suite=True,
            full_suite_reason=EnumFullSuiteReason.MAIN_BRANCH,
            matrix=list(range(1, 41)),
        )


def test_smart_selection_no_reason() -> None:
    selection = ModelTestSelection(
        selected_paths=["tests/unit/cli/"],
        split_count=1,
        is_full_suite=False,
        full_suite_reason=None,
        matrix=[1],
    )
    assert selection.split_count == 1
    assert selection.matrix == [1]
    assert selection.full_suite_reason is None


def test_empty_selected_paths_is_valid_for_docs_only_exemption() -> None:
    """OMN-14753: selected_paths=[] is a legitimate "nothing to run" selection.

    Previously `min_length=1` forced every non-full-suite selection to name at
    least one test directory, which made the docs-only exemption impossible
    to express -- the selector had no way to say "provably zero impact."
    """
    selection = ModelTestSelection(
        selected_paths=[],
        split_count=1,
        is_full_suite=False,
        full_suite_reason=None,
        matrix=[1],
    )
    assert selection.selected_paths == []
    assert selection.is_full_suite is False
