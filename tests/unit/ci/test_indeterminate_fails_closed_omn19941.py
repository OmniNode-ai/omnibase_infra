# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19941: INDETERMINATE mandatory checks never authorize a PR-head receipt."""

import pytest

from scripts.ci import lab_pass_receipt as receipts
from tests.unit.ci.test_pr_head_lab_receipt_omn19566 import _receipt, _verify

pytestmark = pytest.mark.unit


def test_indeterminate_mandatory_check_with_fail_result_is_refused() -> None:
    check = receipts.ModelLabPassCheck.indeterminate_check("unit", "run undecidable")
    receipt = _receipt(checks=(check,))
    assert check.outcome is receipts.EnumLabPassCheckOutcome.INDETERMINATE
    assert receipt.result is receipts.EnumLabPassResult.FAIL

    verdict, _ = _verify(receipt)

    assert verdict is receipts.EnumPrHeadVerdict.RESULT_NOT_PASS
    assert verdict is not receipts.EnumPrHeadVerdict.ACCEPTED


def test_indeterminate_mandatory_check_with_forced_pass_result_is_refused() -> None:
    check = receipts.ModelLabPassCheck.indeterminate_check("unit", "run undecidable")
    receipt = _receipt(checks=(check,))
    # Frozen dataclass equivalent of model_construct: bypass validation to
    # prove the verifier independently refuses an inconsistent PASS receipt.
    object.__setattr__(receipt, "result", receipts.EnumLabPassResult.PASS)

    verdict, reason = _verify(receipt)

    assert verdict is receipts.EnumPrHeadVerdict.MISSING_MANDATORY_CHECK
    assert verdict is not receipts.EnumPrHeadVerdict.ACCEPTED
    assert "unit" in reason
