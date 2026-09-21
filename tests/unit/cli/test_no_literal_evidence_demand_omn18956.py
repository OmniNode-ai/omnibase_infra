# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""No call site arms the evidence refusal with a literal (OMN-18956 residual).

WHY THIS EXISTS, AND IT IS AN ADMISSION. The first OMN-18956 fix moved the
VALIDATOR's two flags off literals and left an identical pair on the receipt
CALLBACK, which calls `_write_local_run_files` and re-runs the same refusal.
Both the unit module and the integration module passed, the PR merged, and the
very next delegate on the lane failed with the same message it always had.

The integration module's own docstring says a fix proven at the leaf can still
be wrong at the entry the CLI wires. It was right, and it still missed this,
because it drove the `receipt_validator` and the surviving literal was on the
`receipt_callback` beside it. Two entry points, one refusal, one of them
tested.

So this asserts the property rather than either path: the module contains no
literal `True` for either flag anywhere. It reads the source, so a third call
site added tomorrow is caught on the day it is written rather than on the day
a lane run disagrees with its own terminal.

Reading source text is usually a poor substitute for behaviour. Here it is the
right instrument, because the defect is precisely a call site nobody thought
to exercise, and a behavioural test can only cover the paths its author
already knows about -- which is the whole reason this residual shipped.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_MODULE = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "cli"
    / "cli_delegate.py"
)

#: The two keywords that arm the completed-terminal evidence refusal.
_EVIDENCE_FLAGS = frozenset({"require_budget_evidence", "require_contract_evidence"})


def _literal_true_demands() -> list[str]:
    """Return every call that hard-codes one of the flags to True."""
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg not in _EVIDENCE_FLAGS:
                continue
            value = keyword.value
            if isinstance(value, ast.Constant) and value.value is True:
                offenders.append(f"{keyword.arg} at line {keyword.value.lineno}")
    return offenders


def test_the_parser_finds_the_flags_at_all() -> None:
    """Positive control: an empty scan would make the next test vacuous."""
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    seen = {
        keyword.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg in _EVIDENCE_FLAGS
    }
    assert seen == _EVIDENCE_FLAGS, sorted(seen)


def test_no_call_site_demands_evidence_unconditionally() -> None:
    offenders = _literal_true_demands()
    assert not offenders, (
        "a call site arms the completed-terminal evidence refusal with a "
        "literal True, so it demands evidence no request asked for: "
        + "; ".join(offenders)
    )
