# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""GitFixVerifier stays within the ONEX Validators pattern limit (OMN-17427).

``scripts/validate.py all`` runs the Patterns validator, which refuses a class
with more than 10 methods. GitFixVerifier had 11, so ``ONEX Validators`` failed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from omnibase_core.validation.checker_generic_pattern import GenericPatternChecker

_SRC = (
    Path(__file__).resolve().parents[2]
    / "src/omnibase_infra/handlers/handler_git_fix_verifier.py"
)


@pytest.mark.unit
def test_git_fix_verifier_has_no_more_methods_than_the_patterns_validator_allows() -> (
    None
):
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "GitFixVerifier"
    )
    checker = GenericPatternChecker(str(_SRC))
    checker.visit(cls)
    assert not checker.issues, checker.issues
