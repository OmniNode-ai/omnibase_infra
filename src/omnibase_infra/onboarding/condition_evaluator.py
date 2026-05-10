# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Condition evaluator for onboarding step visibility — OMN-10779.

Evaluates boolean guard expressions against a state dict without eval().
Supported operators: ==, in [...], not in [...], in <state_key>, and.

LHS may be either a bareword (resolved as a state key) or a quoted string
literal (e.g. ``"llm_inference" in selected_local_services``). RHS supports
quoted literals for ``==``, inline lists for ``in [..]`` / ``not in [..]``,
and bareword state keys resolving to collections.
"""

from __future__ import annotations

import re

# A LHS token is either a bareword (\w+) or a single/double-quoted string.
_LHS_TOKEN = r"""(?:"[^"]*"|'[^']*'|\w+)"""

_NOT_IN_RE = re.compile(rf"^({_LHS_TOKEN})\s+not\s+in\s+(.+)$")
_IN_RE = re.compile(rf"^({_LHS_TOKEN})\s+in\s+(.+)$")
_EQ_RE = re.compile(rf"^({_LHS_TOKEN})\s*==\s*(.+)$")


class ConditionEvaluationError(Exception):
    """Raised when a condition references an unknown state key or is malformed."""


def _resolve_key(key: str, state: dict[str, object]) -> object:
    if key not in state:
        msg = f"Unknown state key: '{key}'"
        raise ConditionEvaluationError(msg)
    return state[key]


def _is_quoted(token: str) -> bool:
    token = token.strip()
    return len(token) >= 2 and (
        (token[0] == '"' and token[-1] == '"') or (token[0] == "'" and token[-1] == "'")
    )


def _resolve_lhs(token: str, state: dict[str, object]) -> str:
    """Resolve a LHS token to its string value.

    Quoted tokens (``"foo"`` / ``'foo'``) are literal values; bareword tokens
    are resolved as state keys.
    """
    token = token.strip()
    if _is_quoted(token):
        return token[1:-1]
    return str(_resolve_key(token, state))


def _parse_inline_list(raw: str) -> list[str]:
    """Parse '[a, b, c]' into ['a', 'b', 'c']."""
    inner = raw.strip()
    if not (inner.startswith("[") and inner.endswith("]")):
        msg = f"Expected inline list, got: {inner!r}"
        raise ConditionEvaluationError(msg)
    items = inner[1:-1].split(",")
    return [item.strip().strip('"').strip("'") for item in items if item.strip()]


def _strip_quotes(value: str) -> str:
    return value.strip().strip('"').strip("'")


def _split_and_clauses(expr: str) -> list[str]:
    """Split an expression on whitespace-bounded ``and`` outside quotes."""
    clauses: list[str] = []
    start = 0
    quote: str | None = None
    index = 0

    while index < len(expr):
        char = expr[index]
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            index += 1
            continue

        if quote is None and expr.startswith("and", index):
            before_is_space = index > 0 and expr[index - 1].isspace()
            after_index = index + len("and")
            after_is_space = after_index < len(expr) and expr[after_index].isspace()
            if before_is_space and after_is_space:
                clauses.append(expr[start:index].strip())
                start = after_index
                index = after_index
                continue

        index += 1

    clauses.append(expr[start:].strip())
    return [clause for clause in clauses if clause]


def _evaluate_single(expr: str, state: dict[str, object]) -> bool:
    expr = expr.strip()

    not_in_match = _NOT_IN_RE.match(expr)
    if not_in_match:
        lhs_token = not_in_match.group(1)
        rhs_raw = not_in_match.group(2).strip()
        lhs_val = _resolve_lhs(lhs_token, state)
        if rhs_raw.startswith("["):
            items = _parse_inline_list(rhs_raw)
        else:
            rhs_key = rhs_raw
            rhs_obj = _resolve_key(rhs_key, state)
            if not isinstance(rhs_obj, (list, tuple, set)):
                msg = f"State key '{rhs_key}' is not a collection"
                raise ConditionEvaluationError(msg)
            items = [str(x) for x in rhs_obj]
        return lhs_val not in items

    in_match = _IN_RE.match(expr)
    if in_match:
        lhs_token = in_match.group(1)
        rhs_raw = in_match.group(2).strip()
        lhs_val = _resolve_lhs(lhs_token, state)
        if rhs_raw.startswith("["):
            items = _parse_inline_list(rhs_raw)
        else:
            rhs_key = rhs_raw
            rhs_obj = _resolve_key(rhs_key, state)
            if not isinstance(rhs_obj, (list, tuple, set)):
                msg = f"State key '{rhs_key}' is not a collection"
                raise ConditionEvaluationError(msg)
            items = [str(x) for x in rhs_obj]
        return lhs_val in items

    eq_match = _EQ_RE.match(expr)
    if eq_match:
        lhs_token = eq_match.group(1)
        rhs_val = _strip_quotes(eq_match.group(2))
        lhs_val = _resolve_lhs(lhs_token, state)
        return lhs_val == rhs_val

    msg = f"Unrecognised condition syntax: {expr!r}"
    raise ConditionEvaluationError(msg)


def evaluate_condition(expr: str | None, state: dict[str, object]) -> bool:
    """Evaluate a guard expression against state.

    Args:
        expr: Condition string or None. None always returns True.
        state: Current onboarding state dict.

    Returns:
        True if the condition holds, False otherwise.

    Raises:
        ConditionEvaluationError: If a referenced state key is unknown or the
            expression syntax is unrecognised.
    """
    if expr is None:
        return True

    clauses = _split_and_clauses(expr)
    return all(_evaluate_single(clause, state) for clause in clauses)


__all__ = ["ConditionEvaluationError", "evaluate_condition"]
