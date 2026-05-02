# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration guard for savings estimator runtime subscriptions."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SERVICE_KERNEL_PATH = Path("src/omnibase_infra/runtime/service_kernel.py")


def _load_service_kernel_ast() -> ast.Module:
    return ast.parse(SERVICE_KERNEL_PATH.read_text(encoding="utf-8"))


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


@pytest.mark.integration
def test_savings_estimator_subscribe_calls_include_group_id() -> None:
    """Savings estimator subscriptions must be consumer-group owned."""
    tree = _load_service_kernel_ast()
    subscribe_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node.func) == "subscribe"
    ]

    savings_subscribe_calls = [
        node
        for node in subscribe_calls
        if any(
            keyword.arg == "group_id"
            and isinstance(keyword.value, ast.JoinedStr)
            and any(
                isinstance(part, ast.Constant) and part.value == "savings-estimator."
                for part in keyword.value.values
            )
            for keyword in node.keywords
        )
    ]

    assert len(savings_subscribe_calls) == 1


@pytest.mark.integration
def test_savings_estimator_subscription_failures_are_warning_logged() -> None:
    """Subscription failures should be visible in runtime logs."""
    tree = _load_service_kernel_ast()
    warning_messages = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "warning"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]

    assert "Could not subscribe to %s for savings estimation" in warning_messages
