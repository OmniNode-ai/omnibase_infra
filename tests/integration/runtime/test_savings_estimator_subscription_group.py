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
def test_savings_estimator_subscribe_calls_use_canonical_identity() -> None:
    """Savings subscriptions derive their groups from a typed node identity."""
    tree = _load_service_kernel_ast()
    savings_identity_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_savings_node_identity"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and _call_name(node.value.func) == "ModelNodeIdentity"
    ]
    assert len(savings_identity_assignments) == 1
    identity_call = savings_identity_assignments[0].value
    assert isinstance(identity_call, ast.Call)
    identity_keywords = {
        keyword.arg: keyword.value for keyword in identity_call.keywords
    }
    assert isinstance(identity_keywords["env"], ast.Name)
    assert identity_keywords["env"].id == "environment"
    assert isinstance(identity_keywords["node_name"], ast.Constant)
    assert identity_keywords["node_name"].value == "savings-estimator"
    assert "service" in identity_keywords
    assert "version" in identity_keywords

    savings_subscribe_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node.func) == "subscribe"
        if any(
            keyword.arg == "node_identity"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "_savings_node_identity"
            for keyword in node.keywords
        )
    ]

    assert len(savings_subscribe_calls) == 1
    assert all(
        keyword.arg != "group_id" for keyword in savings_subscribe_calls[0].keywords
    )


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
