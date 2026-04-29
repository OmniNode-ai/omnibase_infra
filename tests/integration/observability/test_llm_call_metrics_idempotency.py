# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration-level proof for OMN-10332 LLM call metric idempotency."""

from __future__ import annotations

from tests.ci.test_llm_call_metrics_idempotency_migration import (
    test_idempotency_key_prevents_duplicate_durable_rows,
)


def test_duplicate_llm_call_metric_writes_collapse_to_one_durable_row() -> None:
    """The migration's unique key prevents duplicate logical metric rows."""
    test_idempotency_key_prevents_duplicate_durable_rows()
