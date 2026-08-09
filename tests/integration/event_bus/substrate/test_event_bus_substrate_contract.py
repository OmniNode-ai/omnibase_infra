# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Collects the shared ``event_bus_substrate`` contract tests (OMN-15789).

Identical mechanism to ``omnibase_core``'s
``tests/unit/event_bus/test_event_bus_substrate_contract.py``: the test
bodies live in
``omnibase_core.event_bus.testing.contract_event_bus_substrate`` and are
star-imported here so pytest collects and runs the EXACT SAME functions
against THIS package's fixtures (``conftest.py`` in this directory), which
add the ``real_broker`` param. No assertion logic is duplicated -- see that
module's docstring for the full AC3/AC6 rationale.

``real_broker``-parametrized runs are ``pytest.mark.integration`` +
``pytest.mark.kafka``, skipped unless ``KAFKA_INTEGRATION_TESTS=1`` (see
this directory's ``conftest.py``).
"""

from __future__ import annotations

from omnibase_core.event_bus.testing.contract_event_bus_substrate import (
    test_auto_offset_reset_earliest_replays_retained_history,
    test_auto_offset_reset_latest_delivers_only_future_messages,
    test_group_join_gates_delivery_only_while_joined,
    test_publish_subscribe_seam_matches_real_consumer_group_derivation,
    test_rebalance_window_drops_uncommitted_inflight_message,
    test_rejoin_resumes_from_committed_offset_ignoring_auto_offset_reset,
)
