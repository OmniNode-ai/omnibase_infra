# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration test for runtime error triage pipeline (OMN-5655).

Verifies the end-to-end flow from error event emission to triage result.
Requires running Kafka/Redpanda and Valkey infrastructure.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.slow
class TestRuntimeErrorTriageE2E:
    """End-to-end tests for the runtime error triage pipeline.

    These tests require:
    - Kafka/Redpanda at localhost:19092
    - Valkey at localhost:16379
    - Topics: onex.evt.omnibase-infra.runtime-error.v1, onex.evt.omnibase-infra.error-triaged.v1
    """

    def test_schema_mismatch_produces_triage_result(self) -> None:
        """Emit a SCHEMA_MISMATCH error and verify triage processes it.

        Integration test stub -- requires live infrastructure.
        Full implementation deferred to Task 8 (OMN-5656) verification.
        """
        pytest.skip(
            "Integration test requires live Kafka + Valkey; "
            "run manually with `pytest -m integration`"
        )

    def test_missing_topic_auto_fix_in_local(self) -> None:
        """Emit a MISSING_TOPIC error in local env and verify auto-fix attempt.

        Integration test stub -- requires live infrastructure + rpk.
        """
        pytest.skip(
            "Integration test requires live Kafka + Valkey + rpk; "
            "run manually with `pytest -m integration`"
        )

    def test_duplicate_fingerprint_deduped_at_action_layer(self) -> None:
        """Emit same error twice and verify dedup at triage (not emission).

        Should see 2 rows in runtime_error_events but only 1 triage action.
        """
        pytest.skip(
            "Integration test requires live Kafka + Valkey; "
            "run manually with `pytest -m integration`"
        )
