# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""End-to-end Kafka -> Postgres test for the build_loop projection chain.

Mirrors the canonical `tests/integration/registration/e2e/test_runtime_e2e.py`
pattern: publish a synthetic terminal event to Redpanda, then poll
public.build_loop_runs until a row appears (or 30s elapses). Asserts the row's
fields match the publish input.

Skipped unless BOTH:
    - KAFKA_INTEGRATION_TESTS=1 (explicit opt-in)
    - INTEGRATION_POSTGRES_HOST is set (real Postgres available)

This is the OMN-9774 / Wave 2.5 headline proof gate (CP-E2E-Kafka), driven
locally by this worker's pre-push pytest run and verified on .201 by W2.5b.

Ticket: OMN-9774
Plan: docs/plans/2026-04-25-overnight-p0-integration-plan.md (Wave 2.5)
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from tests.helpers.service_env import require_service_env

# OMN-18795: this suite is SELECTED BY MARKER, not gated by three silent skipifs.
#
# It was registered in config/env_gated_test_skips.yaml as having no execution
# path, and the reason recorded there UNDERSTATED the dependency: it said a
# broker, a Postgres and the opt-in. In fact this test publishes a synthetic
# terminal event and then polls for a projection row, so it also needs a
# RUNNING ONEX RUNTIME to consume the event and write that row -- as its own
# failure message says. Kafka and Postgres alone cannot make it pass.
#
# Its home is therefore reusable-runtime-boot.yml, the one workflow that boots
# a runtime. It is deselected from the PR test splits by `not kafka`. Once a
# job has SELECTED it, a missing opt-in is a red failure, not a skip.
INTEGRATION_POSTGRES_HOST = os.getenv("INTEGRATION_POSTGRES_HOST")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.kafka,
    pytest.mark.postgres,
]


@pytest.fixture(autouse=True, scope="module")
def _require_runtime_stack() -> None:
    """Refuse to skip silently once CI has selected this suite."""
    require_service_env(
        opt_in="KAFKA_INTEGRATION_TESTS",
        endpoint="KAFKA_BOOTSTRAP_SERVERS",
        workflow=".github/workflows/reusable-runtime-boot.yml",
        service="a booted ONEX runtime with Kafka and Postgres",
    )
    if not INTEGRATION_POSTGRES_HOST:
        pytest.fail(
            "INTEGRATION_POSTGRES_HOST is unset, so this suite would have "
            "skipped silently after its opt-in was asserted. The job that owns "
            "it (reusable-runtime-boot.yml) provisions Postgres and exports "
            "this variable -- see OMN-18795.",
            pytrace=False,
        )


TERMINAL_TOPIC = "onex.evt.omnimarket.build-loop-orchestrator-completed.v1"


def _build_terminal_event(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "workflow_name": "build_loop",
        "event_type": "build-loop-orchestrator-completed",
        "terminal_event_at": datetime.now(UTC).isoformat(),
        "correlation_id": str(uuid4()),
        "outcome": "success",
    }


@pytest.mark.asyncio
async def test_runtime_consumes_build_loop_terminal_event() -> None:
    """Publish synthetic terminal event; assert build_loop_runs row appears.

    Steps:
        1. Publish one synthetic terminal event to Redpanda on
           onex.evt.omnimarket.build-loop-orchestrator-completed.v1.
        2. Poll public.build_loop_runs (≤30s) for a row matching the
           synthetic run_id.
        3. Assert workflow_name + event_type + payload.run_id round-tripped.

    The asyncpg + aiokafka imports are deferred so the module imports cleanly
    in CI even when integration extras are absent.
    """
    import asyncpg
    from aiokafka import AIOKafkaProducer

    run_id = f"e2e-build-loop-{uuid4().hex[:12]}"
    event_body = _build_terminal_event(run_id)

    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    await producer.start()
    try:
        await producer.send_and_wait(TERMINAL_TOPIC, event_body)
    finally:
        await producer.stop()

    pg_host = INTEGRATION_POSTGRES_HOST
    pg_port = int(os.getenv("INTEGRATION_POSTGRES_PORT", "5436"))
    pg_user = os.getenv("INTEGRATION_POSTGRES_USER", "postgres")
    pg_password = os.getenv("INTEGRATION_POSTGRES_PASSWORD") or os.getenv(
        "POSTGRES_PASSWORD"
    )
    pg_db = os.getenv("INTEGRATION_POSTGRES_DB", "omnibase_infra")

    conn = await asyncpg.connect(
        host=pg_host,
        port=pg_port,
        user=pg_user,
        password=pg_password,
        database=pg_db,
    )
    try:
        deadline = asyncio.get_event_loop().time() + 30.0
        row = None
        while asyncio.get_event_loop().time() < deadline:
            row = await conn.fetchrow(
                "SELECT id, run_id, workflow_name, event_type, payload "
                "FROM public.build_loop_runs WHERE run_id = $1 "
                "ORDER BY created_at DESC LIMIT 1",
                run_id,
            )
            if row is not None:
                break
            await asyncio.sleep(0.5)

        assert row is not None, (
            f"build_loop_runs row for run_id={run_id} did not appear within 30s. "
            "Check that node_build_loop_projection_compute is wired and consuming "
            "from the runtime container."
        )
        assert row["run_id"] == run_id
        assert row["workflow_name"] == "build_loop"
        assert row["event_type"] == "build-loop-orchestrator-completed"

        # Payload column is JSONB — asyncpg returns it as a JSON string.
        payload = (
            json.loads(row["payload"])
            if isinstance(row["payload"], str)
            else row["payload"]
        )
        assert payload.get("run_id") == run_id
    finally:
        await conn.close()
