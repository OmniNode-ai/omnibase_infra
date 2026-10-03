# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20209 -- the chain probe reads a REAL broker's subscription footprint.

On the .201 dev lane the delegation orchestrator's groups are one base on
every topic its contract subscribes to, while a ledger projection is also
Stable on the command topic but consumes only some of them. A suffix match
would count the projection; the footprint does not.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid

import pytest

from omnibase_core.event_bus.util_consumer_group import TOPIC_SCOPE_INFIX
from omnibase_infra.backends.backend_probe import (
    live_chain_consumer_groups,
    live_consumer_groups,
)

from . import redpanda_sasl_harness as harness
from .test_omn18418_lane_probe_against_a_real_broker import (
    _JoinedConsumer,
    _seed_one_record,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    # The session broker's 120s readiness budget belongs inside this module's
    # watchdog too; without it a busy Docker host reports a watchdog fact,
    # rather than a footprint/probe fact.
    pytest.mark.timeout(600),
    pytest.mark.xdist_group("omn18012_redpanda_sasl"),
]


def _group_id(run: str, role: str, topic: str) -> str:
    """Return topic-scoped groups sharing one base for each role."""
    return f"omn20209.{run}.{role}.consume.1.0.0{TOPIC_SCOPE_INFIX}{topic}"


def test_only_a_base_joined_on_the_whole_footprint_answers_for_the_chain(
    kafka_auth_env: harness.RedpandaSasl,
) -> None:
    """A Stable projection on the command topic cannot answer for the chain."""
    broker = kafka_auth_env
    run = uuid.uuid4().hex[:8]
    cmd = f"omn20209.cmd.{run}"
    routing = f"omn20209.routing.{run}"
    inference = f"omn20209.inference.{run}"
    topics = (cmd, routing, inference)
    for topic in topics:
        broker.create_topic(topic)
        asyncio.run(_seed_one_record(topic, broker.bootstrap))
    chain_cmd_group = _group_id(run, "chain", cmd)
    projection_cmd_group = _group_id(run, "projection", cmd)

    with contextlib.ExitStack() as consumers:
        for topic in topics:
            consumers.enter_context(
                _JoinedConsumer(
                    topic=topic,
                    group_id=_group_id(run, "chain", topic),
                    bootstrap=broker.bootstrap,
                )
            )
        consumers.enter_context(
            _JoinedConsumer(
                topic=cmd, group_id=projection_cmd_group, bootstrap=broker.bootstrap
            )
        )
        assert live_chain_consumer_groups(
            command_topic=cmd,
            subscribe_topics=(cmd, routing, inference),
            bootstrap_servers=broker.bootstrap,
        ) == (chain_cmd_group,)

        # Discriminating control: both command groups are Stable, so the
        # projection was excluded by its footprint rather than a failed join.
        found = live_consumer_groups(topic=cmd, bootstrap_servers=broker.bootstrap)
        assert chain_cmd_group in found
        assert projection_cmd_group in found


def test_a_footprint_no_live_base_covers_is_an_empty_answer(
    kafka_auth_env: harness.RedpandaSasl,
) -> None:
    """Live command and routing groups cannot cover a missing inference group."""
    broker = kafka_auth_env
    run = uuid.uuid4().hex[:8]
    cmd = f"omn20209.cmd.{run}"
    routing = f"omn20209.routing.{run}"
    inference = f"omn20209.inference.{run}"
    for topic in (cmd, routing, inference):
        broker.create_topic(topic)
        asyncio.run(_seed_one_record(topic, broker.bootstrap))

    with contextlib.ExitStack() as consumers:
        for role, topic in (("projection", cmd), ("chain", cmd), ("chain", routing)):
            consumers.enter_context(
                _JoinedConsumer(
                    topic=topic,
                    group_id=_group_id(run, role, topic),
                    bootstrap=broker.bootstrap,
                )
            )
        assert (
            live_chain_consumer_groups(
                command_topic=cmd,
                subscribe_topics=(cmd, routing, inference),
                bootstrap_servers=broker.bootstrap,
            )
            == ()
        )
