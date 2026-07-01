# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Contract-sourced topic resolution tests for the emit-daemon topics module (OMN-13700).

Verifies that ``emit_daemon/topics.py`` holds no raw ONEX topic string literals and
that every topic constant resolves to its canonical contract source:

  * ``TOPIC_PHASE_METRICS`` / ``TOPIC_NOTIFICATION_BLOCKED`` /
    ``TOPIC_NOTIFICATION_COMPLETED`` resolve from the ``SUFFIX_OMNICLAUDE_*``
    constants in ``platform_topic_suffixes.py``.
  * ``TOPIC_BASELINES_COMPUTED`` resolves from
    ``EnumOmnibaseInfraTopic.EVT_BASELINES_COMPUTED_V1``.
  * ``TOPIC_DEPLOY_REBUILD_REQUESTED`` is deleted (zero consumers).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from omnibase_infra.enums.generated import EnumOmnibaseInfraTopic
from omnibase_infra.runtime.emit_daemon import topics as emit_topics
from omnibase_infra.topics.platform_topic_suffixes import (
    SUFFIX_OMNICLAUDE_NOTIFICATION_BLOCKED,
    SUFFIX_OMNICLAUDE_NOTIFICATION_COMPLETED,
    SUFFIX_OMNICLAUDE_PHASE_METRICS,
)

pytestmark = pytest.mark.unit

# Full ONEX topic literal: onex.(evt|cmd).<producer>.<event>.v<n>
_TOPIC_LITERAL = re.compile(r"^onex\.(evt|cmd)\.[a-z][a-z0-9._-]*$")


def test_no_raw_topic_literals_in_emit_daemon_topics() -> None:
    """emit_daemon/topics.py must contain zero raw ONEX topic string literals."""
    source = Path(emit_topics.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and _TOPIC_LITERAL.match(node.value.strip())
    ]
    assert literals == [], f"raw topic literals must be removed: {literals}"


def test_phase_metrics_resolves_from_canonical_suffix() -> None:
    """TOPIC_PHASE_METRICS is sourced from the canonical omniclaude suffix."""
    assert emit_topics.TOPIC_PHASE_METRICS == SUFFIX_OMNICLAUDE_PHASE_METRICS
    assert emit_topics.TOPIC_PHASE_METRICS == "onex.evt.omniclaude.phase-metrics.v1"


def test_notification_topics_resolve_from_canonical_suffixes() -> None:
    """Notification topics are sourced from the canonical omniclaude suffixes."""
    assert (
        emit_topics.TOPIC_NOTIFICATION_BLOCKED == SUFFIX_OMNICLAUDE_NOTIFICATION_BLOCKED
    )
    assert (
        emit_topics.TOPIC_NOTIFICATION_COMPLETED
        == SUFFIX_OMNICLAUDE_NOTIFICATION_COMPLETED
    )
    assert (
        emit_topics.TOPIC_NOTIFICATION_BLOCKED
        == "onex.evt.omniclaude.notification-blocked.v1"
    )
    assert (
        emit_topics.TOPIC_NOTIFICATION_COMPLETED
        == "onex.evt.omniclaude.notification-completed.v1"
    )


def test_baselines_computed_resolves_from_enum() -> None:
    """TOPIC_BASELINES_COMPUTED is sourced from the generated topic enum."""
    assert (
        EnumOmnibaseInfraTopic.EVT_BASELINES_COMPUTED_V1.value
        == emit_topics.TOPIC_BASELINES_COMPUTED
    )
    assert (
        emit_topics.TOPIC_BASELINES_COMPUTED
        == "onex.evt.omnibase-infra.baselines-computed.v1"
    )


def test_deploy_rebuild_requested_constant_removed() -> None:
    """The unused TOPIC_DEPLOY_REBUILD_REQUESTED constant is deleted (zero consumers)."""
    assert not hasattr(emit_topics, "TOPIC_DEPLOY_REBUILD_REQUESTED")
    assert "TOPIC_DEPLOY_REBUILD_REQUESTED" not in emit_topics.__all__


def test_registrations_remain_wired_to_resolved_constants() -> None:
    """ModelEventRegistration objects stay wired to the resolved topic constants."""
    assert (
        emit_topics.PHASE_METRICS_REGISTRATION.topic_template
        == emit_topics.TOPIC_PHASE_METRICS
    )
    assert (
        emit_topics.BASELINES_COMPUTED_REGISTRATION.topic_template
        == emit_topics.TOPIC_BASELINES_COMPUTED
    )
    assert (
        emit_topics.TCB_OUTCOME_REGISTRATION.topic_template
        == emit_topics.TOPIC_PHASE_METRICS
    )
