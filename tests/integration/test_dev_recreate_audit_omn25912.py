# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read back the recreate audit built from the checkout's real policy and contract (OMN-25912).

The policy is the rendered ``docker/runtime-policy.env``, the settings are read from the
node's own ``contract.yaml``, and the events are the recorded 24 hour dev window. Only the
bus is absent: the handler is pure, so nothing shared is touched.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from omnibase_infra.enums.enum_dev_recreate_violation_kind import (
    EnumDevRecreateViolationKind as Kind,
)
from omnibase_infra.models.model_dev_container_observation import (
    ModelDevContainerObservation,
)
from omnibase_infra.models.model_dev_recreate_announcement import (
    ModelDevRecreateAnnouncement,
)
from omnibase_infra.models.model_dev_recreate_audit_request import (
    ModelDevRecreateAuditRequest,
)
from omnibase_infra.models.model_dev_recreate_policy import ModelDevRecreatePolicy
from omnibase_infra.models.model_dev_roll_manifest import ModelDevRollManifest
from omnibase_infra.models.model_dev_runtime_started import ModelDevRuntimeStarted
from omnibase_infra.nodes.node_dev_roll_events_effect.dev_recreate_audit import (
    HandlerDevRecreateAudit,
)
from omnibase_infra.nodes.node_dev_roll_events_effect.dev_roll_events import (
    audit_settings,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[2]
_NODE = _ROOT / "src" / "omnibase_infra" / "nodes" / "node_dev_roll_events_effect"
_FIXTURE = (
    _ROOT
    / "tests"
    / "unit"
    / "handlers"
    / "fixtures"
    / "recreate_audit_window_24h_omn25912.json"
)


def test_audit_from_rendered_policy_and_contract_flags_the_recorded_window() -> None:
    env: dict[str, str] = {}
    for line in (_ROOT / "docker" / "runtime-policy.env").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.startswith("#"):
            env[key] = value
    lookback, tolerance = audit_settings(_NODE / "contract.yaml")
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    request = ModelDevRecreateAuditRequest(
        policy=ModelDevRecreatePolicy.from_env(env),
        announcement_lookback_seconds=lookback,
        start_match_tolerance_seconds=tolerance,
        starts=tuple(
            ModelDevRuntimeStarted.model_validate(s) for s in fixture["starts"]
        ),
        manifests=tuple(
            ModelDevRollManifest.model_validate(m) for m in fixture["manifests"]
        ),
        announcements=tuple(
            ModelDevRecreateAnnouncement.model_validate(a)
            for a in fixture["announcements"]
        ),
        container_observations=tuple(
            ModelDevContainerObservation.model_validate(o)
            for o in fixture["container_observations"]["rows"]
        ),
    )

    result = HandlerDevRecreateAudit().handle(request)

    counts = Counter(v.kind for v in result.violations)
    assert result.recreates_read == 16
    assert counts[Kind.TWO_STARTS_IN_WINDOW] == 3
    assert counts[Kind.OFF_WINDOW_START] == 4
    assert counts[Kind.UNANNOUNCED_START] == 4
    assert counts[Kind.START_WITHOUT_JOB_RECORD] == 2
