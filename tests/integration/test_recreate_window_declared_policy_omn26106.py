# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The declared runtime policy file builds a real recreate request (OMN-26106).

Reads ``docker/runtime-policy.env`` from disk, builds the real policy from it and
checks that the window's request model takes a named requester and refuses an empty one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_infra.models.model_dev_recreate_policy import ModelDevRecreatePolicy
from omnibase_infra.models.model_dev_recreate_request import ModelDevRecreateRequest

POLICY_ENV = Path(__file__).resolve().parents[2] / "docker" / "runtime-policy.env"
NOW = datetime(2026, 11, 19, 19, 16, 13, tzinfo=UTC)


def _declared_policy() -> ModelDevRecreatePolicy:
    env: dict[str, str] = {}
    for line in POLICY_ENV.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.startswith("#"):
            env[key] = value
    return ModelDevRecreatePolicy.from_env(env)


@pytest.mark.integration
def test_declared_policy_request_takes_a_named_requester() -> None:
    request = ModelDevRecreateRequest(
        policy=_declared_policy(), now=NOW, requested_by="deploy-agent-stability-follow"
    )
    assert request.requested_by == "deploy-agent-stability-follow"


@pytest.mark.integration
def test_declared_policy_request_refuses_an_empty_requester() -> None:
    with pytest.raises(ValidationError):
        ModelDevRecreateRequest(policy=_declared_policy(), now=NOW, requested_by="")
