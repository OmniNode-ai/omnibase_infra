# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The shipped intent target file feeds the real consumers end to end (OMN-21792)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.models.intent_target_uris import (
    ModelIntentTargetUris,
    load_intent_target_uris,
)

pytestmark = pytest.mark.integration

_SHIPPED = (
    Path(__file__).resolve().parents[4]
    / "src/omnibase_infra/models/intent_target_uris/intent_target_uris.yaml"
)


def test_shipped_file_loads_and_every_base_is_a_postgres_uri() -> None:
    uris = ModelIntentTargetUris.from_file(_SHIPPED)
    assert uris == load_intent_target_uris()
    for name in ModelIntentTargetUris.model_fields:
        assert getattr(uris, name).startswith("postgres://")


def test_shipped_file_declares_exactly_the_model_fields() -> None:
    declared = yaml.safe_load(_SHIPPED.read_text(encoding="utf-8"))["intent_targets"]
    assert set(declared) == set(ModelIntentTargetUris.model_fields)


def test_target_has_no_doubled_separator() -> None:
    uris = load_intent_target_uris()
    target = uris.target(uris.node_registrations, "node-1")
    assert "//" not in target.removeprefix("postgres://")
