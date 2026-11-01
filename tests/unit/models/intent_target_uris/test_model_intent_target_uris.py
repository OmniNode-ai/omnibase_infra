# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Intent target base URIs load from the declared file (OMN-21792)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.intent_target_uris import (
    ModelIntentTargetUris,
    load_intent_target_uris,
)

pytestmark = pytest.mark.unit

_DECLARED = (
    Path(__file__).resolve().parents[4]
    / "src/omnibase_infra/models/intent_target_uris/intent_target_uris.yaml"
)


def test_loaded_uris_equal_the_declared_file() -> None:
    declared = yaml.safe_load(_DECLARED.read_text(encoding="utf-8"))["intent_targets"]
    assert load_intent_target_uris().model_dump() == declared


def test_target_joins_base_and_segments() -> None:
    uris = load_intent_target_uris()
    assert (
        uris.target(uris.event_ledger, "topic.a", 3, 42)
        == f"{uris.event_ledger}/topic.a/3/42"
    )


@pytest.mark.parametrize("missing", sorted(ModelIntentTargetUris.model_fields))
def test_omitted_value_fails_to_load(missing: str, tmp_path: Path) -> None:
    block = yaml.safe_load(_DECLARED.read_text(encoding="utf-8"))["intent_targets"]
    del block[missing]
    path = tmp_path / "uris.yaml"
    path.write_text(yaml.safe_dump({"intent_targets": block}), encoding="utf-8")
    with pytest.raises(ProtocolConfigurationError, match=missing):
        ModelIntentTargetUris.from_file(path)


@pytest.mark.parametrize("extra", ["unexpected_key"])
def test_extra_value_is_refused(extra: str, tmp_path: Path) -> None:
    block = yaml.safe_load(_DECLARED.read_text(encoding="utf-8"))["intent_targets"]
    block[extra] = "postgres://x"
    path = tmp_path / "uris.yaml"
    path.write_text(yaml.safe_dump({"intent_targets": block}), encoding="utf-8")
    with pytest.raises(ProtocolConfigurationError, match=extra):
        ModelIntentTargetUris.from_file(path)


@pytest.mark.parametrize("bad", ["", "postgres://", "postgres://x/", "http://x"])
def test_malformed_base_is_refused(bad: str, tmp_path: Path) -> None:
    block = yaml.safe_load(_DECLARED.read_text(encoding="utf-8"))["intent_targets"]
    block["topics"] = bad
    path = tmp_path / "uris.yaml"
    path.write_text(yaml.safe_dump({"intent_targets": block}), encoding="utf-8")
    with pytest.raises(ProtocolConfigurationError, match="topics"):
        ModelIntentTargetUris.from_file(path)


def test_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ProtocolConfigurationError):
        ModelIntentTargetUris.from_file(tmp_path / "absent.yaml")


def test_malformed_yaml_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "uris.yaml"
    path.write_text("intent_targets: [unclosed\n", encoding="utf-8")
    with pytest.raises(ProtocolConfigurationError):
        ModelIntentTargetUris.from_file(path)


def test_file_without_the_mapping_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "uris.yaml"
    path.write_text("other: {}\n", encoding="utf-8")
    with pytest.raises(ProtocolConfigurationError):
        ModelIntentTargetUris.from_file(path)
