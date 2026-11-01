# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-backed enum values fail closed and preserve import semantics."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_infra.enums.contract_topic_value import contract_topic_value

pytestmark = pytest.mark.unit


def test_topic_comes_from_supplied_contract(tmp_path: Path) -> None:
    source = tmp_path / "contract.yaml"
    source.write_text(
        "event_bus:\n  publish_topics:\n    - onex.evt.example.some-event.v2\n"
    )
    assert (
        contract_topic_value(str(source), "example", "EVT_SOME_EVENT_V2")
        == "onex.evt.example.some-event.v2"
    )
    with pytest.raises(ValueError, match="must bind one contract topic"):
        contract_topic_value(str(source), "example", "EVT_MISSING_V2")


def test_normalization_collision_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "contract.yaml"
    source.write_text(
        "event_bus:\n  publish_topics:\n    - onex.evt.example.some-event.v1\n    - onex.evt.example.some.event.v1\n"
    )
    with pytest.raises(ValueError, match="must bind one contract topic"):
        contract_topic_value(str(source), "example", "EVT_SOME_EVENT_V1")


def test_prose_cannot_replace_a_declaration(tmp_path: Path) -> None:
    source = tmp_path / "contract.yaml"
    source.write_text("description: onex.evt.example.some-event.v1\n")
    with pytest.raises(ValueError, match="must bind one contract topic"):
        contract_topic_value(str(source), "example", "EVT_SOME_EVENT_V1")


@pytest.mark.parametrize(
    "module", ["omnibase_infra.topics", "omnibase_infra.enums.generated"]
)
def test_fresh_import_order(module: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {module}; from omnibase_infra.topics import SUFFIX_OMNICLAUDE_NOTIFICATION_BLOCKED; assert type(SUFFIX_OMNICLAUDE_NOTIFICATION_BLOCKED) is str",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_unreadable_contract_error_names_source() -> None:
    with pytest.raises(ValueError, match=r"nope/missing\.yaml: cannot read contract"):
        contract_topic_value("nope/missing.yaml", "p", "M")
