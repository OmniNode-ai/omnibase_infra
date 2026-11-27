# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""node_build_repository_ref_compute decides from what the clone holds (OMN-26724)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_build_repository_ref_compute/contract.yaml"
)


def _handler() -> Any:
    spec = yaml.safe_load(CONTRACT.read_text())["handler_routing"]["handlers"][0][
        "handler"
    ]
    return getattr(importlib.import_module(spec["module"]), spec["name"])()


def _request(**facts: Any) -> Any:
    from omnibase_infra.models.model_build_repository_ref_request import (
        ModelBuildRepositoryRefRequest,
    )

    return ModelBuildRepositoryRefRequest(git_ref="a" * 40, **facts)


def test_a_held_ref_builds() -> None:
    assert _handler().handle(_request(held=True)).verdict.value == "held"


def test_a_ref_still_missing_after_one_good_fetch_is_refused() -> None:
    decision = _handler().handle(
        _request(held=False, fetch_exit_code=0, held_after_fetch=False)
    )
    assert decision.verdict.value == "missing"
    assert decision.refuses is True
    assert "a" * 40 in decision.reason


def test_a_ref_that_arrives_with_the_fetch_builds() -> None:
    decision = _handler().handle(
        _request(held=False, fetch_exit_code=0, held_after_fetch=True)
    )
    assert decision.verdict.value == "held_after_fetch"


def test_no_fetch_answer_is_unverifiable_and_not_refused() -> None:
    for code in (None, 128):
        decision = _handler().handle(_request(held=False, fetch_exit_code=code))
        assert decision.verdict.value == "unverifiable"
        assert decision.refuses is False
