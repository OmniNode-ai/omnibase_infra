# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""C28 pairs a validation error with its boundary line across a busy runtime log."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.nodes.node_board_probe_effect.handlers import (
    _consumer_flow_collection as collection,
)
from scripts.ci import c28_consumer_flow_probe as script

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).resolve().parents[3] / "tests/fixtures/omn19812"
_BUSY = (
    "2026-10-22 11:56:10 [INFO] omnibase_infra.runtime.service_kernel: Dispatch started"
)


def _spread(name: str, gap: int) -> list[str]:
    out: list[str] = []
    for ln in (_FIXTURES / name).read_text(encoding="utf-8").splitlines():
        out.append(ln)
        if script.VALIDATION_ERROR_RE.search(ln):
            out.extend([_BUSY] * gap)
    return out


@pytest.mark.parametrize("gap", [63, 171])
def test_collector_and_script_excuse_probe_errors_behind_a_busy_log(gap: int) -> None:
    lines = _spread("runtime_seam_probe_prefixed.captured.txt", gap)
    assert collection.natural_validation_errors(
        lines
    ) == script.natural_validation_errors(lines)
    total, own, natural = collection.natural_validation_errors(lines)
    assert total > 0
    assert (own, natural) == (total, 0)


def test_collector_keeps_unprefixed_errors_natural_behind_a_busy_log() -> None:
    lines = _spread("runtime_seam_unprefixed.captured.txt", 171)
    total, own, natural = collection.natural_validation_errors(lines)
    assert total > 0
    assert (own, natural) == (0, total)
