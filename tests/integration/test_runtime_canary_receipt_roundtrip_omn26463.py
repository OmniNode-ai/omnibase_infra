# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26463 -- a canary run writes a receipt on disk and the roll's gate reads it back."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_runtime_canary import (
    RuntimeCanaryRefusedError,
    judge_candidate,
    require_pass,
)
from omnibase_infra.models.model_runtime_canary_candidate import (
    ModelRuntimeCanaryCandidate,
)
from omnibase_infra.models.model_runtime_canary_request import (
    ModelRuntimeCanaryRequest,
)
from omnibase_infra.protocols.protocol_runtime_canary_probe import (
    ProtocolRuntimeCanaryProbe,
)

pytestmark = pytest.mark.integration

_INFRA = "a" * 40
_MARKET = "b" * 12
_CORE = "c" * 12
_OVERLAY = "d" * 40
_TAGS = {
    "omnimarket": f"dev-20261122165900-{_MARKET}",
    "omnibase_core": f"dev-20261122165800-{_CORE}",
}


class _Probe(ProtocolRuntimeCanaryProbe):
    def __init__(self, *, heartbeat: bool) -> None:
        self.now = 0.0
        self._heartbeat = heartbeat

    def ready(self) -> bool:
        return True

    def dormant_contracts(self) -> tuple[str, ...]:
        return ("node_a", "node_b")

    def wake(self, contract: str) -> bool:
        return True

    def heartbeat_ok(self) -> bool:
        return self._heartbeat

    def log_text(self) -> str:
        return ""


def _judge(tmp_path: Path, probe: _Probe) -> Path:
    candidate = ModelRuntimeCanaryCandidate(
        infra_ref=_INFRA, omnimarket_ref=_MARKET, core_ref=_CORE, overlay_rev=_OVERLAY
    )
    request = ModelRuntimeCanaryRequest(
        candidate=candidate,
        receipt_dir=tmp_path,
        heartbeat_hold_seconds=30,
        poll_seconds=5,
    )

    def sleep(seconds: float) -> None:
        probe.now += seconds

    judge_candidate(request, probe, clock=lambda: probe.now, sleep=sleep)
    return tmp_path


def test_pass_receipt_written_by_a_run_admits_the_roll(tmp_path: Path) -> None:
    directory = _judge(tmp_path, _Probe(heartbeat=True))
    path = require_pass(
        directory, git_ref=_INFRA, sibling_tags=_TAGS, overlay_ref=_OVERLAY
    )
    assert path.is_file()


def test_fail_receipt_written_by_a_run_refuses_the_roll(tmp_path: Path) -> None:
    directory = _judge(tmp_path, _Probe(heartbeat=False))
    with pytest.raises(RuntimeCanaryRefusedError, match="FAIL"):
        require_pass(
            directory, git_ref=_INFRA, sibling_tags=_TAGS, overlay_ref=_OVERLAY
        )
