# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26463 -- the host entry binds the receipt directory to the agent's state dir.

The dev agent's host calls ``record`` with each delivered receipt event, and the
canary lane's runner calls ``publish`` with the candidate it judged. A PASS the
agent recorded lets ``require_for_roll`` through; a FAIL makes it refuse.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from omnibase_infra.enums.enum_runtime_canary_verdict import EnumRuntimeCanaryVerdict
from omnibase_infra.handlers.handler_runtime_canary import (
    RuntimeCanaryRefusedError,
    receipt_path,
    require_for_roll,
)
from omnibase_infra.models.model_runtime_canary_candidate import (
    ModelRuntimeCanaryCandidate,
)
from omnibase_infra.models.model_runtime_canary_receipt import (
    ModelRuntimeCanaryReceipt,
)
from omnibase_infra.nodes.node_runtime_canary_receipt_effect import (
    runtime_runtime_canary_receipt as entry,
)
from omnibase_infra.protocols.protocol_runtime_canary_receipt_directory import (
    ProtocolRuntimeCanaryReceiptDirectory,
)

pytestmark = pytest.mark.unit

_MARKET = "b" * 12
_CORE = "c" * 12
_CANDIDATE = ModelRuntimeCanaryCandidate(
    infra_ref="a" * 40, omnimarket_ref=_MARKET, core_ref=_CORE, overlay_rev="d" * 40
)
_TAGS = {
    "omnimarket": f"dev-20261122165900-{_MARKET}",
    "omnibase_core": f"dev-20261122165800-{_CORE}",
}


def _receipt(verdict: EnumRuntimeCanaryVerdict) -> ModelRuntimeCanaryReceipt:
    failing = verdict is EnumRuntimeCanaryVerdict.FAIL
    return ModelRuntimeCanaryReceipt(
        verdict=verdict,
        candidate=_CANDIDATE,
        started_at="2026-11-27T14:00:00Z",
        finished_at="2026-11-27T14:20:00Z",
        first_failing_line="TypeError: consumer_group_handover" if failing else None,
        failed_check="wake" if failing else None,
    )


def _roll(state_dir: Path) -> Path | None:
    command = {
        "git_ref": _CANDIDATE.infra_ref,
        "sibling_tags": _TAGS,
        "overlay_ref": _CANDIDATE.overlay_rev,
    }
    return require_for_roll(command, state_dir)


def test_the_directory_binding_is_the_agents_state_dir(tmp_path: Path) -> None:
    container = asyncio.run(entry.bootstrap(tmp_path))
    binding = container.get_service(ProtocolRuntimeCanaryReceiptDirectory)
    assert binding.state_dir() == tmp_path


@pytest.mark.parametrize(
    ("verdict", "admitted"),
    [(EnumRuntimeCanaryVerdict.PASS, True), (EnumRuntimeCanaryVerdict.FAIL, False)],
)
def test_record_makes_the_roll_read_the_delivered_receipt(
    tmp_path: Path, verdict: EnumRuntimeCanaryVerdict, admitted: bool
) -> None:
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps({"receipt": json.loads(_receipt(verdict).model_dump_json())})
    )
    state = tmp_path / "state"
    assert entry.main(["record", "--state-dir", str(state), "--event", str(event)]) == 0
    if admitted:
        assert _roll(state) is not None
    else:
        with pytest.raises(RuntimeCanaryRefusedError, match="TypeError"):
            _roll(state)


def test_publish_prints_the_event_of_the_receipt_the_run_wrote(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt = _receipt(EnumRuntimeCanaryVerdict.PASS)
    receipt_path(tmp_path, _CANDIDATE).write_text(receipt.model_dump_json())
    code = entry.main(
        [
            "publish",
            "--receipt-dir", str(tmp_path),
            "--infra-ref", _CANDIDATE.infra_ref,
            "--omnimarket-ref", _CANDIDATE.omnimarket_ref,
            "--core-ref", _CANDIDATE.core_ref,
            "--overlay-rev", _CANDIDATE.overlay_rev,
        ]
    )  # fmt: skip
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["receipt"]["verdict"] == "PASS"


def test_publish_refuses_a_candidate_with_no_receipt(tmp_path: Path) -> None:
    code = entry.main(
        [
            "publish",
            "--receipt-dir", str(tmp_path),
            "--infra-ref", "a" * 40,
            "--omnimarket-ref", _MARKET,
            "--core-ref", _CORE,
        ]
    )  # fmt: skip
    assert code == 3
