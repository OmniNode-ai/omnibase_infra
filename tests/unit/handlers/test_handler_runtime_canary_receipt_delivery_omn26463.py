# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26463 -- a canary receipt crosses the bus and the dev agent records it for the roll's gate."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums.enum_runtime_canary_verdict import EnumRuntimeCanaryVerdict
from omnibase_infra.handlers.handler_runtime_canary import (
    RECEIPT_DIR_NAME,
    RuntimeCanaryRefusedError,
    receipt_path,
    require_for_roll,
)
from omnibase_infra.handlers.handler_runtime_canary_receipt_delivery import (
    RECEIPT_DIR_NAME as DELIVERY_DIR_NAME,
)
from omnibase_infra.handlers.handler_runtime_canary_receipt_delivery import (
    HandlerRuntimeCanaryReceiptPublish,
    HandlerRuntimeCanaryReceiptRecord,
    RuntimeCanaryReceiptDeliveryError,
)
from omnibase_infra.models.model_runtime_canary_candidate import (
    ModelRuntimeCanaryCandidate,
)
from omnibase_infra.models.model_runtime_canary_receipt import (
    ModelRuntimeCanaryReceipt,
)
from omnibase_infra.models.model_runtime_canary_receipt_event import (
    ModelRuntimeCanaryReceiptEvent,
)
from omnibase_infra.models.model_runtime_canary_receipt_publish_request import (
    ModelRuntimeCanaryReceiptPublishRequest,
)
from omnibase_infra.protocols.protocol_runtime_canary_receipt_directory import (
    ProtocolRuntimeCanaryReceiptDirectory,
)

pytestmark = pytest.mark.unit


class _StateDir(ProtocolRuntimeCanaryReceiptDirectory):
    def __init__(self, path: Path) -> None:
        self._path = path

    def state_dir(self) -> Path:
        return self._path


class _Container:
    """Resolves the one protocol the record handler asks the container for."""

    def __init__(self, path: Path) -> None:
        self._binding = _StateDir(path)

    def get_service(self, key: type[object]) -> object:
        assert key is ProtocolRuntimeCanaryReceiptDirectory
        return self._binding


def _record(state_dir: Path) -> HandlerRuntimeCanaryReceiptRecord:
    return HandlerRuntimeCanaryReceiptRecord(
        cast("ModelONEXContainer", _Container(state_dir))
    )


_MARKET = "b" * 12
_CORE = "c" * 12
_CANDIDATE = ModelRuntimeCanaryCandidate(
    infra_ref="a" * 40, omnimarket_ref=_MARKET, core_ref=_CORE, overlay_rev="d" * 40
)
_TAGS = {
    "omnimarket": f"dev-20261122165900-{_MARKET}",
    "omnibase_core": f"dev-20261122165800-{_CORE}",
}
_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_runtime_canary_receipt_effect/contract.yaml"
)


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


def test_publish_reads_the_receipt_the_run_wrote(tmp_path: Path) -> None:
    receipt = _receipt(EnumRuntimeCanaryVerdict.PASS)
    receipt_path(tmp_path, _CANDIDATE).write_text(receipt.model_dump_json())
    event = HandlerRuntimeCanaryReceiptPublish().handle(
        ModelRuntimeCanaryReceiptPublishRequest(
            candidate=_CANDIDATE, receipt_dir=tmp_path
        )
    )
    assert event.receipt == receipt


def test_publish_refuses_a_candidate_with_no_receipt(tmp_path: Path) -> None:
    with pytest.raises(RuntimeCanaryReceiptDeliveryError, match="no canary receipt"):
        HandlerRuntimeCanaryReceiptPublish().handle(
            ModelRuntimeCanaryReceiptPublishRequest(
                candidate=_CANDIDATE, receipt_dir=tmp_path
            )
        )


def test_record_pass_event_lets_the_dev_roll_through(
    tmp_path: Path,
) -> None:
    assert _roll(tmp_path) is None  # no directory: the gate is off
    recorded = _record(tmp_path).handle(
        ModelRuntimeCanaryReceiptEvent(receipt=_receipt(EnumRuntimeCanaryVerdict.PASS))
    )
    assert Path(recorded.receipt_path).parent == tmp_path / RECEIPT_DIR_NAME
    assert _roll(tmp_path) == Path(recorded.receipt_path)


def test_record_fail_event_makes_the_dev_roll_refuse(
    tmp_path: Path,
) -> None:
    _record(tmp_path).handle(
        ModelRuntimeCanaryReceiptEvent(receipt=_receipt(EnumRuntimeCanaryVerdict.FAIL))
    )
    with pytest.raises(RuntimeCanaryRefusedError, match="TypeError"):
        _roll(tmp_path)


def test_delivery_names_the_directory_and_file_the_gate_reads() -> None:
    assert DELIVERY_DIR_NAME == RECEIPT_DIR_NAME


def test_contract_declares_one_publish_and_one_subscribe_topic() -> None:
    contract = yaml.safe_load(_CONTRACT.read_text())
    bus = contract["event_bus"]
    assert len(bus["publish_topics"]) == 1
    assert bus["publish_topics"] == bus["subscribe_topics"]
    assert {h["operation"] for h in contract["handler_routing"]["handlers"]} == {
        "publish_receipt",
        "record_receipt",
    }
