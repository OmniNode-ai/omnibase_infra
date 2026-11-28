# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Delivery of a canary receipt from the canary lane to the dev agent over the bus (OMN-26463).

``HandlerRuntimeCanaryReceiptPublish`` turns the receipt a canary run wrote into the
event the runtime publishes on the contract-declared topic; the handler holds no bus.
``HandlerRuntimeCanaryReceiptRecord`` runs on the dev agent's host for each event the
contract subscribes it to and writes the receipt into ``<state dir>/canary-receipts``,
which the dev roll's gate reads. The first receipt recorded creates that directory,
which is what turns the gate on for the lane.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

from pydantic import ValidationError

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
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
from omnibase_infra.models.model_runtime_canary_receipt_recorded import (
    ModelRuntimeCanaryReceiptRecorded,
)
from omnibase_infra.protocols.protocol_runtime_canary_receipt_directory import (
    ProtocolRuntimeCanaryReceiptDirectory,
)

__all__: list[str] = [
    "RECEIPT_DIR_NAME",
    "RuntimeCanaryReceiptDeliveryError",
    "HandlerRuntimeCanaryReceiptPublish",
    "HandlerRuntimeCanaryReceiptRecord",
]

#: The directory the dev roll's gate reads (``handler_runtime_canary.RECEIPT_DIR_NAME``;
#: a test holds the two equal, since a handler module imports no sibling handler module).
RECEIPT_DIR_NAME: Final = "canary-receipts"


class RuntimeCanaryReceiptDeliveryError(RuntimeError):
    """A canary receipt could not be read for publishing or recorded on delivery."""


def _receipt_path(receipt_dir: Path, candidate: ModelRuntimeCanaryCandidate) -> Path:
    return receipt_dir / f"{candidate.key}.json"


class HandlerRuntimeCanaryReceiptPublish:
    """Read one candidate's receipt from the canary lane and return it as the event to publish."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Reads the receipt file; the runtime publishes the returned event."""
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelRuntimeCanaryReceiptPublishRequest
    ) -> ModelRuntimeCanaryReceiptEvent:
        path = _receipt_path(request.receipt_dir, request.candidate)
        if not path.is_file():
            raise RuntimeCanaryReceiptDeliveryError(
                f"no canary receipt for candidate {request.candidate.key} ({path})"
            )
        try:
            receipt = ModelRuntimeCanaryReceipt.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise RuntimeCanaryReceiptDeliveryError(
                f"canary receipt unreadable: {type(exc).__name__} ({path})"
            ) from exc
        if receipt.candidate != request.candidate:
            raise RuntimeCanaryReceiptDeliveryError(
                f"canary receipt names a different candidate ({path})"
            )
        return ModelRuntimeCanaryReceiptEvent(receipt=receipt)


class HandlerRuntimeCanaryReceiptRecord:
    """Write a delivered receipt where the dev roll's gate reads it."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Side-effecting: writes the receipt file."""
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, event: ModelRuntimeCanaryReceiptEvent
    ) -> ModelRuntimeCanaryReceiptRecorded:
        key: type[object] = ProtocolRuntimeCanaryReceiptDirectory
        binding = cast(
            "ProtocolRuntimeCanaryReceiptDirectory", self._container.get_service(key)
        )
        directory = binding.state_dir() / RECEIPT_DIR_NAME
        directory.mkdir(parents=True, exist_ok=True)
        target = _receipt_path(directory, event.receipt.candidate)
        staged = target.with_suffix(".json.tmp")
        staged.write_text(
            event.receipt.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        staged.replace(target)
        return ModelRuntimeCanaryReceiptRecorded(
            receipt_path=str(target), verdict=event.receipt.verdict
        )
