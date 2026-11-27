# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host entry of the canary receipt delivery (OMN-26463).

    record  --state-dir S --event FILE
    publish --receipt-dir D --infra-ref R --omnimarket-ref R --core-ref R [--overlay-rev R]

``record`` runs on the dev agent's host for each delivered receipt event: it binds
``ProtocolRuntimeCanaryReceiptDirectory`` to the agent's state directory and calls the
node's ``record_receipt`` handler, which writes ``<state dir>/canary-receipts/<key>.json``
for the dev roll's gate. ``publish`` runs on the canary lane's host after a run: it calls
the node's ``publish_receipt`` handler and prints the event the runtime publishes.

The exit code is 0 done, 2 usage, 3 refused (no receipt, or an unreadable one).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.handlers.handler_runtime_canary_receipt_delivery import (
    HandlerRuntimeCanaryReceiptPublish,
    HandlerRuntimeCanaryReceiptRecord,
    RuntimeCanaryReceiptDeliveryError,
)
from omnibase_infra.models.model_runtime_canary_candidate import (
    ModelRuntimeCanaryCandidate,
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

__all__: list[str] = ["StateDirReceiptDirectory", "bootstrap", "main"]

_EXIT_USAGE = 2
_EXIT_REFUSED = 3


class StateDirReceiptDirectory(ProtocolRuntimeCanaryReceiptDirectory):
    """The dev agent's state directory, as the record handler asks for it."""

    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir

    def state_dir(self) -> Path:
        return self._state_dir


async def bootstrap(state_dir: Path) -> ModelONEXContainer:
    """Bind the receipt directory to the agent's state directory at the composition root."""
    container = ModelONEXContainer()
    directory_key: type[object] = ProtocolRuntimeCanaryReceiptDirectory
    await container.service_registry.register_instance(
        directory_key, StateDirReceiptDirectory(state_dir)
    )
    await container.service_registry.register_instance(
        HandlerRuntimeCanaryReceiptRecord, HandlerRuntimeCanaryReceiptRecord(container)
    )
    await container.service_registry.register_instance(
        HandlerRuntimeCanaryReceiptPublish, HandlerRuntimeCanaryReceiptPublish()
    )
    return container


def main(argv: list[str] | None = None) -> int:
    """Parse one command, run its handler and print the outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record")
    record.add_argument("--state-dir", type=Path, required=True)
    record.add_argument("--event", type=Path, required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("--receipt-dir", type=Path, required=True)
    publish.add_argument("--infra-ref", required=True)
    publish.add_argument("--omnimarket-ref", required=True)
    publish.add_argument("--core-ref", required=True)
    publish.add_argument("--overlay-rev", default="")
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            event = ModelRuntimeCanaryReceiptEvent.model_validate_json(
                args.event.read_text(encoding="utf-8")
            )
            container = asyncio.run(bootstrap(args.state_dir))
            recorded = container.get_service(HandlerRuntimeCanaryReceiptRecord).handle(
                event
            )
            sys.stdout.write(f"{recorded.model_dump_json()}\n")
            return 0
        candidate = ModelRuntimeCanaryCandidate(
            infra_ref=args.infra_ref,
            omnimarket_ref=args.omnimarket_ref,
            core_ref=args.core_ref,
            overlay_rev=args.overlay_rev,
        )
        container = asyncio.run(bootstrap(args.receipt_dir))
        published = container.get_service(HandlerRuntimeCanaryReceiptPublish).handle(
            ModelRuntimeCanaryReceiptPublishRequest(
                candidate=candidate, receipt_dir=args.receipt_dir
            )
        )
        sys.stdout.write(f"{published.model_dump_json()}\n")
        return 0
    except RuntimeCanaryReceiptDeliveryError as exc:
        sys.stderr.write(f"{exc}\n")
        return _EXIT_REFUSED
    except (OSError, ValidationError, ValueError) as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return _EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
