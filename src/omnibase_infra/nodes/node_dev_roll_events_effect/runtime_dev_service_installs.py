# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Apply one dev-lane service's registered branch installs into its fresh container (OMN-26314).

The dev roll's compose ``up --wait`` judges a replacement runtime's health. A registered
install exists to make that runtime healthy, so it has to be in the replacement BEFORE the
wait reads it; applied after the roll (OMN-26300) it was never reached when the wait
refused the roll. The roll step starts the replacement without the wait, runs this entry,
then issues the very same ``up --wait`` it always issued.

This entry only puts files in. It is never a verdict: the wait judges whatever is running.
A file that cannot be read back whole is logged at ERROR and the exit is ``EXIT_NOT_WHOLE``;
the roll goes on and the wait judges the runtime as it is.

SCOPE
-----
* The service's own container only. The records are viewed through a store that narrows
  each install to the one service and hides the others, so the apply cannot copy a file
  into a peer's container and cannot write the shared records or the shared status file.
* Dev lane only. ``HandlerDevLaneBranchInstall`` refuses a compose project that names prod
  or stability (``assert_dev_project``), and the roll step is the dev roll.
* Idempotent. A file whose sha256 already matches is not copied and its container is not
  restarted, so the post-rebuild apply and the idle apply find nothing to do.

Command line (run by the dev boot guard from the deploy-source clone's ``src``)::

    python src/omnibase_infra/nodes/node_dev_roll_events_effect/runtime_dev_service_installs.py \\
        --state-dir D --compose-project P --service S
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from omnibase_infra.handlers.handler_dev_lane_branch_install import (
    EXIT_NOT_WHOLE,
    EXIT_OK,
    EXIT_REFUSED,
    DevLaneBranchInstallRefusedError,
    DevLaneBranchInstallStore,
    HandlerDevLaneBranchInstall,
    Runner,
    Sleeper,
)
from omnibase_infra.models.model_dev_lane_branch_install import (
    ModelDevLaneBranchInstall,
)
from omnibase_infra.models.model_dev_lane_branch_install_receipt import (
    ModelDevLaneBranchInstallReceipt,
)
from omnibase_infra.models.model_dev_lane_branch_install_records import (
    ModelDevLaneBranchInstallRecords,
)
from omnibase_infra.models.model_dev_lane_branch_install_release import (
    ModelDevLaneBranchInstallRelease,
)

logger = logging.getLogger(__name__)


class DevLaneRolledInstallStore(DevLaneBranchInstallStore):
    """A read-only view of the declared installs that name one service, unexpired.

    Each install is narrowed to that service, so the handler reads and writes that
    service's container alone. Every write the handler could make to the shared store
    (records, released log, status) is refused or dropped here: this view never edits
    what the agent's own apply owns.
    """

    def __init__(
        self,
        state_dir: Path,
        service: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(state_dir)
        self._service = service
        self._clock = clock or (lambda: datetime.now(UTC))

    def load(self) -> ModelDevLaneBranchInstallRecords:
        now = self._clock()
        return ModelDevLaneBranchInstallRecords(
            installs=tuple(
                install.model_copy(update={"services": (self._service,)})
                for install in super().load().installs
                if self._service in install.services and install.expires_at > now
            )
        )

    def save(self, records: ModelDevLaneBranchInstallRecords) -> None:
        raise DevLaneBranchInstallRefusedError(
            "the service view of the declared installs is read only"
        )

    def append_released(
        self,
        install: ModelDevLaneBranchInstall,
        release: ModelDevLaneBranchInstallRelease,
    ) -> None:
        raise DevLaneBranchInstallRefusedError(
            "the service view of the declared installs never releases an install"
        )

    def write_status(self, receipt: ModelDevLaneBranchInstallReceipt) -> None:
        """The shared status file is the agent's apply and idle check; a partial view never writes it."""
        logger.debug(
            "service view of %s: %d readback(s) not written to the shared status",
            self._service,
            len(receipt.readbacks),
        )


def apply_service_installs(
    state_dir: Path,
    compose_project: str,
    service: str,
    *,
    run: Runner | None = None,
    clock: Callable[[], datetime] | None = None,
    sleep: Sleeper | None = None,
) -> ModelDevLaneBranchInstallReceipt:
    """Put every declared install of ``service`` into its running container and read it back."""
    store = DevLaneRolledInstallStore(state_dir, service, clock)
    return HandlerDevLaneBranchInstall(
        store, compose_project, run=run, clock=clock, sleep=sleep
    ).apply()


def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Apply one service's installs. Logs go to stderr; the result is one stdout line."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s branch-installs: %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="python src/omnibase_infra/nodes/node_dev_roll_events_effect/runtime_dev_service_installs.py",
        description="Apply one dev-lane service's declared branch installs.",
    )
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--compose-project", required=True)
    parser.add_argument("--service", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = apply_service_installs(
            args.state_dir, args.compose_project, args.service
        )
    except DevLaneBranchInstallRefusedError as exc:
        _emit(f"REFUSED {exc}")
        return EXIT_REFUSED
    _emit(f"RECEIPT {receipt.model_dump_json()}")
    return EXIT_OK if receipt.ok else EXIT_NOT_WHOLE


__all__ = [
    "DevLaneRolledInstallStore",
    "apply_service_installs",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
