# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Durable ledger that bounds the runtime restart loop [OMN-18550].

On 2026-09-16 the ``omninode-runtime`` container on the dev lane died on a
malformed contract and was relaunched by ``restart: unless-stopped`` every forty
seconds for hours. Nothing counted the exits and nothing recorded them anywhere
but a log. This handler is the counter and the record.

State lives in ``<state_dir>/boot_failures/`` on the runtime data volume, so it
survives the container restarts it counts:

* ``<service>.json`` -- the current streak: image source hash, container,
  consecutive failing exits, and the last event (so a restarted container can say why it
  will not boot).
* ``events.jsonl`` -- every ``ModelRuntimeBootFailedEvent``, append-only.

A streak is consecutive failing exits of ONE image. It ends on a clean exit, on
a failure after a process that ran for ``healthy_after_seconds`` (a healthy run
that later crashed is a new incident, not the same loop), and on a new image
source hash or a new container (a redeploy of a fixed build recreates the
container, so it is never held by the old count even on a lane whose images
all carry the same source hash).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.health.model_boot_attempt_outcome import (
    ModelBootAttemptOutcome,
)
from omnibase_infra.models.health.model_boot_identity import ModelBootIdentity
from omnibase_infra.models.health.model_runtime_boot_failed_event import (
    ModelRuntimeBootFailedEvent,
)

LEDGER_DIRNAME: str = "boot_failures"
EVENTS_FILENAME: str = "events.jsonl"


@dataclass(frozen=True)
class BootFailureStreak:
    """The stored streak: whose failures, how many, and the last one."""

    runtime_source_hash: str
    container_ref: str
    consecutive_failures: int
    last_event: ModelRuntimeBootFailedEvent


def _ledger_dir(state_dir: str) -> Path:
    return Path(state_dir) / LEDGER_DIRNAME


def _streak_path(state_dir: str, service: str) -> Path:
    return _ledger_dir(state_dir) / f"{service}.json"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class HandlerBootFailureLedger:
    """Counts consecutive failing runtime exits and records each as an event."""

    @property
    def handler_type(self) -> EnumHandlerType:
        """INFRA_HANDLER: a file-backed ledger on the runtime data volume."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """EFFECT: every failing exit is written to disk."""
        return EnumHandlerTypeCategory.EFFECT

    def handle(self, request: ModelBootAttemptOutcome) -> ModelRuntimeBootFailedEvent:
        """Record one failing exit and return its event.

        Raises:
            ValueError: ``request.exit_code`` is 0; a clean exit is not a failure
                (use :meth:`record_clean_exit`).
        """
        if request.exit_code == 0:
            raise ValueError("a clean exit is not a boot failure")

        identity = request.identity
        previous = self._read_streak(identity)
        same_incident = (
            previous is not None
            and previous.runtime_source_hash == identity.runtime_source_hash
            and previous.container_ref == identity.container_ref
            and request.uptime_seconds < request.healthy_after_seconds
        )
        count = previous.consecutive_failures + 1 if same_incident and previous else 1

        event = ModelRuntimeBootFailedEvent(
            service=identity.service,
            container_ref=identity.container_ref,
            runtime_source_hash=identity.runtime_source_hash,
            exit_code=request.exit_code,
            cause=request.cause,
            uptime_seconds=request.uptime_seconds,
            consecutive_failures=count,
            max_consecutive_failures=identity.max_consecutive_failures,
            retries_exhausted=count >= identity.max_consecutive_failures,
            failed_at=request.observed_at,
        )

        payload = event.model_dump_json()
        events_path = _ledger_dir(identity.state_dir) / EVENTS_FILENAME
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        _write_atomic(
            _streak_path(identity.state_dir, identity.service),
            json.dumps(
                {
                    "runtime_source_hash": identity.runtime_source_hash,
                    "container_ref": identity.container_ref,
                    "consecutive_failures": count,
                    "last_event": json.loads(payload),
                }
            ),
        )
        return event

    def record_clean_exit(self, identity: ModelBootIdentity) -> None:
        """End the streak: the runtime started and stopped without failing."""
        _streak_path(identity.state_dir, identity.service).unlink(missing_ok=True)

    def exhausted_event(
        self, identity: ModelBootIdentity
    ) -> ModelRuntimeBootFailedEvent | None:
        """Return the last event when this container has used up its retries, else None.

        The streak is the stored count read against the bound declared NOW, so
        lowering the bound takes effect on the next start without a reset.
        """
        streak = self._read_streak(identity)
        if (
            streak is None
            or streak.runtime_source_hash != identity.runtime_source_hash
            or streak.container_ref != identity.container_ref
            or streak.consecutive_failures < identity.max_consecutive_failures
        ):
            return None
        return streak.last_event

    @staticmethod
    def _read_streak(identity: ModelBootIdentity) -> BootFailureStreak | None:
        path = _streak_path(identity.state_dir, identity.service)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            # A corrupt streak file must not wedge the runtime: restart the count.
            return None
        if not isinstance(raw, dict):
            return None
        try:
            return BootFailureStreak(
                runtime_source_hash=str(raw["runtime_source_hash"]),
                container_ref=str(raw["container_ref"]),
                consecutive_failures=int(raw["consecutive_failures"]),
                last_event=ModelRuntimeBootFailedEvent.model_validate(
                    raw["last_event"]
                ),
            )
        except (KeyError, TypeError, ValueError, ValidationError):
            return None


__all__: list[str] = [
    "EVENTS_FILENAME",
    "LEDGER_DIRNAME",
    "BootFailureStreak",
    "HandlerBootFailureLedger",
]
