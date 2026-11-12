# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Publishes the dev roll events (start, manifest, incident) without ever failing a deploy.

The topics arrive from the node's contract.yaml through runtime_dev_roll_events.py.

Five whys plan roots R1 and R2. The deploy agent loads its host effects by file path from
its own checkout (see ``load_dev_boot_guard`` in the agent's executor), and the
``omnibase_infra`` it can import may be an older install without the new model files, so
the five models are loaded by file path too, in dependency order, before use. Nothing here
may raise into a deploy: the events are evidence, never part of a verdict.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

from pydantic import JsonValue

logger = logging.getLogger(__name__)

FRICTION = "friction_type=dev_event_publish_failed"
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _load_sibling(relative: str) -> ModuleType:
    """Load one package file by path under its dotted name, once."""
    name = "omnibase_infra." + relative.removesuffix(".py").replace("/", ".")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _PACKAGE_ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load dev roll event module: {relative}")
    module = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cause = _load_sibling("enums/enum_dev_start_cause.py")
_load_sibling("models/model_dev_log_capture.py")
_started = _load_sibling("models/model_dev_runtime_started.py")
_manifest = _load_sibling("models/model_dev_roll_manifest.py")
_incident = _load_sibling("models/model_dev_roll_incident.py")

EnumDevStartCause = _cause.EnumDevStartCause
ModelDevRuntimeStarted = _started.ModelDevRuntimeStarted
ModelDevRollManifest = _manifest.ModelDevRollManifest
ModelDevRollIncident = _incident.ModelDevRollIncident


def docker_started_at(value: object, *, fallback: datetime) -> datetime:
    """Docker's StartedAt (RFC 3339, up to nine fractional digits, Z) as aware UTC.

    The fraction is cut to six digits, never rounded. None, empty or unparsable
    input returns ``fallback``.
    """
    if not isinstance(value, str) or not value.strip():
        return fallback
    text = value.strip().removesuffix("Z")
    seconds, dot, fraction = text.partition(".")
    if dot:
        digits = ""
        for char in fraction:
            if not char.isdigit():
                break
            digits += char
        zone = fraction[len(digits) :]
        text = f"{seconds}.{digits[:6].ljust(6, '0')}{zone}"
    if text[-6:-5] not in ("+", "-") and not text.endswith("Z"):
        text += "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return fallback
    return parsed.astimezone(UTC)


class DevRollEventsPublisher:
    """Builds the typed roll events and sends them; every failure is logged and swallowed."""

    def __init__(
        self,
        send: Callable[[str, str, dict[str, JsonValue]], bool],
        topics: Mapping[str, str],
    ) -> None:
        self._send = send
        self._topics = topics

    def _publish(self, topic: str, key: str, build: Callable[[], Any]) -> None:
        try:
            payload = build().model_dump(mode="json")
            if not self._send(topic, key, payload):
                logger.error(
                    "dev event %s was not accepted by the bus %s", key, FRICTION
                )
        except Exception:
            logger.exception("dev event %s could not be published %s", key, FRICTION)

    def manifest(self, command: Mapping[str, Any]) -> None:
        job_id = str(command.get("correlation_id", ""))
        self._publish(
            self._topics["dev_roll_manifest"],
            f"dev-roll-manifest/{job_id}",
            lambda: ModelDevRollManifest(
                event_id=uuid4(),
                job_id=job_id,
                issuer=command["requested_by"],
                git_ref=command["git_ref"],
                scope=str(command["scope"]),
                requested_at=command.get("requested_at"),
                accepted_at=datetime.now(UTC),
                open_pr=command.get("open_pr", ""),
                rollback=command.get("rollback") is not None,
            ),
        )

    def started(self, job_id: str, started_at: datetime, cause: str) -> None:
        self._publish(
            self._topics["dev_runtime_started"],
            f"dev-runtime-started/{job_id}",
            lambda: ModelDevRuntimeStarted(
                event_id=uuid4(),
                job_id=job_id,
                runtime_lane="dev",
                started_at=started_at,
                cause=cause,
            ),
        )

    def incident(self, guard_incident: Mapping[str, Any], *, issuer: str) -> None:
        """Publish the incident; a verified recovery recreate is one more start."""
        job_id = str(guard_incident.get("job", ""))
        self._publish(
            self._topics["dev_roll_incident"],
            f"dev-roll-incident/{job_id}",
            lambda: ModelDevRollIncident.from_guard_incident(
                guard_incident, event_id=uuid4(), issuer=issuer
            ),
        )
        if (
            job_id
            and guard_incident.get("recovery_verified") is True
            and guard_incident.get("recovery_skipped_healthy") is False
            and "runtime_kept_revision" not in guard_incident
        ):
            self.started(job_id, datetime.now(UTC), EnumDevStartCause.OTHER)
