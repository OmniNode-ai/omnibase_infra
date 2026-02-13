# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event Registry - Event type to Kafka topic mapping.

This module provides a generic event registry that maps semantic event types
to Kafka topics and handles metadata injection.

The registry is the central configuration point for:
- Event type -> topic routing
- Partition key extraction
- Payload validation
- Metadata injection (correlation IDs, timestamps, schema versions)

The registry ships with no default registrations. Consumers must register
their own event types via ``register()`` or ``register_batch()``.

Example Usage:
    ```python
    from omnibase_infra.runtime.emit_daemon.event_registry import (
        EventRegistry,
        ModelEventRegistration,
    )

    # Create registry
    registry = EventRegistry(environment="dev")

    # Register event types
    registry.register_batch([
        ModelEventRegistration(
            event_type="myapp.submitted",
            topic_template="onex.evt.myapp.submitted.v1",
            partition_key_field="session_id",
            required_fields=["session_id", "payload"],
        ),
        ModelEventRegistration(
            event_type="myapp.completed",
            topic_template="onex.evt.myapp.completed.v1",
            partition_key_field="session_id",
            required_fields=["session_id"],
        ),
    ])

    # Resolve topic for event type (realm-agnostic, no env prefix)
    topic = registry.resolve_topic("myapp.submitted")
    # Returns: "onex.evt.myapp.submitted.v1"

    # Inject metadata into payload
    enriched = registry.inject_metadata(
        event_type="myapp.submitted",
        payload={"session_id": "abc123", "payload": "data"},
        correlation_id="corr-123",
    )
    # Returns payload with correlation_id, causation_id, emitted_at, schema_version
    ```

Note:
    Topics are realm-agnostic in ONEX. The environment/realm is enforced via
    envelope identity, not topic naming.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.errors import OnexError

if TYPE_CHECKING:
    from omnibase_infra.runtime.emit_daemon.model_event_registry_fingerprint import (
        ModelEventRegistryFingerprint,
    )

logger = logging.getLogger(__name__)


class ModelEventRegistration(BaseModel):
    """Registration configuration for a single event type.

    Defines how a semantic event type maps to Kafka infrastructure including
    topic naming, partition keys, and payload validation rules.

    Attributes:
        event_type: Semantic event type identifier (e.g., "myapp.submitted").
            This is the logical name used by event emitters.
        topic_template: Kafka topic name (realm-agnostic, no environment prefix).
            Example: "onex.evt.myapp.submitted.v1"
            Note: Topics are realm-agnostic in ONEX. The environment/realm is
            enforced via envelope identity, not topic naming.
        partition_key_field: Optional field name in payload to use as partition key.
            When set, ensures events with same key go to same partition for ordering.
        required_fields: List of field names that must be present in payload.
            Validation will fail if any required field is missing.
        schema_version: Semantic version of the event schema (default: "1.0.0").
            Injected into event metadata for schema evolution tracking.

    Example:
        >>> reg = ModelEventRegistration(
        ...     event_type="myapp.submitted",
        ...     topic_template="onex.evt.myapp.submitted.v1",
        ...     partition_key_field="session_id",
        ...     required_fields=["session_id", "payload"],
        ...     schema_version="1.0.0",
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
    )

    event_type: str = Field(
        description="Semantic event type identifier (e.g., 'myapp.submitted')",
    )
    topic_template: str = Field(
        description="Kafka topic name (realm-agnostic, no environment prefix)",
    )
    partition_key_field: str | None = Field(
        default=None,
        description="Optional field name in payload to use as partition key",
    )
    required_fields: list[str] = Field(
        default_factory=list,
        description="List of field names that must be present in payload",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Semantic version of the event schema",
    )


class EventRegistry:
    """Registry for event type to Kafka topic mappings.

    Manages the mapping between semantic event types and Kafka infrastructure,
    including topic resolution, partition key extraction, payload validation,
    and metadata injection.

    The registry starts empty. Consumers must register their own event types
    via ``register()`` or ``register_batch()``.

    Attributes:
        environment: Deployment environment name (e.g., "dev", "staging", "prod").
            Stored for potential use in consumer group derivation by related
            components.

    Example:
        >>> registry = EventRegistry(environment="dev")
        >>> registry.register(
        ...     ModelEventRegistration(
        ...         event_type="myapp.submitted",
        ...         topic_template="onex.evt.myapp.submitted.v1",
        ...         required_fields=["payload"],
        ...     )
        ... )
        >>> registry.resolve_topic("myapp.submitted")
        'onex.evt.myapp.submitted.v1'

    Note:
        Topics are realm-agnostic in ONEX. The environment is stored for
        potential use in consumer group derivation by related components,
        but topics themselves do not include environment prefixes.
    """

    def __init__(self, environment: str = "dev") -> None:
        """Initialize an empty event registry.

        Args:
            environment: Deployment environment name. Stored for potential
                use in consumer group derivation. Defaults to "dev".

        Note:
            The registry starts with no registrations. Use ``register()``
            or ``register_batch()`` to add event type mappings.
        """
        self._environment = environment
        self._registrations: dict[str, ModelEventRegistration] = {}

    def register(self, registration: ModelEventRegistration) -> None:
        """Register an event type mapping.

        Adds or updates a registration for the given event type.
        Existing registrations for the same event type are overwritten.

        Args:
            registration: Event registration configuration.

        Example:
            >>> registry = EventRegistry()
            >>> registry.register(
            ...     ModelEventRegistration(
            ...         event_type="custom.event",
            ...         topic_template="onex.evt.custom.event.v1",
            ...     )
            ... )
            >>> registry.resolve_topic("custom.event")
            'onex.evt.custom.event.v1'
        """
        self._registrations[registration.event_type] = registration

    def register_batch(self, registrations: Iterable[ModelEventRegistration]) -> None:
        """Register multiple event type mappings.

        Convenience method to register multiple event types in a single call.
        Each registration is added via ``register()``, overwriting existing
        registrations for the same event type.

        Args:
            registrations: Iterable of event registration configurations.

        Example:
            >>> registry = EventRegistry()
            >>> registry.register_batch([
            ...     ModelEventRegistration(
            ...         event_type="custom.one",
            ...         topic_template="onex.evt.custom.one.v1",
            ...     ),
            ...     ModelEventRegistration(
            ...         event_type="custom.two",
            ...         topic_template="onex.evt.custom.two.v1",
            ...     ),
            ... ])
            >>> registry.resolve_topic("custom.one")
            'onex.evt.custom.one.v1'
        """
        for registration in registrations:
            self.register(registration)

    def resolve_topic(self, event_type: str) -> str:
        """Get the Kafka topic for an event type (realm-agnostic).

        Topics are realm-agnostic in ONEX. The environment/realm is enforced via
        envelope identity, not topic naming. This enables cross-environment event
        routing when needed while maintaining proper isolation through identity.

        Args:
            event_type: Semantic event type identifier.

        Returns:
            Kafka topic name (no environment prefix).

        Raises:
            OnexError: If the event type is not registered.

        Example:
            >>> registry = EventRegistry()
            >>> registry.register(
            ...     ModelEventRegistration(
            ...         event_type="myapp.submitted",
            ...         topic_template="onex.evt.myapp.submitted.v1",
            ...     )
            ... )
            >>> registry.resolve_topic("myapp.submitted")
            'onex.evt.myapp.submitted.v1'
        """
        registration = self._registrations.get(event_type)
        if registration is None:
            registered = list(self._registrations.keys())
            raise OnexError(
                f"Unknown event type: '{event_type}'. Registered types: {registered}"
            )
        return registration.topic_template

    def get_partition_key(
        self,
        event_type: str,
        payload: dict[str, object],
    ) -> str | None:
        """Extract partition key from payload based on registration.

        Uses the configured partition_key_field to extract the value
        from the payload. Returns None if no partition key is configured
        or the field is not present in the payload.

        Args:
            event_type: Semantic event type identifier.
            payload: Event payload dictionary.

        Returns:
            Partition key value as string, or None if not applicable.

        Raises:
            OnexError: If the event type is not registered.

        Example:
            >>> registry = EventRegistry()
            >>> registry.register(
            ...     ModelEventRegistration(
            ...         event_type="myapp.submitted",
            ...         topic_template="onex.evt.myapp.submitted.v1",
            ...         partition_key_field="session_id",
            ...     )
            ... )
            >>> registry.get_partition_key(
            ...     "myapp.submitted",
            ...     {"session_id": "sess-123"},
            ... )
            'sess-123'
        """
        registration = self._registrations.get(event_type)
        if registration is None:
            registered = list(self._registrations.keys())
            raise OnexError(
                f"Unknown event type: '{event_type}'. Registered types: {registered}"
            )

        if registration.partition_key_field is None:
            return None

        value = payload.get(registration.partition_key_field)
        if value is None:
            return None

        return str(value)

    def validate_payload(
        self,
        event_type: str,
        payload: dict[str, object],
    ) -> bool:
        """Validate payload has all required fields.

        Checks that all fields specified in the registration's required_fields
        are present in the payload.

        Args:
            event_type: Semantic event type identifier.
            payload: Event payload dictionary to validate.

        Returns:
            True if validation passes.

        Raises:
            OnexError: If the event type is not registered or if any
                required field is missing from the payload.

        Example:
            >>> registry = EventRegistry()
            >>> registry.register(
            ...     ModelEventRegistration(
            ...         event_type="myapp.submitted",
            ...         topic_template="onex.evt.myapp.submitted.v1",
            ...         required_fields=["payload"],
            ...     )
            ... )
            >>> registry.validate_payload("myapp.submitted", {"payload": "data"})
            True
        """
        registration = self._registrations.get(event_type)
        if registration is None:
            registered = list(self._registrations.keys())
            raise OnexError(
                f"Unknown event type: '{event_type}'. Registered types: {registered}"
            )

        missing_fields = [
            field for field in registration.required_fields if field not in payload
        ]

        if missing_fields:
            raise OnexError(
                f"Missing required fields for '{event_type}': {missing_fields}"
            )

        return True

    def inject_metadata(
        self,
        event_type: str,
        payload: dict[str, object],
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> dict[str, object]:
        """Inject correlation_id, causation_id, emitted_at, and schema_version.

        Creates a new payload dictionary with metadata fields added.
        The original payload is not modified.

        Injected fields:
        - correlation_id: Trace ID for the event chain (auto-generated if None)
        - causation_id: ID of the event that caused this event (None if root event)
        - emitted_at: ISO-8601 timestamp of when the event was emitted
        - schema_version: Version of the event schema from registration

        Args:
            event_type: Semantic event type identifier.
            payload: Event payload dictionary to enrich.
            correlation_id: Optional correlation ID for tracing. If None,
                a new UUID will be generated.
            causation_id: Optional ID of the event that directly caused this event.
                This parameter enables event chain tracing by linking derived events
                back to their source. When None (the default), indicates this is a
                root event with no direct cause in the event stream.

        Returns:
            New dictionary with original payload plus injected metadata.

        Raises:
            OnexError: If the event type is not registered.

        Example:
            >>> registry = EventRegistry()
            >>> registry.register(
            ...     ModelEventRegistration(
            ...         event_type="myapp.submitted",
            ...         topic_template="onex.evt.myapp.submitted.v1",
            ...     )
            ... )
            >>> enriched = registry.inject_metadata(
            ...     "myapp.submitted",
            ...     {"data": "value"},
            ...     correlation_id="corr-123",
            ... )
            >>> enriched["correlation_id"]
            'corr-123'
            >>> enriched["causation_id"] is None
            True
        """
        registration = self._registrations.get(event_type)
        if registration is None:
            registered = list(self._registrations.keys())
            raise OnexError(
                f"Unknown event type: '{event_type}'. Registered types: {registered}"
            )

        # Create new dict with original payload
        enriched: dict[str, object] = dict(payload)

        # Inject metadata
        enriched["correlation_id"] = correlation_id or str(uuid4())
        enriched["causation_id"] = causation_id
        enriched["emitted_at"] = datetime.now(UTC).isoformat()
        enriched["schema_version"] = registration.schema_version

        return enriched

    def get_registration(self, event_type: str) -> ModelEventRegistration | None:
        """Get the registration for an event type.

        Args:
            event_type: Semantic event type identifier.

        Returns:
            The registration configuration, or None if not registered.

        Example:
            >>> registry = EventRegistry()
            >>> registry.register(
            ...     ModelEventRegistration(
            ...         event_type="myapp.submitted",
            ...         topic_template="onex.evt.myapp.submitted.v1",
            ...         partition_key_field="session_id",
            ...     )
            ... )
            >>> reg = registry.get_registration("myapp.submitted")
            >>> reg.topic_template
            'onex.evt.myapp.submitted.v1'
        """
        return self._registrations.get(event_type)

    def list_event_types(self) -> list[str]:
        """List all registered event types.

        Returns:
            List of registered event type identifiers.

        Example:
            >>> registry = EventRegistry()
            >>> registry.register(
            ...     ModelEventRegistration(
            ...         event_type="myapp.submitted",
            ...         topic_template="onex.evt.myapp.submitted.v1",
            ...     )
            ... )
            >>> "myapp.submitted" in registry.list_event_types()
            True
        """
        return list(self._registrations.keys())

    def compute_fingerprint(self) -> ModelEventRegistryFingerprint:
        """Compute deterministic fingerprint from all current registrations.

        Builds a canonical tuple per registration:
        ``(event_type, topic_template, schema_version, partition_key_field or '',
        sorted(required_fields))`` and SHA-256 hashes each element.  The overall
        fingerprint is the SHA-256 of the sorted list of
        ``(event_type, element_sha256)`` pairs.

        Invariant:
            ``topic_template`` MUST be a canonical ONEX 5-segment topic suffix
            (no env prefix, no dynamic expansion).  Fingerprinting a template
            that varies by environment produces env-dependent hashes, defeating
            the purpose.

        Returns:
            ``ModelEventRegistryFingerprint`` with overall hash and per-element
            details.
        """
        from omnibase_infra.runtime.emit_daemon.model_event_registry_fingerprint import (
            ModelEventRegistryFingerprint,
            ModelEventRegistryFingerprintElement,
        )

        elements: list[ModelEventRegistryFingerprintElement] = []

        for event_type in sorted(self._registrations):
            reg = self._registrations[event_type]
            canonical = (
                reg.event_type,
                reg.topic_template,
                reg.schema_version,
                reg.partition_key_field or "",
                tuple(sorted(reg.required_fields)),
            )
            element_hash = _sha256_json(canonical)
            elements.append(
                ModelEventRegistryFingerprintElement(
                    event_type=reg.event_type,
                    topic_template=reg.topic_template,
                    schema_version=reg.schema_version,
                    partition_key_field=reg.partition_key_field or "",
                    required_fields=tuple(sorted(reg.required_fields)),
                    element_sha256=element_hash,
                )
            )

        # Overall fingerprint: hash of sorted (event_type, element_sha256) pairs
        overall_input = [(e.event_type, e.element_sha256) for e in elements]
        overall_hash = _sha256_json(overall_input)

        return ModelEventRegistryFingerprint(
            version=1,
            fingerprint_sha256=overall_hash,
            elements=tuple(elements),
        )

    def assert_fingerprint(
        self,
        expected: ModelEventRegistryFingerprint,
    ) -> None:
        """Hard gate: compare live fingerprint to expected manifest.

        Computes the live fingerprint from current registrations and compares
        it to ``expected``.  On mismatch, raises
        ``EventRegistryFingerprintMismatchError`` with an actionable diff.

        This method is intended to be called immediately after
        ``register_batch()`` at startup, before any events are emitted.

        Args:
            expected: Parsed fingerprint manifest (typically loaded from an
                artifact file via ``ModelEventRegistryFingerprint.from_json_path``).

        Raises:
            EventRegistryFingerprintMismatchError: Live fingerprint differs
                from expected, with a bounded diff summary.
        """
        from omnibase_infra.errors.error_event_registry_fingerprint import (
            EventRegistryFingerprintMismatchError,
        )

        actual = self.compute_fingerprint()

        if actual.fingerprint_sha256 == expected.fingerprint_sha256:
            logger.info(
                "Event registry fingerprint validated: %s (%d registrations)",
                actual.fingerprint_sha256[:16],
                len(actual.elements),
            )
            return

        diff_summary = _compute_registry_diff(expected, actual)
        raise EventRegistryFingerprintMismatchError(
            f"Event registry fingerprint mismatch: "
            f"expected '{expected.fingerprint_sha256[:16]}...', "
            f"computed '{actual.fingerprint_sha256[:16]}...'. "
            "The live event registry does not match the expected manifest. "
            "Re-run 'stamp' to update the artifact after changing registrations.",
            expected_fingerprint=expected.fingerprint_sha256,
            actual_fingerprint=actual.fingerprint_sha256,
            diff_summary=diff_summary,
        )


# ---------------------------------------------------------------------------
# Module-level helpers for fingerprint computation
# ---------------------------------------------------------------------------


def _sha256_json(obj: object) -> str:
    """SHA-256 hex digest of a JSON-serialized object.

    Produces a deterministic hash by serializing with sorted keys and
    compact separators (no whitespace).

    Args:
        obj: Python object (dict, list, tuple) to hash. Must be JSON-serializable.

    Returns:
        64-character hexadecimal SHA-256 digest.
    """
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _compute_registry_diff(
    expected: ModelEventRegistryFingerprint,
    actual: ModelEventRegistryFingerprint,
) -> str:
    """Bounded diff summary (max 10 lines).

    Shows additions, removals, and modifications by event_type.  For
    modifications, lists which fields changed between expected and actual.

    Args:
        expected: The expected fingerprint manifest.
        actual: The live-computed fingerprint.

    Returns:
        Human-readable diff summary, truncated at 10 lines.
    """
    expected_by_type = {e.event_type: e for e in expected.elements}
    actual_by_type = {e.event_type: e for e in actual.elements}

    expected_types = set(expected_by_type)
    actual_types = set(actual_by_type)

    lines: list[str] = []

    for event_type in sorted(actual_types - expected_types):
        lines.append(f"  + added: {event_type}")

    for event_type in sorted(expected_types - actual_types):
        lines.append(f"  - removed: {event_type}")

    for event_type in sorted(expected_types & actual_types):
        exp_elem = expected_by_type[event_type]
        act_elem = actual_by_type[event_type]
        if exp_elem.element_sha256 != act_elem.element_sha256:
            changed_fields: list[str] = []
            if exp_elem.topic_template != act_elem.topic_template:
                changed_fields.append("topic_template")
            if exp_elem.schema_version != act_elem.schema_version:
                changed_fields.append("schema_version")
            if exp_elem.partition_key_field != act_elem.partition_key_field:
                changed_fields.append("partition_key_field")
            if exp_elem.required_fields != act_elem.required_fields:
                changed_fields.append("required_fields")
            field_detail = ", ".join(changed_fields) if changed_fields else "hash"
            lines.append(f"  ~ changed: {event_type} ({field_detail})")

    max_lines = 10
    if len(lines) > max_lines:
        overflow = len(lines) - (max_lines - 1)
        lines = lines[: max_lines - 1]
        lines.append(f"  ... and {overflow} more")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point -- stamp / verify subcommands (mirrors OMN-2087 pattern)
# ---------------------------------------------------------------------------

_ARTIFACT_DEFAULT_PATH: Final[str] = str(
    Path(__file__).parent / "event_registry_fingerprint.json"
)


def validate_event_registry_fingerprint(
    artifact_path: str = "",
) -> None:
    """Hard gate: validate event registry against expected fingerprint artifact.

    Builds a canonical registry from ``ALL_EVENT_REGISTRATIONS``, loads the
    expected fingerprint from the artifact file, and asserts they match.
    Raises immediately on mismatch or missing artifact.

    This function is the startup integration point. Call it during service
    initialization, after all event registrations are known but before any
    events are emitted.

    Args:
        artifact_path: Path to the expected fingerprint artifact. If empty,
            defaults to ``_ARTIFACT_DEFAULT_PATH`` (co-located with module).

    Raises:
        EventRegistryFingerprintMismatchError: Live registry != expected.
        EventRegistryFingerprintMissingError: Artifact file not found.
    """
    from omnibase_infra.errors.error_event_registry_fingerprint import (
        EventRegistryFingerprintMissingError,
    )
    from omnibase_infra.runtime.emit_daemon.model_event_registry_fingerprint import (
        ModelEventRegistryFingerprint,
    )
    from omnibase_infra.runtime.emit_daemon.topics import ALL_EVENT_REGISTRATIONS

    dest = Path(artifact_path) if artifact_path else Path(_ARTIFACT_DEFAULT_PATH)
    if not dest.exists():
        raise EventRegistryFingerprintMissingError(
            f"Event registry fingerprint artifact not found: {dest}. "
            "Run 'stamp' to generate the artifact.",
            artifact_path=str(dest),
        )

    expected = ModelEventRegistryFingerprint.from_json_path(dest)

    registry = EventRegistry()
    registry.register_batch(ALL_EVENT_REGISTRATIONS)

    registry.assert_fingerprint(expected)
    logger.info(
        "Event registry fingerprint validated: %s (%d registrations)",
        expected.fingerprint_sha256[:16],
        len(expected.elements),
    )


def _cli_stamp(artifact_path: str, *, dry_run: bool = False) -> None:
    """Compute fingerprint from known registrations and write artifact.

    Populates a registry with all known event registrations, computes the
    deterministic fingerprint, and writes the JSON artifact to disk.

    Args:
        artifact_path: Destination path for the JSON artifact.
        dry_run: If True, compute and print but do not write the file.
    """
    from omnibase_infra.runtime.emit_daemon.topics import ALL_EVENT_REGISTRATIONS

    registry = EventRegistry()
    registry.register_batch(ALL_EVENT_REGISTRATIONS)

    fingerprint = registry.compute_fingerprint()

    print(f"fingerprint: {fingerprint.fingerprint_sha256}")
    print(f"registrations: {len(fingerprint.elements)}")

    for elem in fingerprint.elements:
        print(f"  {elem.event_type}: {elem.element_sha256[:16]}...")

    if dry_run:
        print("\n--dry-run: skipping artifact write")
        return

    dest = Path(artifact_path)
    fingerprint.to_json(dest)
    print(f"\nArtifact written to {dest}")


def _cli_verify(artifact_path: str) -> None:
    """Load artifact, compute live fingerprint, compare.

    Args:
        artifact_path: Path to the expected fingerprint artifact file.

    Raises:
        EventRegistryFingerprintMismatchError: On fingerprint mismatch.
        EventRegistryFingerprintMissingError: If artifact file not found.
    """
    validate_event_registry_fingerprint(artifact_path=artifact_path)
    print("Event registry fingerprint OK")


def _main() -> None:
    """CLI entry point: stamp / verify subcommands."""
    import argparse
    import sys

    from omnibase_infra.errors.error_event_registry_fingerprint import (
        EventRegistryFingerprintMismatchError,
        EventRegistryFingerprintMissingError,
    )

    parser = argparse.ArgumentParser(
        prog="python -m omnibase_infra.runtime.emit_daemon.event_registry",
        description="Event registry fingerprint CLI for omnibase_infra (OMN-2088).",
    )
    sub = parser.add_subparsers(dest="command")

    stamp_parser = sub.add_parser(
        "stamp",
        help="Compute fingerprint from known registrations and write artifact.",
    )
    stamp_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute fingerprint but do not write artifact.",
    )
    stamp_parser.add_argument(
        "--artifact",
        default=_ARTIFACT_DEFAULT_PATH,
        help=f"Path to fingerprint artifact (default: {_ARTIFACT_DEFAULT_PATH}).",
    )

    verify_parser = sub.add_parser(
        "verify",
        help="Validate live registry matches expected fingerprint artifact.",
    )
    verify_parser.add_argument(
        "--artifact",
        default=_ARTIFACT_DEFAULT_PATH,
        help=f"Path to fingerprint artifact (default: {_ARTIFACT_DEFAULT_PATH}).",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "stamp":
            _cli_stamp(args.artifact, dry_run=args.dry_run)
        elif args.command == "verify":
            _cli_verify(args.artifact)
    except (
        EventRegistryFingerprintMismatchError,
        EventRegistryFingerprintMissingError,
    ) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main()


__all__: list[str] = [
    "EventRegistry",
    "ModelEventRegistration",
    "validate_event_registry_fingerprint",
]
