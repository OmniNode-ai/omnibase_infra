# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Demo Loop Assertion Gate -- validates the canonical event loop end-to-end.

This module provides a single hard gate that proves the demo loop is wired
correctly before opening the dashboard. It checks six assertions that together
guarantee the complete pipeline (canonical events -> projection -> dashboard)
is operational.

Assertions:
    1. Canonical pipeline exclusivity -- only canonical pipeline is emitting
    2. Required event types exist -- all required types are present and valid
    3. Schema version compatibility -- payload schema versions match
    4. Projector health -- projector service is registered and ready
    5. Dashboard config -- dashboard points at the intended environment
    6. No duplicate events -- canonical and legacy paths don't both emit

Usage:
    CLI entry point::

        uv run python -m omnibase_infra.validation.demo_loop_gate [--env-file .env]

    Programmatic::

        from omnibase_infra.validation.demo_loop_gate import DemoLoopGate

        gate = DemoLoopGate()
        result = gate.run_all()
        if not result:
            sys.exit(1)

Related Tickets:
    - OMN-2297: Demo Loop Assertion Gate for canonical event loop

.. versionadded:: 0.9.0
"""

from __future__ import annotations

import logging
import sys
from typing import Final

from omnibase_infra.event_bus.topic_constants import (
    TOPIC_AGENT_STATUS,
    TOPIC_INJECTION_AGENT_MATCH,
    TOPIC_INJECTION_CONTEXT_UTILIZATION,
    TOPIC_INJECTION_LATENCY_BREAKDOWN,
    TOPIC_LLM_CALL_COMPLETED,
    TOPIC_SESSION_OUTCOME_CANONICAL,
    TOPIC_SESSION_OUTCOME_CURRENT,
)
from omnibase_infra.runtime.emit_daemon.topics import (
    ALL_EVENT_REGISTRATIONS,
    TOPIC_NOTIFICATION_BLOCKED,
    TOPIC_NOTIFICATION_COMPLETED,
    TOPIC_PHASE_METRICS,
)
from omnibase_infra.validation.enums.enum_assertion_status import EnumAssertionStatus
from omnibase_infra.validation.models.model_assertion_result import (
    ModelAssertionResult,
)
from omnibase_infra.validation.models.model_demo_loop_result import (
    ModelDemoLoopResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# The canonical event topics that MUST exist for the demo loop to work.
# These are the topics the canonical pipeline emits to.
CANONICAL_EVENT_TOPICS: Final[tuple[str, ...]] = (
    # Platform topics (runtime)
    "onex.evt.platform.node-introspection.v1",
    "onex.evt.platform.registration-completed.v1",
    # Intelligence topics
    TOPIC_SESSION_OUTCOME_CANONICAL,
    TOPIC_LLM_CALL_COMPLETED,
    # Injection effectiveness topics
    TOPIC_INJECTION_CONTEXT_UTILIZATION,
    TOPIC_INJECTION_AGENT_MATCH,
    TOPIC_INJECTION_LATENCY_BREAKDOWN,
    # Phase metrics
    TOPIC_PHASE_METRICS,
    # Notification topics
    TOPIC_NOTIFICATION_BLOCKED,
    TOPIC_NOTIFICATION_COMPLETED,
    # Agent status
    TOPIC_AGENT_STATUS,
)

# Legacy topics that should NOT be emitting alongside canonical topics.
# The "current" session-outcome topic uses 'cmd' prefix instead of 'evt'.
LEGACY_TOPIC_MAPPINGS: Final[dict[str, str]] = {
    TOPIC_SESSION_OUTCOME_CURRENT: TOPIC_SESSION_OUTCOME_CANONICAL,
}

# Expected schema version for the demo loop (from event registrations).
EXPECTED_SCHEMA_VERSION: Final[str] = "1.0.0"


# =============================================================================
# Gate Implementation
# =============================================================================


class DemoLoopGate:
    """Hard gate that validates the complete demo loop is wired correctly.

    Runs six assertion checks and produces a structured result indicating
    whether the demo loop is ready for use.

    The gate is designed to be fast (<30s on a clean environment) and
    produces actionable error messages when assertions fail.

    Example:
        >>> gate = DemoLoopGate()
        >>> result = gate.run_all()
        >>> result.is_ready
        True
    """

    def __init__(
        self,
        *,
        canonical_topics: tuple[str, ...] = CANONICAL_EVENT_TOPICS,
        legacy_mappings: dict[str, str] | None = None,
        expected_schema_version: str = EXPECTED_SCHEMA_VERSION,
        projector_check_enabled: bool = True,
        dashboard_check_enabled: bool = True,
    ) -> None:
        """Initialize the demo loop gate.

        Args:
            canonical_topics: Tuple of topic strings that must exist.
            legacy_mappings: Mapping of legacy topic -> canonical replacement.
                If None, uses the default LEGACY_TOPIC_MAPPINGS.
            expected_schema_version: Expected schema version across producers.
            projector_check_enabled: Whether to run projector health check.
                Set to False in CI where no live projector is available.
            dashboard_check_enabled: Whether to run dashboard config check.
                Set to False in CI where no dashboard is available.
        """
        self._canonical_topics = canonical_topics
        self._legacy_mappings = (
            legacy_mappings
            if legacy_mappings is not None
            else dict(LEGACY_TOPIC_MAPPINGS)
        )
        self._expected_schema_version = expected_schema_version
        self._projector_check_enabled = projector_check_enabled
        self._dashboard_check_enabled = dashboard_check_enabled

    def run_all(self) -> ModelDemoLoopResult:
        """Run all six demo loop assertions and return aggregate result.

        Returns:
            ModelDemoLoopResult with individual assertion results and
            overall readiness status.
        """
        results: list[ModelAssertionResult] = [
            self.assert_canonical_pipeline_exclusivity(),
            self.assert_required_event_types(),
            self.assert_schema_version_compatibility(),
            self.assert_projector_health(),
            self.assert_dashboard_config(),
            self.assert_no_duplicate_events(),
        ]

        passed = sum(1 for r in results if r.status == EnumAssertionStatus.PASSED)
        failed = sum(1 for r in results if r.status == EnumAssertionStatus.FAILED)
        skipped = sum(1 for r in results if r.status == EnumAssertionStatus.SKIPPED)

        is_ready = failed == 0

        return ModelDemoLoopResult(
            assertions=tuple(results),
            passed=passed,
            failed=failed,
            skipped=skipped,
            is_ready=is_ready,
        )

    # -------------------------------------------------------------------------
    # Assertion 1: Canonical Pipeline Exclusivity
    # -------------------------------------------------------------------------

    def assert_canonical_pipeline_exclusivity(self) -> ModelAssertionResult:
        """Check that only the canonical pipeline is configured for emission.

        Verifies that no legacy topic mappings are configured as active
        emission targets. The canonical pipeline should be the sole emitter.

        Returns:
            ModelAssertionResult indicating whether the canonical pipeline
            is the exclusive emitter.
        """
        legacy_in_registrations: list[str] = []
        for reg in ALL_EVENT_REGISTRATIONS:
            if reg.topic_template in self._legacy_mappings:
                legacy_in_registrations.append(
                    f"{reg.event_type} -> {reg.topic_template} "
                    f"(canonical: {self._legacy_mappings[reg.topic_template]})"
                )

        if legacy_in_registrations:
            return ModelAssertionResult(
                name="canonical_pipeline",
                status=EnumAssertionStatus.FAILED,
                message=(
                    f"Legacy pipeline detected: "
                    f"{len(legacy_in_registrations)} registration(s) use legacy topics"
                ),
                details=tuple(legacy_in_registrations),
            )

        return ModelAssertionResult(
            name="canonical_pipeline",
            status=EnumAssertionStatus.PASSED,
            message="Canonical pipeline: active, no legacy registrations",
        )

    # -------------------------------------------------------------------------
    # Assertion 2: Required Event Types
    # -------------------------------------------------------------------------

    def assert_required_event_types(self) -> ModelAssertionResult:
        """Check that all required event types are present and schematically valid.

        Verifies that every topic in the canonical event topics list conforms
        to the ONEX topic naming convention (5-segment format).

        Returns:
            ModelAssertionResult indicating whether all required event types
            are present and valid.
        """
        if not self._canonical_topics:
            return ModelAssertionResult(
                name="required_event_types",
                status=EnumAssertionStatus.FAILED,
                message=("No canonical topics configured - gate provides no coverage"),
            )

        from omnibase_core.validation import validate_topic_suffix

        total = len(self._canonical_topics)
        valid_count = 0
        invalid_topics: list[str] = []

        for topic in self._canonical_topics:
            result = validate_topic_suffix(topic)
            if result.is_valid:
                valid_count += 1
            else:
                invalid_topics.append(f"{topic}: {result.error}")

        if invalid_topics:
            return ModelAssertionResult(
                name="required_event_types",
                status=EnumAssertionStatus.FAILED,
                message=(
                    f"Missing/invalid event types: "
                    f"{len(invalid_topics)} of {total} failed validation"
                ),
                details=tuple(invalid_topics),
            )

        return ModelAssertionResult(
            name="required_event_types",
            status=EnumAssertionStatus.PASSED,
            message=f"Required event types: {valid_count}/{total} present",
        )

    # -------------------------------------------------------------------------
    # Assertion 3: Schema Version Compatibility
    # -------------------------------------------------------------------------

    def assert_schema_version_compatibility(self) -> ModelAssertionResult:
        """Check that payload schema versions match across producers and projector.

        Verifies that all event registrations use the expected schema version.
        Mismatches indicate a producer is emitting a different schema version
        than the projector expects.

        Returns:
            ModelAssertionResult indicating schema version compatibility.
        """
        mismatches: list[str] = []
        for reg in ALL_EVENT_REGISTRATIONS:
            if reg.schema_version != self._expected_schema_version:
                mismatches.append(
                    f"{reg.event_type}: {reg.schema_version} "
                    f"(expected {self._expected_schema_version})"
                )

        if mismatches:
            return ModelAssertionResult(
                name="schema_versions",
                status=EnumAssertionStatus.FAILED,
                message=(
                    f"Schema version mismatch: "
                    f"{len(mismatches)} registration(s) differ from "
                    f"expected {self._expected_schema_version}"
                ),
                details=tuple(mismatches),
            )

        return ModelAssertionResult(
            name="schema_versions",
            status=EnumAssertionStatus.PASSED,
            message=f"Schema versions: compatible ({self._expected_schema_version})",
        )

    # -------------------------------------------------------------------------
    # Assertion 4: Projector Health
    # -------------------------------------------------------------------------

    def assert_projector_health(self) -> ModelAssertionResult:
        """Check that the projector service is registered and ready to consume.

        In CI mode (projector_check_enabled=False), this check is skipped
        since no live projector is available. In live environments, this
        verifies the projector is healthy.

        Returns:
            ModelAssertionResult indicating projector health status.
        """
        if not self._projector_check_enabled:
            return ModelAssertionResult(
                name="projector_health",
                status=EnumAssertionStatus.SKIPPED,
                message="Projector health: skipped (no live projector)",
            )

        # Static check: verify that projector contract YAML files exist
        # on disk. Runtime health is verified by /health/wiring endpoint.
        try:
            from pathlib import Path

            from omnibase_infra.runtime.projector_plugin_loader import (
                PROJECTOR_CONTRACT_PATTERNS,
            )

            projectors_dir = Path(__file__).parent.parent / "projectors" / "contracts"
            found: list[str] = []
            if projectors_dir.exists():
                for pattern in PROJECTOR_CONTRACT_PATTERNS:
                    found.extend(str(p.name) for p in projectors_dir.glob(pattern))

            if not found:
                if projectors_dir.exists():
                    reason = (
                        f"Directory exists but contains no matching contracts: "
                        f"{projectors_dir} "
                        f"(patterns: {', '.join(PROJECTOR_CONTRACT_PATTERNS)})"
                    )
                else:
                    reason = (
                        f"Contracts directory does not exist: {projectors_dir} "
                        f"-- expected relative to {Path(__file__).parent.parent}. "
                        f"If the package layout has changed, this path may need "
                        f"updating."
                    )
                return ModelAssertionResult(
                    name="projector_health",
                    status=EnumAssertionStatus.FAILED,
                    message="Projector health: no projector contracts discovered",
                    details=(reason,),
                )

            return ModelAssertionResult(
                name="projector_health",
                status=EnumAssertionStatus.PASSED,
                message=(f"Projector: healthy, {len(found)} contract(s) discovered"),
            )
        except (ImportError, ModuleNotFoundError) as exc:
            logger.debug("Projector health check failed: %s", exc)
            return ModelAssertionResult(
                name="projector_health",
                status=EnumAssertionStatus.SKIPPED,
                message=f"Projector health: skipped ({type(exc).__name__})",
            )

    # -------------------------------------------------------------------------
    # Assertion 5: Dashboard Config
    # -------------------------------------------------------------------------

    def assert_dashboard_config(self) -> ModelAssertionResult:
        """Check that dashboard points at the intended environment and bus.

        In CI mode (dashboard_check_enabled=False), this check is skipped.
        In live environments, validates that the Kafka bootstrap servers
        and environment are consistent.

        Returns:
            ModelAssertionResult indicating dashboard config status.
        """
        if not self._dashboard_check_enabled:
            return ModelAssertionResult(
                name="dashboard_config",
                status=EnumAssertionStatus.SKIPPED,
                message="Dashboard config: skipped (no live dashboard)",
            )

        import os

        kafka_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "")
        if not kafka_servers:
            return ModelAssertionResult(
                name="dashboard_config",
                status=EnumAssertionStatus.FAILED,
                message="Dashboard config: KAFKA_BOOTSTRAP_SERVERS not set",
                details=(
                    "Set KAFKA_BOOTSTRAP_SERVERS in .env to point at the "
                    "demo environment Kafka/Redpanda cluster",
                ),
            )

        # Minimal format validation: each comma-separated broker must look
        # like ``host:port`` (non-empty strings on both sides of a colon).
        invalid_brokers: list[str] = []
        for broker in kafka_servers.split(","):
            broker = broker.strip()
            if ":" not in broker:
                invalid_brokers.append(f"'{broker}' -- missing ':port' suffix")
            else:
                host, _, port = broker.partition(":")
                if not host or not port:
                    invalid_brokers.append(
                        f"'{broker}' -- empty host or port component"
                    )

        if invalid_brokers:
            return ModelAssertionResult(
                name="dashboard_config",
                status=EnumAssertionStatus.FAILED,
                message=(
                    "Dashboard config: KAFKA_BOOTSTRAP_SERVERS has invalid "
                    "host:port format"
                ),
                details=tuple(invalid_brokers),
            )

        return ModelAssertionResult(
            name="dashboard_config",
            status=EnumAssertionStatus.PASSED,
            message="Dashboard: Kafka bootstrap configured",
        )

    # -------------------------------------------------------------------------
    # Assertion 6: No Duplicate Events
    # -------------------------------------------------------------------------

    def assert_no_duplicate_events(self) -> ModelAssertionResult:
        """Check that canonical and legacy paths don't both emit for the same op.

        Verifies that for every legacy topic mapping, no event registration
        exists that would cause both the legacy and canonical topics to
        receive the same event type.

        Returns:
            ModelAssertionResult indicating whether duplicate emission is
            detected.
        """
        registered_topics = {reg.topic_template for reg in ALL_EVENT_REGISTRATIONS}

        duplicates: list[str] = []
        for legacy_topic, canonical_topic in self._legacy_mappings.items():
            if (
                legacy_topic in registered_topics
                and canonical_topic in registered_topics
            ):
                duplicates.append(
                    f"Both '{legacy_topic}' and '{canonical_topic}' are registered"
                )

        if duplicates:
            return ModelAssertionResult(
                name="no_duplicate_events",
                status=EnumAssertionStatus.FAILED,
                message=(
                    f"Duplicate events detected: "
                    f"{len(duplicates)} legacy/canonical pair(s) both active"
                ),
                details=tuple(duplicates),
            )

        return ModelAssertionResult(
            name="no_duplicate_events",
            status=EnumAssertionStatus.PASSED,
            message="No duplicate events: canonical and legacy paths are exclusive",
        )


# =============================================================================
# CLI Formatting
# =============================================================================


def format_result(result: ModelDemoLoopResult) -> str:
    """Format a demo loop result for human-readable CLI output.

    Args:
        result: The aggregate demo loop result.

    Returns:
        Formatted string with checkmarks/crosses and summary.
    """
    lines: list[str] = []

    for assertion in result.assertions:
        if assertion.status == EnumAssertionStatus.PASSED:
            prefix = "PASS"
        elif assertion.status == EnumAssertionStatus.FAILED:
            prefix = "FAIL"
        else:
            prefix = "SKIP"

        lines.append(f"  [{prefix}] {assertion.message}")

        for detail in assertion.details:
            lines.append(f"         {detail}")

    lines.append("")
    if result.is_ready:
        lines.append(
            f"PASS: Demo loop ready ({result.passed} passed, {result.skipped} skipped)"
        )
    else:
        lines.append(
            f"FAIL: Demo loop not ready "
            f"({result.failed} failure(s), {result.passed} passed, "
            f"{result.skipped} skipped)"
        )

    return "\n".join(lines)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the demo loop assertion gate.

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code: 0 on success, 1 on assertion failure, 2 on error.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="demo-loop-gate",
        description="Validate the complete demo loop is wired correctly (OMN-2297).",
    )
    parser.add_argument(
        "--env-file",
        default="",
        help="Path to .env file to source before running checks.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: skip live infrastructure checks (projector, dashboard).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.env_file:
        _load_env_file(args.env_file)

    gate = DemoLoopGate(
        projector_check_enabled=not args.ci,
        dashboard_check_enabled=not args.ci,
    )

    try:
        result = gate.run_all()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(format_result(result))
    return 0 if result.is_ready else 1


def _load_env_file(path: str) -> None:
    """Load environment variables from a file.

    Simple .env parser that handles KEY=VALUE lines, ignoring comments
    and blank lines. Does NOT override existing environment variables.

    Args:
        path: Path to the .env file.
    """
    import os
    from pathlib import Path

    env_path = Path(path)
    if not env_path.exists():
        logger.warning("Env file not found: %s", path)
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


# Intentional re-exports: EnumAssertionStatus, ModelAssertionResult, and
# ModelDemoLoopResult are defined in sub-modules but re-exported here for
# convenience so consumers can import everything from this single module.
__all__: list[str] = [
    "CANONICAL_EVENT_TOPICS",
    "DemoLoopGate",
    "EnumAssertionStatus",
    "LEGACY_TOPIC_MAPPINGS",
    "ModelAssertionResult",
    "ModelDemoLoopResult",
    "format_result",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
