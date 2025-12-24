# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests verifying orchestrator uses injected `now`, not system clock.

These tests ensure the NodeRegistrationOrchestrator follows ONEX time injection
rules by:
1. NOT calling datetime.now() or datetime.utcnow() in node.py
2. NOT calling time.time() or time.monotonic() in node.py
3. Declaring time_injection configuration in contract.yaml
4. Using "RuntimeTick" as the time injection source
5. Configuring the evaluate_timeout step with time_injection: true
6. Consuming RuntimeTick events for time-based operations

Ticket: OMN-952
Related: OMN-973 (DispatchContextEnforcer implementation)

ONEX Architecture Constraint:
    Orchestrators receive `now` from the dispatch context (via DispatchContextEnforcer),
    NOT from system clock calls. This ensures:
    - Testability: Time can be injected in tests for deterministic behavior
    - Consistency: All time-dependent decisions use the same dispatch-time timestamp
    - Replay safety: Event replay doesn't produce different results due to clock drift

Design Pattern:
    The DispatchContextEnforcer creates ModelDispatchContext with `now` for orchestrators
    at dispatch time. The orchestrator receives this injected time and uses it for
    all time-dependent operations (timeout evaluation, deadline calculation, etc.).

See Also:
    - src/omnibase_infra/runtime/dispatch_context_enforcer.py
    - tests/unit/runtime/test_dispatch_context_enforcer.py
"""

from __future__ import annotations

import ast
import re

import pytest

__all__ = [
    "TestOrchestratorNoSystemClockCalls",
    "TestContractTimeInjectionConfiguration",
]

# =============================================================================
# Time-Dependency Detection
# =============================================================================
# This module uses keyword-based heuristics to identify workflow steps that
# likely perform time-dependent operations. These steps should declare
# `time_injection: true` in their step_config.
#
# Detection Strategy:
#   1. Pattern matching with word boundaries to avoid substring false positives
#   2. Partial word stems (e.g., "expir", "schedul") to catch variations
#   3. Exclusion list for known false positives (async patterns, etc.)
#   4. Check step_config for time-related keys as secondary indicator
#
# Known Limitations:
#   - May miss custom time-related operations with non-standard naming
#   - Domain-specific time concepts may require additions to patterns
#   - Heuristic approach cannot guarantee 100% accuracy
#
# To add new patterns: Add regex to TIME_KEYWORD_PATTERNS
# To exclude false positives: Add node_id to TIME_KEYWORD_EXCEPTIONS

# Regex patterns that suggest time-dependent operations.
# Uses word boundaries (\b) where appropriate to avoid substring matches.
# Partial stems (e.g., "expir", "schedul") intentionally lack trailing \b
# to match variations like "expire", "expired", "expiration", etc.
TIME_KEYWORD_PATTERNS: tuple[str, ...] = (
    r"\btimeout\b",  # timeout handling
    r"\bdeadline\b",  # deadline enforcement
    r"\bschedul",  # schedule, scheduled, scheduling
    r"\bexpir",  # expire, expired, expiration, expiry
    r"\bduration\b",  # duration calculations
    r"\bdelay\b",  # delay operations
    r"\bttl\b",  # time-to-live
    r"\bwait\b",  # wait operations (not "await" due to \b)
    r"\bretry.*interval\b",  # retry intervals
    r"\brate.?limit",  # rate limiting
    r"\bcooldown\b",  # cooldown periods
    r"\bthrottle\b",  # throttling
)

# Known exceptions: node_ids that match keywords but are NOT time-dependent.
# These are typically async programming patterns or domain terms that
# coincidentally contain time-related substrings.
TIME_KEYWORD_EXCEPTIONS: frozenset[str] = frozenset(
    {
        # Add known false positives here as they are discovered.
        # Example: "await_response" would match "wait" but is async pattern
    }
)

# Step config keys that indicate time-dependency when present.
# If a step declares these keys, it likely performs time-dependent operations.
TIME_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "timeout_seconds",
        "timeout_ms",
        "deadline",
        "ttl",
        "max_duration",
        "retry_delay",
        "cooldown_period",
    }
)


def _is_time_dependent(node_id: str, description: str, step_config: dict) -> bool:
    """Check if a workflow node appears to be time-dependent.

    Uses multiple signals to determine time-dependency:
    1. Node ID matches time-related keyword patterns
    2. Description contains time-related keyword patterns
    3. Step config contains time-related configuration keys

    Args:
        node_id: The workflow node identifier.
        description: The node's description text.
        step_config: The node's step_config dictionary.

    Returns:
        True if the node appears to be time-dependent.
    """
    # Check exclusion list first
    if node_id in TIME_KEYWORD_EXCEPTIONS:
        return False

    # Check for time-related config keys
    if step_config and any(key in step_config for key in TIME_CONFIG_KEYS):
        return True

    # Check keyword patterns in combined text
    combined_text = f"{node_id} {description}".lower()
    return any(re.search(pattern, combined_text) for pattern in TIME_KEYWORD_PATTERNS)


# =============================================================================
# Fixtures
# =============================================================================
# Note: The following fixtures are provided by conftest.py with module-level
# scope for performance (parse once per module):
#   - contract_path, contract_data: Contract loading
#   - node_module_path, node_source_code, node_ast: Node source/AST parsing


# =============================================================================
# AST Analysis Helpers
# =============================================================================
# These utilities are imported from the shared test helpers module.
# See: tests/helpers/ast_analysis.py for implementation details.

from tests.helpers.ast_analysis import (
    find_datetime_now_calls,
    find_time_module_calls,
    get_imported_root_modules,
)


# Alias for backward compatibility with existing test code in this file
_find_datetime_now_calls = find_datetime_now_calls
_find_time_module_calls = find_time_module_calls
_find_imports = get_imported_root_modules


# =============================================================================
# TestOrchestratorNoSystemClockCalls - AST Analysis Tests
# =============================================================================


class TestOrchestratorNoSystemClockCalls:
    """Tests verifying orchestrator node.py does not call system clock functions.

    ONEX Architecture Rule:
        Orchestrators MUST use injected `now` from the dispatch context, not
        from system clock functions like datetime.now() or time.time().

    Why This Matters:
        - System clock calls break testability (can't inject fixed time)
        - System clock calls break replay determinism (different results on replay)
        - DispatchContextEnforcer provides `now` at dispatch time for orchestrators

    Implementation:
        These tests use AST analysis to statically verify no forbidden clock
        function calls exist in the orchestrator node.py. This is a compile-time
        guarantee that prevents introduction of clock dependencies.
    """

    def test_orchestrator_no_datetime_now_calls(self, node_ast: ast.AST) -> None:
        """Verify node.py does not call datetime.now() or datetime.utcnow().

        This is a CRITICAL test for OMN-952 acceptance criteria:
        "Uses injected now, not system clock"

        The orchestrator must receive time via the dispatch context, not by
        calling datetime.now() directly. This ensures:
        - Testability: Tests can inject specific timestamps
        - Determinism: Same dispatch context produces same behavior
        - Consistency: All time-dependent logic uses the same base time

        AST Patterns Detected:
        - datetime.datetime.now() - Full module path
        - datetime.now() - After 'from datetime import datetime'
        - datetime.datetime.utcnow() - Deprecated but still forbidden
        - datetime.utcnow() - Deprecated but still forbidden
        """
        violations = _find_datetime_now_calls(node_ast)

        assert len(violations) == 0, (
            "CRITICAL: Orchestrator node.py calls datetime.now() or datetime.utcnow()!\n"
            "Violations found:\n"
            "  - " + "\n  - ".join(violations) + "\n\n"
            "Orchestrators MUST use injected `now` from the dispatch context, "
            "not from system clock calls.\n"
            "See: src/omnibase_infra/runtime/dispatch_context_enforcer.py\n"
            "Ticket: OMN-952, OMN-973"
        )

    def test_orchestrator_no_time_time_calls(self, node_ast: ast.AST) -> None:
        """Verify node.py does not call time.time() or time.monotonic().

        Similar to datetime.now(), the time module's clock functions are also
        forbidden in orchestrator code. Orchestrators must use the injected
        `now` from the dispatch context for all time-dependent operations.

        AST Patterns Detected:
        - time.time() - Unix timestamp
        - time.monotonic() - Monotonic clock
        - time.perf_counter() - Performance counter

        Note: time.sleep() is NOT forbidden as it's used for delays, not
        for reading the current time. However, orchestrators typically
        shouldn't use sleep either (use async patterns instead).
        """
        violations = _find_time_module_calls(node_ast)

        assert len(violations) == 0, (
            "CRITICAL: Orchestrator node.py calls time module clock functions!\n"
            "Violations found:\n"
            "  - " + "\n  - ".join(violations) + "\n\n"
            "Orchestrators MUST use injected `now` from the dispatch context, "
            "not from time.time() or similar functions.\n"
            "See: src/omnibase_infra/runtime/dispatch_context_enforcer.py\n"
            "Ticket: OMN-952, OMN-973"
        )

    def test_orchestrator_minimal_imports(self, node_ast: ast.AST) -> None:
        """Verify orchestrator has minimal imports (declarative pattern).

        The declarative orchestrator pattern means workflow behavior is 100%
        driven by contract.yaml, not Python code. Therefore, the node.py
        should have minimal imports and no time-related modules.

        This test serves as an additional safety check - if datetime or time
        modules are imported, it suggests the orchestrator may be doing
        time-related operations that should use injected time instead.
        """
        imports = _find_imports(node_ast)

        # Time-related modules that suggest clock usage
        time_related_imports = {
            "time"
        }  # Note: 'datetime' might be imported for type hints

        suspicious_imports = imports & time_related_imports

        # 'time' module import is a red flag - likely used for time.time() or sleep
        assert "time" not in imports, (
            "Orchestrator imports the 'time' module. "
            "This suggests time-related operations that may violate "
            "the injected-time-only rule for orchestrators."
        )


# =============================================================================
# TestContractTimeInjectionConfiguration - Contract Validation Tests
# =============================================================================


class TestContractTimeInjectionConfiguration:
    """Tests verifying contract.yaml declares time injection configuration.

    The orchestrator contract.yaml must declare how time is injected for
    time-dependent workflow steps. This configuration tells the runtime
    that the orchestrator expects to receive `now` from specific sources.

    Contract Configuration Structure:
        time_injection:
          enabled: true
          source: "RuntimeTick"
          field: "now"
          description: "Use injected timestamp from RuntimeTick for timeout evaluation"

        execution_graph:
          nodes:
            - node_id: "evaluate_timeout"
              step_config:
                time_injection: true
    """

    def test_contract_declares_time_injection(self, contract_data: dict) -> None:
        """Verify contract.yaml has time_injection configuration.

        The time_injection section in contract.yaml declares that this
        orchestrator uses injected time for its operations. This is the
        contract-level declaration that enables time injection at dispatch.

        Expected Structure:
            time_injection:
              enabled: true
              source: <source_name>
              field: <field_name>
        """
        assert "time_injection" in contract_data, (
            "Contract is missing 'time_injection' configuration.\n"
            "Orchestrators that need time must declare time_injection in contract.yaml.\n"
            "See: ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md"
        )

        time_injection = contract_data["time_injection"]

        # Verify required fields exist
        assert "enabled" in time_injection, "time_injection must have 'enabled' field"
        assert time_injection["enabled"] is True, (
            "time_injection.enabled must be true for orchestrators that need time"
        )

        assert "source" in time_injection, "time_injection must have 'source' field"
        assert "field" in time_injection, "time_injection must have 'field' field"

    def test_time_injection_source_is_runtime_tick(self, contract_data: dict) -> None:
        """Verify time_injection.source is "RuntimeTick".

        The RuntimeTick is the canonical source for time injection in ONEX.
        RuntimeTick events are internal ticks that carry the current timestamp
        for time-dependent operations like timeout evaluation.

        Why RuntimeTick:
        - Provides consistent time across all workflow steps in a tick
        - Enables time-based decisions without system clock access
        - Supports testing with fixed/mocked time values
        """
        assert "time_injection" in contract_data, (
            "Contract is missing 'time_injection' configuration"
        )

        time_injection = contract_data["time_injection"]

        assert "source" in time_injection, "time_injection must have 'source' field"

        source = time_injection["source"]
        assert source == "RuntimeTick", (
            f"time_injection.source must be 'RuntimeTick', got '{source}'.\n"
            "RuntimeTick is the canonical time source for orchestrator timeout evaluation."
        )

    def test_time_injection_field_is_now(self, contract_data: dict) -> None:
        """Verify time_injection.field is "now".

        The 'now' field is the standard field name for injected time in
        dispatch contexts and RuntimeTick events.
        """
        assert "time_injection" in contract_data, (
            "Contract is missing 'time_injection' configuration"
        )

        time_injection = contract_data["time_injection"]

        assert "field" in time_injection, "time_injection must have 'field' field"

        field = time_injection["field"]
        assert field == "now", (
            f"time_injection.field must be 'now', got '{field}'.\n"
            "The 'now' field is the standard name for injected time."
        )

    def test_evaluate_timeout_step_uses_injected_time(
        self, contract_data: dict
    ) -> None:
        """Verify the evaluate_timeout step has time_injection: true in step_config.

        The evaluate_timeout workflow step is specifically designed to use
        injected time for timeout evaluation. Its step_config must declare
        that it uses time injection.

        Expected Structure:
            execution_graph:
              nodes:
                - node_id: "evaluate_timeout"
                  step_config:
                    time_injection: true
                    timeout_evaluation: true
        """
        assert "workflow_coordination" in contract_data, (
            "Contract is missing 'workflow_coordination'"
        )

        workflow = contract_data["workflow_coordination"]
        assert "workflow_definition" in workflow, (
            "workflow_coordination is missing 'workflow_definition'"
        )

        workflow_def = workflow["workflow_definition"]
        assert "execution_graph" in workflow_def, (
            "workflow_definition is missing 'execution_graph'"
        )

        execution_graph = workflow_def["execution_graph"]
        assert "nodes" in execution_graph, "execution_graph is missing 'nodes'"

        nodes = execution_graph["nodes"]

        # Find the evaluate_timeout node
        evaluate_timeout_node = None
        for node in nodes:
            if node.get("node_id") == "evaluate_timeout":
                evaluate_timeout_node = node
                break

        assert evaluate_timeout_node is not None, (
            "Contract execution_graph is missing 'evaluate_timeout' node.\n"
            "This node is required for time-based timeout evaluation."
        )

        # Verify step_config exists and has time_injection: true
        assert "step_config" in evaluate_timeout_node, (
            "evaluate_timeout node is missing 'step_config'"
        )

        step_config = evaluate_timeout_node["step_config"]

        assert step_config.get("time_injection") is True, (
            "evaluate_timeout.step_config.time_injection must be true.\n"
            "This declares that the step uses injected time for timeout evaluation."
        )

        # Also verify timeout_evaluation is true (related configuration)
        assert step_config.get("timeout_evaluation") is True, (
            "evaluate_timeout.step_config.timeout_evaluation must be true.\n"
            "This declares that the step performs timeout evaluation."
        )

    def test_runtime_tick_event_consumed(self, contract_data: dict) -> None:
        """Verify contract consumes RuntimeTick events for time injection.

        RuntimeTick events are internal events that carry the current timestamp.
        The orchestrator must consume these events to receive time injection
        for timeout evaluation and other time-dependent operations.

        Expected Structure in consumed_events:
            - topic: "{env}.{namespace}.onex.internal.runtime-tick.v1"
              event_type: "RuntimeTick"
              internal: true
        """
        assert "consumed_events" in contract_data, (
            "Contract is missing 'consumed_events'"
        )

        consumed_events = contract_data["consumed_events"]

        # Find RuntimeTick in consumed events
        runtime_tick_found = False
        runtime_tick_event = None

        for event in consumed_events:
            if event.get("event_type") == "RuntimeTick":
                runtime_tick_found = True
                runtime_tick_event = event
                break

        assert runtime_tick_found, (
            "Contract consumed_events does not include RuntimeTick.\n"
            "Orchestrators that use time injection must consume RuntimeTick events.\n"
            "Expected: event_type='RuntimeTick' in consumed_events"
        )

        # Verify the RuntimeTick event configuration
        assert runtime_tick_event is not None

        # RuntimeTick should be marked as internal
        assert runtime_tick_event.get("internal") is True, (
            "RuntimeTick event should be marked as internal=true.\n"
            "RuntimeTick is an internal infrastructure event, not an external event."
        )

        # Verify topic pattern includes 'runtime-tick'
        topic = runtime_tick_event.get("topic", "")
        assert "runtime-tick" in topic.lower(), (
            f"RuntimeTick topic pattern should contain 'runtime-tick', got: {topic}"
        )


# =============================================================================
# TestOrchestratorTimeInjectionIntegration - Integration-style Tests
# =============================================================================


class TestOrchestratorTimeInjectionIntegration:
    """Integration-style tests verifying time injection end-to-end.

    These tests verify the relationship between the orchestrator's contract
    configuration and the ONEX runtime's time injection mechanisms.
    """

    def test_contract_and_node_consistent(
        self, node_ast: ast.AST, contract_data: dict
    ) -> None:
        """Verify contract declares time injection AND node doesn't use clock.

        This is the comprehensive test that verifies both sides:
        1. Contract declares time_injection (the "what")
        2. Node.py doesn't call clock functions (the "how")

        Together, these ensure the orchestrator follows ONEX time injection rules.
        """
        # Verify contract declares time injection
        assert "time_injection" in contract_data, "Contract must declare time_injection"
        assert contract_data["time_injection"].get("enabled") is True

        # Verify node.py has no clock calls
        datetime_violations = _find_datetime_now_calls(node_ast)
        time_violations = _find_time_module_calls(node_ast)

        all_violations = datetime_violations + time_violations
        assert len(all_violations) == 0, (
            "Orchestrator declares time_injection in contract but node.py "
            "still has system clock calls:\n"
            "  - " + "\n  - ".join(all_violations)
        )

    def test_all_time_dependent_steps_configured(self, contract_data: dict) -> None:
        """Verify all time-dependent workflow steps have time_injection config.

        Steps that need time for their operations should declare
        time_injection: true in their step_config. This test identifies
        potentially time-dependent steps and verifies they're configured.

        Detection uses the module-level _is_time_dependent() function which:
        - Matches time-related keywords with word boundaries (regex-based)
        - Checks step_config for time-related configuration keys
        - Respects an exclusion list for known false positives

        See TIME_KEYWORD_PATTERNS, TIME_KEYWORD_EXCEPTIONS, and TIME_CONFIG_KEYS
        at module level for configuration.
        """
        assert "workflow_coordination" in contract_data
        workflow = contract_data["workflow_coordination"]
        assert "workflow_definition" in workflow
        workflow_def = workflow["workflow_definition"]
        assert "execution_graph" in workflow_def
        nodes = workflow_def["execution_graph"]["nodes"]

        time_dependent_steps_without_config: list[str] = []

        for node in nodes:
            node_id = node.get("node_id", "")
            description = node.get("description", "")
            step_config = node.get("step_config", {})

            # Use helper function for robust time-dependency detection
            if _is_time_dependent(node_id, description, step_config):
                if not step_config.get("time_injection"):
                    time_dependent_steps_without_config.append(
                        f"{node_id}: {description}"
                    )

        assert len(time_dependent_steps_without_config) == 0, (
            "Time-dependent workflow steps missing time_injection config:\n"
            "  - " + "\n  - ".join(time_dependent_steps_without_config) + "\n\n"
            "Add 'time_injection: true' to step_config for these steps.\n"
            "If this is a false positive, add node_id to TIME_KEYWORD_EXCEPTIONS."
        )
