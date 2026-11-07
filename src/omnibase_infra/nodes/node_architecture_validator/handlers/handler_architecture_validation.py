# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for architecture validation operations.

This handler processes architecture validation requests by validating nodes and
handlers against configurable architecture rules. It implements the validation
logic that was previously in NodeArchitectureValidatorCompute.

Design Pattern:
    HandlerArchitectureValidation follows the ONEX handler pattern:
    - Receives strongly-typed request (ModelArchitectureValidationRequest)
    - Returns strongly-typed result (ModelArchitectureValidationResult)
    - Manages validation rules internally
    - Stateless operation (rules injected at construction)

Thread Safety:
    The handler is thread-safe when used with immutable rules. Each handle()
    call is independent and does not modify shared state.

Related:
    - OMN-1138: Architecture Validator for omnibase_infra
    - OMN-1726: Refactor to declarative pattern with contract-driven handler

Example:
    >>> from omnibase_infra.nodes.node_architecture_validator.handlers import (
    ...     HandlerArchitectureValidation,
    ... )
    >>> from omnibase_infra.nodes.node_architecture_validator.models import (
    ...     ModelArchitectureValidationRequest,
    ... )
    >>>
    >>> # Create handler with rules
    >>> handler = HandlerArchitectureValidation(
    ...     rules=(no_handler_publishing_rule, no_workflow_in_reducer_rule),
    ... )
    >>>
    >>> # Validate nodes and handlers
    >>> request = ModelArchitectureValidationRequest(
    ...     nodes=(my_orchestrator, my_reducer),
    ...     handlers=(my_handler,),
    ... )
    >>> result = handler.handle(request)
    >>> if result.valid:
    ...     print("All architecture rules passed")

.. versionadded:: 0.9.0
    Created as part of OMN-1726 declarative refactoring.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import shlex
from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.models.contracts.subcontracts.model_validator_rule import (
    ModelValidatorRule,
)
from omnibase_core.utils.util_safe_yaml_loader import load_yaml_mapping_no_duplicates
from omnibase_core.validation.checker_stack_shape import compose_services
from omnibase_core.validation.validator_stack_shape import ValidatorStackShape
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
    EnumValidationSeverity,
)
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.nodes.node_architecture_validator.constants import (
    SUPPORTED_RULE_IDS,
)
from omnibase_infra.nodes.node_architecture_validator.models import (
    ModelArchitectureValidationRequest,
    ModelArchitectureValidationResult,
    ModelArchitectureViolation,
    ModelRuleCheckResult,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_architecture_validator.protocols import (
        ProtocolArchitectureRule,
    )

HANDLER_ID_ARCHITECTURE_VALIDATION: str = "handler-architecture-validation"

__all__ = ["HandlerArchitectureValidation", "HANDLER_ID_ARCHITECTURE_VALIDATION"]


class HandlerArchitectureValidation:
    """Handler for architecture validation operations.

    Validates nodes and handlers against architecture rules. This handler
    encapsulates all validation logic that was previously in the compute node.

    Can be called:
    - At startup (direct call, pre-runtime validation)
    - During runtime (via orchestrator for dynamic validation)
    - From CI/CD (standalone validation in test/build pipelines)

    Attributes:
        _rules: Tuple of architecture rules to enforce during validation.

    Thread Safety:
        This handler is thread-safe when:
        - Rules are stateless (recommended)
        - Request objects are not shared across threads
        - Each handle() call operates independently

    Example:
        >>> # Create handler with rules
        >>> handler = HandlerArchitectureValidation(rules=rules)
        >>> result = handler.handle(request)
        >>> if not result:
        ...     from omnibase_infra.enums import EnumInfraTransportType
        ...     from omnibase_infra.errors import (
        ...         ModelInfraErrorContext,
        ...         ProtocolConfigurationError,
        ...     )
        ...     context = ModelInfraErrorContext.with_correlation(
        ...         transport_type=EnumInfraTransportType.RUNTIME,
        ...         operation="architecture_validation",
        ...     )
        ...     raise ProtocolConfigurationError(
        ...         f"Validation failed: {result.violation_count} violations",
        ...         context=context,
        ...         code="ARCHITECTURE_VALIDATION_FAILED",
        ...         violations=[v.to_structured_dict() for v in result.violations],
        ...     )

    .. versionadded:: 0.9.0
        Created as part of OMN-1726 declarative refactoring.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role: COMPUTE_HANDLER (pure validation logic)."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification: COMPUTE (pure, deterministic)."""
        return EnumHandlerTypeCategory.COMPUTE

    def __init__(
        self,
        rules: tuple[ProtocolArchitectureRule, ...] = (),
    ) -> None:
        """Initialize handler with architecture rules.

        Args:
            rules: Architecture rules to enforce. These should implement
                ProtocolArchitectureRule. Rules from OMN-1099 validators can be
                passed directly.

        Raises:
            ProtocolConfigurationError: If any rule has a rule_id not in
                SUPPORTED_RULE_IDS. The error message includes the unsupported
                rule_id and the list of supported rules for debugging.

        Example:
            >>> from my_rules import NoHandlerPublishingRule, NoWorkflowInReducerRule
            >>>
            >>> handler = HandlerArchitectureValidation(
            ...     rules=(NoHandlerPublishingRule(), NoWorkflowInReducerRule()),
            ... )

        .. versionadded:: 0.9.0
        """
        _validate_rules_against_contract(rules)
        self._rules = rules
        self._stack_rules = {r.rule_id: r for r in ValidatorStackShape().contract.rules}

    @property
    def handler_id(self) -> str:
        """Return unique identifier for this handler.

        Returns:
            str: Handler identifier.

        .. versionadded:: 0.9.0
        """
        return HANDLER_ID_ARCHITECTURE_VALIDATION

    @property
    def supported_operations(self) -> frozenset[str]:
        """Return supported operations for this handler.

        Returns:
            frozenset[str]: Set of supported operation names.

        .. versionadded:: 0.9.0
        """
        return frozenset({"architecture.validate"})

    def validate_architecture(
        self,
        request: ModelArchitectureValidationRequest,
    ) -> ModelArchitectureValidationResult:
        """Validate architecture compliance.

        Applies all registered rules to the nodes and handlers in the request.
        Returns a result containing any violations found and summary statistics.

        This is a pure transformation with no side effects:
        - Same input always produces same output
        - Does not modify request, rules, or external state
        - Safe to call concurrently from multiple threads

        Args:
            request: Validation request containing:
                - nodes: Nodes to validate
                - handlers: Handlers to validate
                - rule_ids: Optional filter for specific rules
                - fail_fast: Whether to stop on first violation
                - correlation_id: For distributed tracing

        Returns:
            ModelArchitectureValidationResult with:
            - violations: All violations found (empty if validation passed)
            - rules_checked: IDs of rules that were evaluated
            - nodes_checked: Count of nodes validated
            - handlers_checked: Count of handlers validated
            - correlation_id: Propagated from request

        Example:
            >>> # Check all rules
            >>> result = handler.handle(ModelArchitectureValidationRequest(
            ...     nodes=all_nodes,
            ...     handlers=all_handlers,
            ... ))
            >>> print(f"Checked {result.nodes_checked} nodes, "
            ...       f"{result.handlers_checked} handlers")
            >>> print(f"Found {result.violation_count} violations")

            >>> # Check specific rules only
            >>> result = handler.handle(ModelArchitectureValidationRequest(
            ...     nodes=all_nodes,
            ...     rule_ids=("NO_HANDLER_PUBLISHING", "PURE_REDUCERS"),
            ...     fail_fast=True,
            ... ))

        .. versionadded:: 0.9.0
        """
        violations: list[ModelArchitectureViolation] = []
        rules_to_check = self._get_rules_to_check(request.rule_ids)

        # Validate nodes
        for node in request.nodes:
            for rule in rules_to_check:
                result = rule.check(node)
                if not result.passed:
                    violation = self._create_violation(rule, node, result)
                    violations.append(violation)
                    if request.fail_fast:
                        return self._build_result(violations, rules_to_check, request)

        # Validate handlers
        for handler in request.handlers:
            for rule in rules_to_check:
                result = rule.check(handler)
                if not result.passed:
                    violation = self._create_violation(rule, handler, result)
                    violations.append(violation)
                    if request.fail_fast:
                        return self._build_result(violations, rules_to_check, request)

        return self._build_result(violations, rules_to_check, request)

    def handle(
        self, request: ModelArchitectureValidationRequest
    ) -> ModelArchitectureValidationResult:
        """Canonical compute entry; stack commands extend T1's existing parser and gate."""
        architecture = self.validate_architecture(request)
        if request.stack_shape is None:
            return architecture
        snapshot = request.stack_shape
        violations = list(architecture.violations)

        def refuse(
            rule: str, target: str, message: str, path: str | None = None
        ) -> None:
            violations.append(
                ModelArchitectureViolation(
                    rule_id=rule,
                    rule_name=rule,
                    severity=EnumValidationSeverity.ERROR,
                    target_type="compose_service" if path else "stack_policy",
                    target_name=target,
                    message=message,
                    location=path,
                )
            )

        try:
            policy = load_yaml_mapping_no_duplicates(
                snapshot.policy_yaml, source="stack-shape data"
            )
            allowed = _command_rows(policy, "allowed_commands")
            baseline = _command_rows(policy, "command_baseline")
            actual = _stack_commands(snapshot.compose_files, self._stack_rules)
            digests = Counter(
                {
                    (service, field, _command_digest(argv)): count
                    for (service, field, argv), count in actual.items()
                }
            )
            budgets = Counter(
                {_command_key(row): _command_files(row) for row in baseline}
            )
            owners = _baseline_services(policy)
            for row in baseline:
                key = _command_key(row)
                service = str(row["service"])
                if not any(fnmatch.fnmatchcase(service, owner) for owner in owners):
                    refuse(
                        "command_baseline",
                        service,
                        f"command baseline for {service} has no T1 service baseline",
                    )
                if digests[key] != budgets[key]:
                    refuse(
                        "command_baseline",
                        service,
                        f"command baseline for {service} counts {budgets[key]} files but {digests[key]} declare it; the baseline only shrinks",
                    )
            for (service, field, argv), count in actual.items():
                admitted = any(
                    _command_allowed(row, service, field, argv) for row in allowed
                )
                if (
                    not admitted
                    and budgets[(service, field, _command_digest(argv))] != count
                ):
                    for path, text in snapshot.compose_files.items():
                        body = compose_services(text, self._stack_rules).get(
                            service, {}
                        )
                        if _command_argv(body.get(field)) == argv:
                            refuse(
                                "command_not_allowed",
                                service,
                                f"service {service} in {path} has {field} {shlex.join(argv)!r} outside allowed_commands",
                                path,
                            )
            if snapshot.base_policy_yaml is not None:
                base = load_yaml_mapping_no_duplicates(
                    snapshot.base_policy_yaml, source="base stack-shape data"
                )
                # Bootstrap from what T1 already baselines on the base, never from a plant
                # in this branch. After bootstrap the declared budgets are authoritative.
                if "command_baseline" in base:
                    base_budgets = {
                        _command_key(row): _command_files(row)
                        for row in _command_rows(base, "command_baseline")
                    }
                else:
                    owners = _baseline_services(base)
                    base_budgets = {
                        (key[0], key[1], _command_digest(key[2])): count
                        for key, count in _stack_commands(
                            snapshot.base_compose_files, self._stack_rules
                        ).items()
                        if any(fnmatch.fnmatchcase(key[0], owner) for owner in owners)
                    }
                commands = {
                    (service, field, _command_digest(argv)): shlex.join(argv)
                    for service, field, argv in actual
                }
                for key, count in budgets.items():
                    if count > base_budgets.get(key, 0):
                        refuse(
                            "command_baseline",
                            key[0],
                            f"command baseline for {key[0]} {commands.get(key, key[2])!r} is not on the base revision at this count; the baseline only shrinks",
                        )
            inventory = policy.get("console_scripts")
            if not isinstance(inventory, dict):
                raise ValueError(
                    "console_scripts must account for each package's declared scripts"
                )
            for package, scripts in snapshot.package_scripts.items():
                recorded = inventory.get(package)
                if not isinstance(recorded, dict):
                    refuse(
                        "console_script_unaccounted",
                        package,
                        f"package {package} has no console_scripts inventory",
                    )
                    continue
                for name, target in scripts.items():
                    entry = recorded.get(name)
                    if (
                        not isinstance(entry, dict)
                        or entry.get("target") != target
                        or entry.get("status") not in {"run_by_lane", "developer_only"}
                        or not entry.get("reason")
                    ):
                        refuse(
                            "console_script_unaccounted",
                            name,
                            f"console script {package}:{name} = {target} is neither run by a lane nor marked developer-only",
                        )
                for name in recorded.keys() - scripts.keys():
                    refuse(
                        "console_script_stale",
                        str(name),
                        f"console script {package}:{name} is no longer declared; remove its inventory entry",
                    )
        except (ModelOnexError, ValueError, TypeError, KeyError) as exc:
            refuse(
                "stack_command_policy_invalid",
                "allowed_commands",
                f"invalid allowed_commands policy: {exc}",
            )
        return architecture.model_copy(
            update={
                "violations": tuple(violations),
                "rules_checked": architecture.rules_checked + ("stack_shape_commands",),
            }
        )

    def _get_rules_to_check(
        self,
        rule_ids: tuple[str, ...] | None,
    ) -> tuple[ProtocolArchitectureRule, ...]:
        """Get rules to check based on request filter.

        Args:
            rule_ids: Optional tuple of rule IDs to filter by.
                If None, all registered rules are returned.

        Returns:
            Tuple of rules to check. If rule_ids is provided, only
            rules with matching IDs are included.

        .. versionadded:: 0.9.0
        """
        if rule_ids is None:
            return self._rules
        return tuple(r for r in self._rules if r.rule_id in rule_ids)

    def _create_violation(
        self,
        rule: ProtocolArchitectureRule,
        target: object,
        result: ModelRuleCheckResult,
    ) -> ModelArchitectureViolation:
        """Create violation from rule check result.

        Args:
            rule: The rule that was violated.
            target: The node or handler that violated the rule.
            result: The check result with violation details.

        Returns:
            ModelArchitectureViolation with full context for debugging
            and remediation.

        Note:
            Uses getattr for target_name to handle both class types
            (with __name__) and instances (fallback to str()).

        .. versionadded:: 0.9.0
        """
        return ModelArchitectureViolation(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            target_type=type(target).__name__,
            target_name=getattr(target, "__name__", str(target)),
            message=result.message or rule.description,
            details=result.details,
        )

    def _build_result(
        self,
        violations: list[ModelArchitectureViolation],
        rules_checked: tuple[ProtocolArchitectureRule, ...],
        request: ModelArchitectureValidationRequest,
    ) -> ModelArchitectureValidationResult:
        """Build final validation result.

        Args:
            violations: All violations found during validation.
            rules_checked: Rules that were evaluated.
            request: Original validation request (for counts and correlation_id).

        Returns:
            ModelArchitectureValidationResult with complete validation summary.

        .. versionadded:: 0.9.0
        """
        return ModelArchitectureValidationResult(
            violations=tuple(violations),
            rules_checked=tuple(r.rule_id for r in rules_checked),
            nodes_checked=len(request.nodes),
            handlers_checked=len(request.handlers),
            correlation_id=request.correlation_id,
        )


def _command_argv(value: object) -> tuple[str, ...]:
    """Compose string commands are argv, not implicit shell commands."""
    if value is None or value == "" or value == []:
        return ()
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return tuple(value)
    raise ValueError(
        f"command/entrypoint must be a string or string list, got {value!r}"
    )


def _command_rows(policy: Mapping[object, object], key: str) -> list[dict[str, object]]:
    rows = policy.get(key)
    if not isinstance(rows, list) or (key == "allowed_commands" and not rows):
        raise ValueError(
            f"{key} must be a {'non-empty ' if key == 'allowed_commands' else ''}list"
        )
    identities: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"every {key} entry must be a mapping")
        if "argv_sha256" in row:
            if "argv" in row or not re.fullmatch(
                r"[0-9a-f]{64}", str(row["argv_sha256"])
            ):
                raise ValueError(
                    f"{key}.argv_sha256 must be a SHA256, without an argv list"
                )
        elif not isinstance(row.get("argv"), list) or not _command_argv(row["argv"]):
            raise ValueError(
                f"every {key} entry needs a non-empty argv list or argv_sha256"
            )
        if row.get("field") not in {"command", "entrypoint"}:
            raise ValueError(f"every {key} entry needs field: command or entrypoint")
        if key == "allowed_commands":
            if not row.get("reason") or row.get("match") not in {"exact", "executable"}:
                raise ValueError(
                    "allowed_commands entries need a reason and an exact/executable match"
                )
            _string_list(row.get("services"), "allowed_commands.services")
            if row["match"] == "executable" and (
                "argv_sha256" in row or len(row["argv"]) != 1
            ):
                raise ValueError("executable matches name exactly one executable")
        else:
            if not isinstance(row.get("service"), str):
                raise ValueError("command_baseline entries need a service")
            _command_files(row)
        identity = json.dumps(
            [
                row.get("service"),
                row.get("services"),
                row["field"],
                row.get("match"),
                _row_digest(row),
            ]
        )
        if identity in identities:
            raise ValueError(f"duplicate {key} entry")
        identities.add(identity)
    return rows


def _command_files(row: Mapping[str, object]) -> int:
    """Narrow a baseline budget without coercing strings or booleans to counts."""
    value = row.get("files")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("command_baseline entries need a positive files count")
    return value


def _string_list(value: object, name: str) -> tuple[str, ...]:
    """Validate untyped YAML sequences before iterating over their strings."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a string list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{name} must be a string list")
        result.append(item)
    return tuple(result)


def _baseline_services(policy: Mapping[object, object]) -> tuple[str, ...]:
    """Read the T1 ownership patterns used to constrain command baselines."""
    rows = policy.get("baseline", [])
    if not isinstance(rows, list):
        raise ValueError("baseline must be a list")
    services: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("baseline entries need a service string")
        service = row.get("service")
        if not isinstance(service, str):
            raise ValueError("baseline entries need a service string")
        services.append(service)
    return tuple(services)


def _command_digest(argv: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(argv, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


def _row_digest(row: Mapping[str, object]) -> str:
    return (
        str(row["argv_sha256"])
        if "argv_sha256" in row
        else _command_digest(_command_argv(row["argv"]))
    )


def _command_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return str(row["service"]), str(row["field"]), _row_digest(row)


def _command_allowed(
    row: Mapping[str, object], service: str, field: str, argv: tuple[str, ...]
) -> bool:
    services = _string_list(row["services"], "allowed_commands.services")
    identities = {service, "sha256:" + hashlib.sha256(service.encode()).hexdigest()}
    if row["field"] != field or (services and not identities.intersection(services)):
        return False
    if row["match"] == "executable":
        return argv[:1] == _command_argv(row["argv"])
    return _command_digest(argv) == _row_digest(row)


def _validate_rules_against_contract(
    rules: tuple[ProtocolArchitectureRule, ...],
) -> None:
    """Validate that all provided rules are in the contract's supported_rules.

    This validation ensures that only rules defined in the contract's
    supported_rules list can be used with this handler. Using unsupported
    rules would indicate a misconfiguration or version mismatch between
    the handler and the rules being passed.

    Args:
        rules: Tuple of architecture rules to validate against the contract.

    Raises:
        ProtocolConfigurationError: If any rule has a rule_id not in
            SUPPORTED_RULE_IDS. The error message includes the unsupported
            rule_id and the list of supported rules for debugging.
            Includes ModelInfraErrorContext with RUNTIME transport type.

    .. versionadded:: 0.9.0
    """
    for rule in rules:
        if rule.rule_id not in SUPPORTED_RULE_IDS:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="validate_rules_against_contract",
            )
            raise ProtocolConfigurationError(
                f"Rule '{rule.rule_id}' is not in contract supported_rules. "
                f"Supported rules: {sorted(SUPPORTED_RULE_IDS)}",
                context=context,
                rule_id=rule.rule_id,
                supported_rules=sorted(SUPPORTED_RULE_IDS),
            )


def _stack_commands(
    files: Mapping[str, str], rules: Mapping[str, ModelValidatorRule]
) -> Counter[tuple[str, str, tuple[str, ...]]]:
    """Count declarations with T1's compose parser, including merge directives."""
    result: Counter[tuple[str, str, tuple[str, ...]]] = Counter()
    for text in files.values():
        for service, body in compose_services(text, rules).items():
            for field in ("command", "entrypoint"):
                argv = _command_argv(body.get(field))
                if argv:
                    result[(service, field, argv)] += 1
    return result
