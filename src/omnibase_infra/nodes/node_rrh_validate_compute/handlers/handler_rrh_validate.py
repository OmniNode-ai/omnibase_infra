# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pure COMPUTE handler implementing the 13 RRH validation rules.

Rule Catalog:
    RRH-1001  Repo clean working tree
    RRH-1002  Repo on expected branch
    RRH-1101  Environment target valid
    RRH-1102  Environment kafka_broker configured
    RRH-1201  Kafka broker reachable (conditional)
    RRH-1301  Kubernetes context valid (conditional)
    RRH-1401  Toolchain pre-commit present
    RRH-1402  Toolchain ruff present
    RRH-1403  Toolchain pytest present (conditional)
    RRH-1404  Toolchain mypy present
    RRH-1501  Cross-check branch matches ticket ID
    RRH-1601  Cross-check no disallowed contract fields
    RRH-1701  Repo-boundary no cross-repo imports

Profile Precedence:
    PROFILE sets baseline -> CONTRACT can only TIGHTEN (never loosen)

Contract Tightening:
    - ``evidence_requirements: ["tests"]`` -> activates RRH-1403
    - ``interfaces_touched: ["topics"]`` -> activates RRH-1201
    - ``deployment_targets: ["k8s"]`` -> activates RRH-1301
    - ``is_seam_ticket: true`` -> switches to seam-ticket profile

CRITICAL INVARIANTS:
    - Pure computation: no I/O, no side effects, no event bus access
    - Deterministic: same input always produces same output
    - Contract can only tighten, never loosen profile rules
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_infra.diagnostics.enum_verdict import EnumVerdict
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.rrh.model_rrh_result import ModelRRHResult
from omnibase_infra.models.rrh.model_rrh_rule_severity import ModelRRHRuleSeverity
from omnibase_infra.nodes.architecture_validator.models.model_rule_check_result import (
    ModelRuleCheckResult,
)
from omnibase_infra.nodes.node_rrh_validate_compute.profiles import get_profile

if TYPE_CHECKING:
    from omnibase_infra.models.rrh.model_rrh_environment_data import (
        ModelRRHEnvironmentData,
    )
    from omnibase_infra.models.rrh.model_rrh_profile import ModelRRHProfile
    from omnibase_infra.nodes.node_rrh_validate_compute.models.model_rrh_contract_governance import (
        ModelRRHContractGovernance,
    )
    from omnibase_infra.nodes.node_rrh_validate_compute.models.model_rrh_validate_request import (
        ModelRRHValidateRequest,
    )

logger = logging.getLogger(__name__)

_VALID_ENVIRONMENTS = frozenset({"dev", "staging", "production", "ci", "test"})

# Heuristic to detect regex patterns vulnerable to catastrophic backtracking
# (ReDoS).  Catches quantified groups containing quantifiers, e.g. (a+)+, (a*)*.
_NESTED_QUANTIFIER_RE = re.compile(r"\([^)]*[+*][^)]*\)\s*[+*?{]")
_MAX_BRANCH_PATTERN_LEN = 200
_MAX_DISPLAY_PATTERN_LEN = 80


def _truncate_pattern(pattern: str, max_len: int = _MAX_DISPLAY_PATTERN_LEN) -> str:
    """Truncate a regex pattern for safe display in error messages."""
    if len(pattern) <= max_len:
        return pattern
    return pattern[:max_len] + "..."


# All 13 rule IDs in catalog order.
ALL_RULE_IDS: tuple[str, ...] = (
    "RRH-1001",
    "RRH-1002",
    "RRH-1101",
    "RRH-1102",
    "RRH-1201",
    "RRH-1301",
    "RRH-1401",
    "RRH-1402",
    "RRH-1403",
    "RRH-1404",
    "RRH-1501",
    "RRH-1601",
    "RRH-1701",
)


class HandlerRRHValidate:
    """Pure COMPUTE handler for RRH validation.

    Evaluates 13 rules against collected environment data using
    profile-driven severity and contract tightening enforcement.

    Attributes:
        handler_type: ``COMPUTE_HANDLER``
        handler_category: ``COMPUTE``
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def handle(self, request: ModelRRHValidateRequest) -> ModelRRHResult:
        """Evaluate all RRH rules against the environment data.

        Note:
            Intentionally synchronous.  This is a pure COMPUTE handler with
            no I/O — all inputs are pre-collected values.  The handler is
            called directly (not via the envelope-based ``ProtocolHandler``
            dispatch), so ``async`` is unnecessary and would add overhead.

        Args:
            request: Validation request with environment data, profile,
                and governance.

        Returns:
            ``ModelRRHResult`` with per-rule checks and aggregate verdict.
        """
        governance = request.governance

        # Seam ticket overrides profile to maximum coverage.
        profile_name = (
            "seam-ticket" if governance.is_seam_ticket else request.profile_name
        )
        try:
            profile = self._load_profile(profile_name)
        except KeyError:
            # Unknown profile name -- return immediate FAIL with all rules
            # marked as failed so the caller gets actionable feedback.
            return ModelRRHResult(
                checks=tuple(
                    ModelRuleCheckResult(
                        passed=False,
                        rule_id=rule_id,
                        message=f"Unknown profile '{profile_name}'.",
                    )
                    for rule_id in ALL_RULE_IDS
                ),
                verdict=EnumVerdict.FAIL,
                profile_name=profile_name,
                ticket_id=governance.ticket_id,
                repo_name=request.repo_name,
                correlation_id=request.correlation_id,
                evaluated_at=datetime.now(UTC),
            )

        # Apply contract tightening — can only enable rules, never disable.
        effective_rules = self._apply_tightening(profile, governance)

        # Run all 13 rules.
        checks: list[ModelRuleCheckResult] = []
        for rule_id in ALL_RULE_IDS:
            rule_cfg = effective_rules.get(rule_id)
            if rule_cfg is None or not rule_cfg.enabled:
                checks.append(
                    ModelRuleCheckResult(
                        passed=True,
                        skipped=True,
                        rule_id=rule_id,
                        reason=f"Disabled in profile '{profile_name}'.",
                    )
                )
                continue
            result = self._evaluate_rule(rule_id, request.environment_data, governance)
            checks.append(result)

        # Derive aggregate verdict from worst applicable check.
        verdict = self._derive_verdict(checks, effective_rules)

        return ModelRRHResult(
            checks=tuple(checks),
            verdict=verdict,
            profile_name=profile_name,
            ticket_id=governance.ticket_id,
            repo_name=request.repo_name,
            correlation_id=request.correlation_id,
            evaluated_at=datetime.now(UTC),
        )

    # ------------------------------------------------------------------
    # Profile loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_profile(name: str) -> ModelRRHProfile:
        """Retrieve a built-in RRH profile by name.

        Args:
            name: Profile name (``default``, ``ticket-pipeline``,
                ``ci-repair``, ``seam-ticket``).

        Returns:
            The matching ``ModelRRHProfile``.

        Raises:
            KeyError: If the profile name is not recognized.
        """
        return get_profile(name)

    # ------------------------------------------------------------------
    # Contract tightening
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_tightening(
        profile: ModelRRHProfile,
        governance: ModelRRHContractGovernance,
    ) -> dict[str, ModelRRHRuleSeverity]:
        """Apply contract governance tightening to profile rules.

        Tightening rules:
        - ``evidence_requirements: ["tests"]`` -> enable RRH-1403
        - ``interfaces_touched: ["topics"]`` -> enable RRH-1201
        - ``deployment_targets: ["k8s"]`` -> enable RRH-1301

        CRITICAL: Contract can only ENABLE rules or RAISE severity.
        It can NEVER disable a rule that the profile enables, nor
        lower severity from FAIL to WARN.
        """
        rules: dict[str, ModelRRHRuleSeverity] = {r.rule_id: r for r in profile.rules}

        # Tighten: evidence_requirements includes "tests" -> RRH-1403
        if "tests" in governance.evidence_requirements:
            existing = rules.get("RRH-1403")
            if existing is None or not existing.enabled:
                rules["RRH-1403"] = ModelRRHRuleSeverity(
                    rule_id="RRH-1403", enabled=True, severity=EnumVerdict.FAIL
                )

        # Tighten: interfaces_touched includes "topics" -> RRH-1201
        if "topics" in governance.interfaces_touched:
            existing = rules.get("RRH-1201")
            if existing is None or not existing.enabled:
                rules["RRH-1201"] = ModelRRHRuleSeverity(
                    rule_id="RRH-1201", enabled=True, severity=EnumVerdict.FAIL
                )

        # Tighten: deployment_targets includes "k8s" -> RRH-1301
        if "k8s" in governance.deployment_targets:
            existing = rules.get("RRH-1301")
            if existing is None or not existing.enabled:
                rules["RRH-1301"] = ModelRRHRuleSeverity(
                    rule_id="RRH-1301", enabled=True, severity=EnumVerdict.FAIL
                )

        return rules

    # ------------------------------------------------------------------
    # Rule evaluation dispatcher
    # ------------------------------------------------------------------

    @cached_property
    def _rule_dispatcher(self) -> dict[str, Callable[..., ModelRuleCheckResult]]:
        """Build rule dispatcher once and cache on the instance."""
        return {
            "RRH-1001": self._check_1001_clean_tree,
            "RRH-1002": self._check_1002_expected_branch,
            "RRH-1101": self._check_1101_env_target_valid,
            "RRH-1102": self._check_1102_kafka_broker_configured,
            "RRH-1201": self._check_1201_kafka_reachable,
            "RRH-1301": self._check_1301_k8s_context_valid,
            "RRH-1401": self._check_1401_precommit_present,
            "RRH-1402": self._check_1402_ruff_present,
            "RRH-1403": self._check_1403_pytest_present,
            "RRH-1404": self._check_1404_mypy_present,
            "RRH-1501": self._check_1501_branch_matches_ticket,
            "RRH-1601": self._check_1601_no_disallowed_fields,
            "RRH-1701": self._check_1701_repo_boundary,
        }

    def _evaluate_rule(
        self,
        rule_id: str,
        env: ModelRRHEnvironmentData,
        gov: ModelRRHContractGovernance,
    ) -> ModelRuleCheckResult:
        """Dispatch to the appropriate rule checker."""
        checker = self._rule_dispatcher.get(rule_id)
        if checker is None:
            return ModelRuleCheckResult(
                passed=False,
                rule_id=rule_id,
                message=f"Unknown rule: {rule_id}",
            )
        return checker(env, gov)

    # ------------------------------------------------------------------
    # Individual rule implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _check_1001_clean_tree(
        env: ModelRRHEnvironmentData, _gov: ModelRRHContractGovernance
    ) -> ModelRuleCheckResult:
        """RRH-1001: Working tree must be clean (no uncommitted changes)."""
        if env.repo_state.is_dirty:
            return ModelRuleCheckResult(
                passed=False,
                rule_id="RRH-1001",
                message="Working tree has uncommitted changes.",
            )
        return ModelRuleCheckResult(passed=True, rule_id="RRH-1001")

    @staticmethod
    def _check_1002_expected_branch(
        env: ModelRRHEnvironmentData, gov: ModelRRHContractGovernance
    ) -> ModelRuleCheckResult:
        """RRH-1002: Branch matches expected pattern from governance."""
        if not gov.expected_branch_pattern:
            return ModelRuleCheckResult(
                passed=True,
                skipped=True,
                rule_id="RRH-1002",
                reason="No expected_branch_pattern in governance.",
            )
        pattern = gov.expected_branch_pattern
        # Guard against ReDoS: reject overly long or structurally unsafe patterns.
        if len(pattern) > _MAX_BRANCH_PATTERN_LEN or _NESTED_QUANTIFIER_RE.search(
            pattern
        ):
            return ModelRuleCheckResult(
                passed=False,
                rule_id="RRH-1002",
                message=f"Unsafe branch pattern (possible ReDoS): {_truncate_pattern(pattern)!r}",
            )
        branch = env.repo_state.branch
        try:
            if re.fullmatch(pattern, branch):
                return ModelRuleCheckResult(passed=True, rule_id="RRH-1002")
        except re.error:
            return ModelRuleCheckResult(
                passed=False,
                rule_id="RRH-1002",
                message=f"Invalid branch pattern: {_truncate_pattern(gov.expected_branch_pattern)}",
            )
        return ModelRuleCheckResult(
            passed=False,
            rule_id="RRH-1002",
            message=f"Branch '{branch}' does not match pattern '{_truncate_pattern(gov.expected_branch_pattern)}'.",
        )

    @staticmethod
    def _check_1101_env_target_valid(
        env: ModelRRHEnvironmentData, _gov: ModelRRHContractGovernance
    ) -> ModelRuleCheckResult:
        """RRH-1101: Environment target is a recognized value."""
        target = env.runtime_target.environment
        if target in _VALID_ENVIRONMENTS:
            return ModelRuleCheckResult(passed=True, rule_id="RRH-1101")
        return ModelRuleCheckResult(
            passed=False,
            rule_id="RRH-1101",
            message=f"Unknown environment '{target}'. Valid: {sorted(_VALID_ENVIRONMENTS)}.",
        )

    @staticmethod
    def _check_1102_kafka_broker_configured(
        env: ModelRRHEnvironmentData, _gov: ModelRRHContractGovernance
    ) -> ModelRuleCheckResult:
        """RRH-1102: Kafka broker address is configured (non-empty)."""
        if env.runtime_target.kafka_broker:
            return ModelRuleCheckResult(passed=True, rule_id="RRH-1102")
        return ModelRuleCheckResult(
            passed=False,
            rule_id="RRH-1102",
            message="kafka_broker is not configured.",
        )

    @staticmethod
    def _check_1201_kafka_reachable(
        env: ModelRRHEnvironmentData, _gov: ModelRRHContractGovernance
    ) -> ModelRuleCheckResult:
        """RRH-1201: Kafka broker reachable (pure check — only validates config).

        Note: True reachability requires I/O; this COMPUTE rule validates
        that the broker address looks well-formed (host:port).
        """
        broker = env.runtime_target.kafka_broker
        if not broker:
            return ModelRuleCheckResult(
                passed=False,
                rule_id="RRH-1201",
                message="Kafka broker not configured but interfaces_touched includes 'topics'.",
            )
        # Validate host:port format(s). Supports comma-separated broker lists
        # and underscores in hostnames. Basic format check only (not full
        # URI validation).
        parts = broker.split(",")
        valid = True
        for p in parts:
            m = re.fullmatch(r"[^\s,]+:(\d{1,5})", p.strip())
            if not m or not (1 <= int(m.group(1)) <= 65535):
                valid = False
                break
        if valid:
            return ModelRuleCheckResult(passed=True, rule_id="RRH-1201")
        return ModelRuleCheckResult(
            passed=False,
            rule_id="RRH-1201",
            message=f"Kafka broker '{broker}' is not in host:port format.",
        )

    @staticmethod
    def _check_1301_k8s_context_valid(
        env: ModelRRHEnvironmentData, _gov: ModelRRHContractGovernance
    ) -> ModelRuleCheckResult:
        """RRH-1301: Kubernetes context is non-empty."""
        if env.runtime_target.kubernetes_context:
            return ModelRuleCheckResult(passed=True, rule_id="RRH-1301")
        return ModelRuleCheckResult(
            passed=False,
            rule_id="RRH-1301",
            message="kubernetes_context is empty but deployment_targets includes 'k8s'.",
        )

    @staticmethod
    def _check_1401_precommit_present(
        env: ModelRRHEnvironmentData, _gov: ModelRRHContractGovernance
    ) -> ModelRuleCheckResult:
        """RRH-1401: pre-commit is installed."""
        if env.toolchain.pre_commit:
            return ModelRuleCheckResult(passed=True, rule_id="RRH-1401")
        return ModelRuleCheckResult(
            passed=False,
            rule_id="RRH-1401",
            message="pre-commit is not installed.",
        )

    @staticmethod
    def _check_1402_ruff_present(
        env: ModelRRHEnvironmentData, _gov: ModelRRHContractGovernance
    ) -> ModelRuleCheckResult:
        """RRH-1402: ruff is installed."""
        if env.toolchain.ruff:
            return ModelRuleCheckResult(passed=True, rule_id="RRH-1402")
        return ModelRuleCheckResult(
            passed=False,
            rule_id="RRH-1402",
            message="ruff is not installed.",
        )

    @staticmethod
    def _check_1403_pytest_present(
        env: ModelRRHEnvironmentData, _gov: ModelRRHContractGovernance
    ) -> ModelRuleCheckResult:
        """RRH-1403: pytest is installed (conditional on evidence_requirements)."""
        if env.toolchain.pytest:
            return ModelRuleCheckResult(passed=True, rule_id="RRH-1403")
        return ModelRuleCheckResult(
            passed=False,
            rule_id="RRH-1403",
            message="pytest is not installed but evidence_requirements includes 'tests'.",
        )

    @staticmethod
    def _check_1404_mypy_present(
        env: ModelRRHEnvironmentData, _gov: ModelRRHContractGovernance
    ) -> ModelRuleCheckResult:
        """RRH-1404: mypy is installed."""
        if env.toolchain.mypy:
            return ModelRuleCheckResult(passed=True, rule_id="RRH-1404")
        return ModelRuleCheckResult(
            passed=False,
            rule_id="RRH-1404",
            message="mypy is not installed.",
        )

    @staticmethod
    def _check_1501_branch_matches_ticket(
        env: ModelRRHEnvironmentData, gov: ModelRRHContractGovernance
    ) -> ModelRuleCheckResult:
        """RRH-1501: Branch name contains the ticket ID."""
        if not gov.ticket_id:
            return ModelRuleCheckResult(
                passed=True,
                skipped=True,
                rule_id="RRH-1501",
                reason="No ticket_id in governance.",
            )
        # Normalize: OMN-2136 -> omn-2136
        ticket_lower = gov.ticket_id.lower().replace(" ", "")
        branch_lower = env.repo_state.branch.lower()
        if ticket_lower in branch_lower:
            return ModelRuleCheckResult(passed=True, rule_id="RRH-1501")
        return ModelRuleCheckResult(
            passed=False,
            rule_id="RRH-1501",
            message=f"Branch '{env.repo_state.branch}' does not contain ticket '{gov.ticket_id}'.",
        )

    @staticmethod
    def _check_1601_no_disallowed_fields(
        _env: ModelRRHEnvironmentData, gov: ModelRRHContractGovernance
    ) -> ModelRuleCheckResult:
        """RRH-1601: No disallowed contract fields present."""
        if not gov.disallowed_fields:
            return ModelRuleCheckResult(passed=True, rule_id="RRH-1601")
        return ModelRuleCheckResult(
            passed=False,
            rule_id="RRH-1601",
            message=f"Disallowed contract fields present: {list(gov.disallowed_fields)}.",
            details={"disallowed_fields": list(gov.disallowed_fields)},
        )

    @staticmethod
    def _check_1701_repo_boundary(
        env: ModelRRHEnvironmentData, _gov: ModelRRHContractGovernance
    ) -> ModelRuleCheckResult:
        """RRH-1701: Repo boundary — no cross-repo import indicators.

        Checks that the repo root path and remote URL are consistent,
        and that the repo name matches expected scope.

        Handles both HTTPS URLs (``https://github.com/org/repo.git``)
        and SSH URLs (``git@github.com:org/repo.git``).  Comparison is
        case-insensitive to accommodate macOS case-insensitive filesystems.
        """
        if not env.repo_state.repo_root:
            return ModelRuleCheckResult(
                passed=False,
                rule_id="RRH-1701",
                message="repo_root is empty — cannot verify repo boundary.",
            )
        # Extract repo name from root path (last path component).
        repo_dir = Path(env.repo_state.repo_root).name
        # If remote URL is set, check that it references the same repo.
        if env.repo_state.remote_url:
            # Handle both HTTPS (path separator /) and SSH (path separator :).
            url = env.repo_state.remote_url.rstrip("/")
            # For SSH URLs like git@github.com:org/repo.git, split on ":"
            # first to isolate the path portion.
            if ":" in url and not url.startswith(("http://", "https://")):
                url = url.rsplit(":", 1)[-1]
            remote_repo = url.rsplit("/", 1)[-1].removesuffix(".git")
            if remote_repo and remote_repo.lower() != repo_dir.lower():
                return ModelRuleCheckResult(
                    passed=False,
                    rule_id="RRH-1701",
                    message=(
                        f"Repo directory '{repo_dir}' does not match "
                        f"remote repo name '{remote_repo}'."
                    ),
                    details={
                        "repo_dir": repo_dir,
                        "remote_repo": remote_repo,
                    },
                )
        return ModelRuleCheckResult(passed=True, rule_id="RRH-1701")

    # ------------------------------------------------------------------
    # Verdict derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_verdict(
        checks: list[ModelRuleCheckResult],
        effective_rules: dict[str, ModelRRHRuleSeverity],
    ) -> EnumVerdict:
        """Derive aggregate verdict from individual check results.

        Verdict precedence: FAIL > WARN > PASS.
        """
        verdict = EnumVerdict.PASS
        for check in checks:
            if check.skipped or check.passed:
                continue
            # Failed check — look up configured severity.
            rule_cfg = effective_rules.get(check.rule_id)
            severity = rule_cfg.severity if rule_cfg else EnumVerdict.FAIL
            if severity == EnumVerdict.FAIL:
                return EnumVerdict.FAIL
            if severity == EnumVerdict.WARN and verdict == EnumVerdict.PASS:
                verdict = EnumVerdict.WARN
        return verdict


__all__: list[str] = ["HandlerRRHValidate"]
