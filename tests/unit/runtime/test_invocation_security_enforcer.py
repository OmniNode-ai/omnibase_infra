# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""RED tests for invocation-time security enforcement.

These tests define expected behavior for InvocationSecurityEnforcer.
They are expected to FAIL until the enforcer is implemented (TDD RED phase).

Ticket: OMN-1098

TDD Requirements Covered:
    - RED: Test domain violation at invocation

Test Categories:
    - TestDomainAccessEnforcement: Outbound domain access control
    - TestSecretScopeAccessEnforcement: Secret scope access control
    - TestClassificationConstraintEnforcement: Data classification level enforcement
    - TestEnforcerIntegration: Integration and statelessness tests

Expected Import Failures (TDD RED):
    - InvocationSecurityEnforcer: Not implemented yet
    - SecurityViolationError: Not implemented yet
    - ModelHandlerSecurityPolicy: Not implemented yet
    - EnumSecurityRuleId: Not implemented yet
"""

from __future__ import annotations

from uuid import uuid4

import pytest

# Import from omnibase_core - this should work
from omnibase_core.enums import EnumDataClassification

from omnibase_infra.enums.enum_security_rule_id import EnumSecurityRuleId

# These imports SHOULD FAIL - models not yet implemented (TDD RED)
# Once models are created, these imports will work
from omnibase_infra.models.security.model_handler_security_policy import (
    ModelHandlerSecurityPolicy,
)

# These imports SHOULD FAIL - enforcer not implemented yet (TDD RED)
from omnibase_infra.runtime.invocation_security_enforcer import (
    InvocationSecurityEnforcer,
    SecurityViolationError,
)


class TestDomainAccessEnforcement:
    """Tests for outbound domain access enforcement at invocation time.

    Validates that handlers can only access domains explicitly declared
    in their security policy. Domain wildcards are supported.

    Security Rule: SECURITY-310 (DOMAIN_ACCESS_DENIED)
    """

    def test_domain_violation_at_invocation(self) -> None:
        """Handler attempting unauthorized domain access should raise error.

        TDD Requirement: RED: Test domain violation at invocation
        Expected Error: SECURITY-310 (DOMAIN_ACCESS_DENIED)

        This is the primary test case for OMN-1098.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=["api.allowed.com", "storage.allowed.com"],
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # ACT & ASSERT
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_domain_access("api.forbidden.com")

        assert exc_info.value.rule_id == EnumSecurityRuleId.DOMAIN_ACCESS_DENIED
        assert "api.forbidden.com" in str(exc_info.value)

    def test_allowed_domain_access_succeeds(self) -> None:
        """Handler accessing allowed domain should succeed.

        Verifies that domains explicitly listed in allowed_domains
        are permitted without raising any error.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=["api.allowed.com", "storage.allowed.com"],
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # ACT & ASSERT - Should not raise
        enforcer.check_domain_access("api.allowed.com")
        enforcer.check_domain_access("storage.allowed.com")

    def test_wildcard_domain_matching(self) -> None:
        """Wildcard domain patterns should be matched correctly.

        Verifies that wildcard patterns like "*.example.com" match
        subdomains correctly (e.g., "api.example.com" matches).
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=["*.example.com", "api.specific.com"],
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # ACT & ASSERT - Should succeed (matches wildcard)
        enforcer.check_domain_access("api.example.com")
        enforcer.check_domain_access("storage.example.com")

        # Should succeed (exact match)
        enforcer.check_domain_access("api.specific.com")

        # Should fail (doesn't match any pattern)
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_domain_access("api.other.com")

        assert exc_info.value.rule_id == EnumSecurityRuleId.DOMAIN_ACCESS_DENIED

    def test_empty_domain_allowlist_blocks_all(self) -> None:
        """Empty domain allowlist should block all outbound access.

        When no domains are declared, all outbound domain access
        should be denied. This is the most restrictive default.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=[],  # Empty - no domains allowed
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # ACT & ASSERT
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_domain_access("any.domain.com")

        assert exc_info.value.rule_id == EnumSecurityRuleId.DOMAIN_ACCESS_DENIED

    def test_subdomain_does_not_match_parent_domain(self) -> None:
        """Subdomain access should not match parent domain.

        If "example.com" is allowed, "sub.example.com" should NOT be
        allowed (must use "*.example.com" for subdomain matching).
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=["example.com"],  # Only root domain
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # Root domain should work
        enforcer.check_domain_access("example.com")

        # Subdomain should NOT match parent
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_domain_access("sub.example.com")

        assert exc_info.value.rule_id == EnumSecurityRuleId.DOMAIN_ACCESS_DENIED

    def test_wildcard_does_not_match_nested_subdomains(self) -> None:
        """Wildcard should only match single-level subdomains.

        The pattern '*.example.com' matches 'api.example.com' but NOT
        'api.staging.example.com' (nested/multi-level subdomain).

        This is intentional behavior to enforce explicit domain declarations
        and prevent overly broad wildcard matching.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=["*.example.com"],
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # Single-level subdomain should succeed
        enforcer.check_domain_access("api.example.com")
        enforcer.check_domain_access("storage.example.com")

        # Nested/multi-level subdomains should fail
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_domain_access("api.staging.example.com")

        assert exc_info.value.rule_id == EnumSecurityRuleId.DOMAIN_ACCESS_DENIED
        assert "api.staging.example.com" in str(exc_info.value)

        # Another nested subdomain example
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_domain_access("a.b.example.com")

        assert exc_info.value.rule_id == EnumSecurityRuleId.DOMAIN_ACCESS_DENIED


class TestSecretScopeAccessEnforcement:
    """Tests for secret scope access enforcement at invocation time.

    Validates that handlers can only access secret scopes explicitly
    declared in their security policy.

    Security Rule: SECURITY-311 (SECRET_SCOPE_ACCESS_DENIED)
    """

    def test_secret_scope_access_denied(self) -> None:
        """Handler accessing undeclared secret scope should raise error.

        Expected Error: SECURITY-311 (SECRET_SCOPE_ACCESS_DENIED)
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"api-keys"}),  # Only api-keys declared
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # ACT & ASSERT
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_secret_scope_access("database-creds")  # Not declared

        assert exc_info.value.rule_id == EnumSecurityRuleId.SECRET_SCOPE_ACCESS_DENIED
        assert "database-creds" in str(exc_info.value)

    def test_declared_secret_scope_access_succeeds(self) -> None:
        """Handler accessing declared secret scope should succeed."""
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"api-keys", "database-creds"}),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # ACT & ASSERT - Should not raise
        enforcer.check_secret_scope_access("api-keys")
        enforcer.check_secret_scope_access("database-creds")

    def test_no_secret_scopes_blocks_all_access(self) -> None:
        """Handler with no declared secret scopes should block all access.

        When no secret scopes are declared (empty frozenset), all
        secret access should be denied.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),  # No secrets declared
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # ACT & ASSERT
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_secret_scope_access("any-secret")

        assert exc_info.value.rule_id == EnumSecurityRuleId.SECRET_SCOPE_ACCESS_DENIED

    def test_secret_scope_matching_is_exact(self) -> None:
        """Secret scope matching should be exact (no wildcards/prefixes).

        Unlike domain matching, secret scopes use exact string matching.
        "api-keys" does not match "api-keys-v2" or "api".
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"api-keys"}),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # Exact match works
        enforcer.check_secret_scope_access("api-keys")

        # Partial matches should fail
        with pytest.raises(SecurityViolationError):
            enforcer.check_secret_scope_access("api-keys-v2")

        with pytest.raises(SecurityViolationError):
            enforcer.check_secret_scope_access("api")


class TestClassificationConstraintEnforcement:
    """Tests for data classification constraint enforcement at invocation time.

    Validates that handlers cannot process data above their declared
    classification level.

    Security Rule: SECURITY-312 (CLASSIFICATION_CONSTRAINT_VIOLATION)
    """

    def test_classification_constraint_violation(self) -> None:
        """Handler processing data above its classification should raise error.

        Expected Error: SECURITY-312 (CLASSIFICATION_CONSTRAINT_VIOLATION)
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,  # Handler level
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # ACT & ASSERT - Try to process CONFIDENTIAL data with INTERNAL handler
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_classification_constraint(
                EnumDataClassification.CONFIDENTIAL
            )

        assert (
            exc_info.value.rule_id
            == EnumSecurityRuleId.CLASSIFICATION_CONSTRAINT_VIOLATION
        )

    def test_classification_within_limit_succeeds(self) -> None:
        """Handler processing data at or below its classification should succeed.

        Classification hierarchy (lowest to highest):
        PUBLIC < INTERNAL < CONFIDENTIAL < RESTRICTED < SECRET < TOP_SECRET
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=[],
            data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # ACT & ASSERT - Should not raise (at or below handler's level)
        enforcer.check_classification_constraint(EnumDataClassification.PUBLIC)
        enforcer.check_classification_constraint(EnumDataClassification.INTERNAL)
        enforcer.check_classification_constraint(EnumDataClassification.CONFIDENTIAL)

    def test_classification_hierarchy_is_enforced(self) -> None:
        """Classification hierarchy should be strictly enforced.

        An INTERNAL handler should NOT be able to process:
        - CONFIDENTIAL
        - RESTRICTED
        - SECRET
        - TOP_SECRET
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # All levels above INTERNAL should fail
        with pytest.raises(SecurityViolationError):
            enforcer.check_classification_constraint(
                EnumDataClassification.CONFIDENTIAL
            )

        with pytest.raises(SecurityViolationError):
            enforcer.check_classification_constraint(EnumDataClassification.RESTRICTED)

        with pytest.raises(SecurityViolationError):
            enforcer.check_classification_constraint(EnumDataClassification.SECRET)

        with pytest.raises(SecurityViolationError):
            enforcer.check_classification_constraint(EnumDataClassification.TOP_SECRET)

    def test_public_handler_can_only_process_public_data(self) -> None:
        """PUBLIC handler should only be able to process PUBLIC data.

        PUBLIC is the lowest classification level.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=[],
            data_classification=EnumDataClassification.PUBLIC,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # PUBLIC data is allowed
        enforcer.check_classification_constraint(EnumDataClassification.PUBLIC)

        # INTERNAL and above should fail
        with pytest.raises(SecurityViolationError):
            enforcer.check_classification_constraint(EnumDataClassification.INTERNAL)


class TestEnforcerIntegration:
    """Integration tests for the invocation security enforcer.

    Tests statelessness, correlation ID propagation, and combined
    policy enforcement scenarios.
    """

    def test_enforcer_is_stateless_after_init(self) -> None:
        """Enforcer should be stateless after initialization.

        Multiple calls to check methods should not affect internal state.
        The enforcer should behave identically regardless of call history.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"api-keys"}),
            allowed_domains=["api.example.com"],
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # ACT - Multiple calls
        enforcer.check_domain_access("api.example.com")
        enforcer.check_domain_access("api.example.com")
        enforcer.check_secret_scope_access("api-keys")
        enforcer.check_secret_scope_access("api-keys")

        # ASSERT - Still works identically
        result = enforcer.check_domain_access("api.example.com")
        assert result is None  # Should return None on success

    def test_security_violation_error_has_correlation_id(self) -> None:
        """SecurityViolationError should support correlation ID for tracing.

        When enforcer is created with a correlation_id, that ID should
        be propagated to any SecurityViolationError raised.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        correlation_id = uuid4()
        enforcer = InvocationSecurityEnforcer(
            handler_policy, correlation_id=correlation_id
        )

        # ACT & ASSERT
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_domain_access("forbidden.com")

        assert exc_info.value.correlation_id == correlation_id

    def test_security_violation_error_auto_generates_correlation_id(self) -> None:
        """SecurityViolationError should auto-generate correlation ID if not provided.

        When no correlation_id is passed to enforcer, errors should still
        have a valid correlation_id (auto-generated UUID4).
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)  # No correlation_id

        # ACT & ASSERT
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_domain_access("forbidden.com")

        # Should have a correlation_id (auto-generated)
        assert exc_info.value.correlation_id is not None

    def test_combined_policy_enforcement(self) -> None:
        """All policy checks should work together correctly.

        A handler with restrictive policies should be able to pass
        all checks when operating within its declared permissions.
        """
        # ARRANGE - Restrictive but valid policy
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"api-keys", "cache-config"}),
            allowed_domains=["api.internal.com", "*.cache.internal.com"],
            data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # ACT & ASSERT - All within policy should succeed
        enforcer.check_domain_access("api.internal.com")
        enforcer.check_domain_access("redis.cache.internal.com")
        enforcer.check_secret_scope_access("api-keys")
        enforcer.check_secret_scope_access("cache-config")
        enforcer.check_classification_constraint(EnumDataClassification.INTERNAL)
        enforcer.check_classification_constraint(EnumDataClassification.CONFIDENTIAL)

        # Out of policy should fail
        with pytest.raises(SecurityViolationError):
            enforcer.check_domain_access("api.external.com")

        with pytest.raises(SecurityViolationError):
            enforcer.check_secret_scope_access("database-creds")

        with pytest.raises(SecurityViolationError):
            enforcer.check_classification_constraint(EnumDataClassification.RESTRICTED)

    def test_enforcer_policy_is_immutable(self) -> None:
        """Enforcer's policy reference should not allow external mutation.

        Once created, the enforcer's behavior should be consistent.
        External changes to the policy object should not affect enforcer.
        """
        # ARRANGE
        allowed_domains_list = ["api.example.com"]
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=allowed_domains_list,
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # Verify initial behavior
        enforcer.check_domain_access("api.example.com")

        # Even if we try to mutate the original list, enforcer should be unaffected
        # (This tests that enforcer makes a defensive copy or uses immutable structures)
        allowed_domains_list.append("api.hacked.com")

        # ASSERT - Original behavior should be preserved
        # (If enforcer shares reference, this would NOT raise)
        with pytest.raises(SecurityViolationError):
            enforcer.check_domain_access("api.hacked.com")


class TestSecurityViolationErrorAttributes:
    """Tests for SecurityViolationError attributes and behavior.

    Validates that the error class has all required attributes
    for proper error handling and observability.
    """

    def test_error_includes_rule_id(self) -> None:
        """SecurityViolationError should include rule_id attribute."""
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # ACT & ASSERT
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_domain_access("forbidden.com")

        # Error should have rule_id attribute
        assert hasattr(exc_info.value, "rule_id")
        assert exc_info.value.rule_id == EnumSecurityRuleId.DOMAIN_ACCESS_DENIED

    def test_error_message_is_descriptive(self) -> None:
        """SecurityViolationError message should be descriptive.

        Error messages should include:
        - What resource was denied
        - Why it was denied (rule violated)
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=["api.allowed.com"],
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # ACT & ASSERT
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_domain_access("api.forbidden.com")

        error_message = str(exc_info.value)
        assert "api.forbidden.com" in error_message
        # Should indicate this is a domain access violation
        assert "domain" in error_message.lower() or "DOMAIN" in error_message

    def test_error_extends_onex_error_hierarchy(self) -> None:
        """SecurityViolationError should extend ONEX error hierarchy.

        This ensures proper integration with ONEX error handling patterns.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        enforcer = InvocationSecurityEnforcer(handler_policy)

        # ACT & ASSERT
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_domain_access("forbidden.com")

        # SecurityViolationError should be catchable as a broader exception type
        # (This will be validated once we know the exact hierarchy)
        error = exc_info.value
        assert isinstance(error, SecurityViolationError)
        # Should also be an Exception
        assert isinstance(error, Exception)


__all__: list[str] = [
    "TestClassificationConstraintEnforcement",
    "TestDomainAccessEnforcement",
    "TestEnforcerIntegration",
    "TestSecretScopeAccessEnforcement",
    "TestSecurityViolationErrorAttributes",
]
