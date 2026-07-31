# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Executable RED-control bindings for application database enforcement."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from omnibase_infra.validation.enums.enum_application_database_enforcement_gate import (
    EnumApplicationDatabaseEnforcementGate,
)

_CONTROL_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class ApplicationDatabaseRedControlBinding:
    """One control's exact executable source proof and optional Docker result."""

    control_id: str
    pytest_node_id: str
    docker_result: str | None = None

    def __post_init__(self) -> None:
        """Reject broad or semantically unrelated pytest bindings."""
        if _CONTROL_ID.fullmatch(self.control_id) is None:
            raise ValueError(f"invalid RED control id: {self.control_id!r}")
        path, separator, test_case = self.pytest_node_id.partition("::")
        if (
            not separator
            or not path.startswith("tests/")
            or not path.endswith(".py")
            or not test_case.startswith("test_")
            or "::" in test_case
        ):
            raise ValueError(
                f"invalid exact pytest node id for {self.control_id!r}: "
                f"{self.pytest_node_id!r}"
            )
        normalized = self.control_id.replace("-", "_")
        dedicated_test = f"test_red_control_{normalized}"
        exact_case = f"[{self.control_id}]"
        if test_case != dedicated_test and not test_case.endswith(exact_case):
            raise ValueError(
                f"RED control {self.control_id!r} must bind to an exact semantic "
                f"pytest case named {dedicated_test!r} or ending in {exact_case!r}"
            )
        if self.docker_result is not None:
            expected = f"domain_control={self.control_id} status=PASS"
            if self.docker_result != expected:
                raise ValueError(
                    f"RED control {self.control_id!r} Docker result must be "
                    f"{expected!r}"
                )


ApplicationDatabaseRedControlRegistry = Mapping[
    EnumApplicationDatabaseEnforcementGate,
    Mapping[str, ApplicationDatabaseRedControlBinding],
]

_OWNERSHIP = "tests/unit/validation/test_application_relation_ownership.py"
_DOMAIN = "tests/unit/validation/test_application_database_domain_enforcement.py"
_ACL = "tests/unit/validation/test_application_database_acl.py"
_TOPOLOGY = "tests/unit/topology/test_application_database_topology.py"
_ADAPTER = "tests/unit/runtime/auto_wiring/test_projection_domain_adapters.py"
_SQL_GATE = "tests/ci/test_application_database_sql_gate.py"


def _case(
    path: str,
    test: str,
    control_id: str,
    *,
    docker: bool = False,
) -> ApplicationDatabaseRedControlBinding:
    return ApplicationDatabaseRedControlBinding(
        control_id=control_id,
        pytest_node_id=f"{path}::{test}[{control_id}]",
        docker_result=(f"domain_control={control_id} status=PASS" if docker else None),
    )


def _control(
    path: str,
    control_id: str,
    *,
    docker: bool = False,
) -> ApplicationDatabaseRedControlBinding:
    return ApplicationDatabaseRedControlBinding(
        control_id=control_id,
        pytest_node_id=(f"{path}::test_red_control_{control_id.replace('-', '_')}"),
        docker_result=(f"domain_control={control_id} status=PASS" if docker else None),
    )


def _freeze_registry(
    registry: dict[
        EnumApplicationDatabaseEnforcementGate,
        dict[str, ApplicationDatabaseRedControlBinding],
    ],
) -> ApplicationDatabaseRedControlRegistry:
    """Freeze the reviewable control-to-result registry at both mapping levels."""
    bindings = [binding for proofs in registry.values() for binding in proofs.values()]
    mismatched_keys = sorted(
        control_id
        for proofs in registry.values()
        for control_id, binding in proofs.items()
        if binding.control_id != control_id
    )
    if mismatched_keys:
        raise ValueError(f"RED control registry key drift: {mismatched_keys}")
    node_ids = [binding.pytest_node_id for binding in bindings]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("RED controls must bind to unique exact pytest node ids")
    return MappingProxyType(
        {gate: MappingProxyType(dict(proofs)) for gate, proofs in registry.items()}
    )


APPLICATION_DATABASE_RED_CONTROL_REGISTRY = _freeze_registry(
    {
        EnumApplicationDatabaseEnforcementGate.CLASSIFICATION: {
            "missing-owner": _case(
                _OWNERSHIP, "test_seeded_ownership_defects_fail_closed", "missing-owner"
            ),
            "duplicate-owner": _case(
                _OWNERSHIP,
                "test_seeded_ownership_defects_fail_closed",
                "duplicate-owner",
            ),
            "conflicting-location": _case(
                _OWNERSHIP,
                "test_seeded_ownership_defects_fail_closed",
                "conflicting-location",
            ),
            "incomplete-retained-census": _control(
                _OWNERSHIP, "incomplete-retained-census"
            ),
            "empty-authoritative-relation-set": _control(
                _DOMAIN, "empty-authoritative-relation-set", docker=True
            ),
            "public-catalog-leak": _control(
                _DOMAIN, "public-catalog-leak", docker=True
            ),
        },
        EnumApplicationDatabaseEnforcementGate.SCHEMA_QUALIFICATION: {
            "public-application-table": _control(_DOMAIN, "public-application-table"),
            "unqualified-application-table": _control(
                _DOMAIN, "unqualified-application-table"
            ),
            "unqualified-application-mutation-target": _control(
                _DOMAIN, "unqualified-application-mutation-target"
            ),
            "unknown-topology-schema": _control(_DOMAIN, "unknown-topology-schema"),
            "wrong-object-kind": _control(_SQL_GATE, "wrong-object-kind"),
            "wrong-routine-overload": _control(_SQL_GATE, "wrong-routine-overload"),
            "dynamic-sql-target": _case(
                _DOMAIN,
                "test_valid_postgres_alternate_target_forms_fail_closed",
                "dynamic-sql-target",
            ),
            "implicit-multirange-identity": _case(
                _DOMAIN,
                "test_valid_postgres_alternate_target_forms_fail_closed",
                "implicit-multirange-identity",
            ),
        },
        EnumApplicationDatabaseEnforcementGate.TENANT_RLS: {
            "tenant-text-key": _case(
                _DOMAIN,
                "test_seeded_tenant_shape_defects_fail_closed",
                "tenant-text-key",
                docker=True,
            ),
            "tenant-nullable": _case(
                _DOMAIN,
                "test_seeded_tenant_shape_defects_fail_closed",
                "tenant-nullable",
                docker=True,
            ),
            "tenant-default": _case(
                _DOMAIN,
                "test_seeded_tenant_shape_defects_fail_closed",
                "tenant-default",
                docker=True,
            ),
            "missing-enable-rls": _case(
                _DOMAIN,
                "test_seeded_tenant_shape_defects_fail_closed",
                "missing-enable-rls",
                docker=True,
            ),
            "missing-force-rls": _case(
                _DOMAIN,
                "test_seeded_tenant_shape_defects_fail_closed",
                "missing-force-rls",
                docker=True,
            ),
            "using-drift": _case(
                _DOMAIN,
                "test_seeded_policy_predicate_drift_fails_closed",
                "using-drift",
                docker=True,
            ),
            "with-check-drift": _case(
                _DOMAIN,
                "test_seeded_policy_predicate_drift_fails_closed",
                "with-check-drift",
                docker=True,
            ),
            "canonical-policy-unrelated-role": _control(
                _DOMAIN, "canonical-policy-unrelated-role", docker=True
            ),
            "uncontracted-identity-root": _control(
                _DOMAIN, "uncontracted-identity-root", docker=True
            ),
            "identity-root-runtime-login": _control(
                _DOMAIN, "identity-root-runtime-login", docker=True
            ),
            "identity-root-unproven-enumeration": _control(
                _DOMAIN, "identity-root-unproven-enumeration", docker=True
            ),
            "identity-root-runtime-membership": _control(
                _DOMAIN, "identity-root-runtime-membership", docker=True
            ),
            "identity-root-runtime-set-role": _control(
                _DOMAIN, "identity-root-runtime-set-role", docker=True
            ),
            "widening-permissive-policy": _control(
                _DOMAIN, "widening-permissive-policy"
            ),
            "owner-security-view": _control(
                _DOMAIN, "owner-security-view", docker=True
            ),
            "unsafe-security-definer": _control(
                _DOMAIN, "unsafe-security-definer", docker=True
            ),
            "unproven-security-view": _control(
                _DOMAIN, "unproven-security-view", docker=True
            ),
            "unproven-security-definer": _control(
                _DOMAIN, "unproven-security-definer", docker=True
            ),
            "security-definer-volatility-drift": _control(
                _DOMAIN, "security-definer-volatility-drift", docker=True
            ),
        },
        EnumApplicationDatabaseEnforcementGate.INTERNAL_CATALOG: {
            "internal-tenant-id": _case(
                _DOMAIN,
                "test_non_tenant_domain_red_control_fails_closed",
                "internal-tenant-id",
                docker=True,
            ),
            "catalog-tenant-id": _case(
                _DOMAIN,
                "test_non_tenant_domain_red_control_fails_closed",
                "catalog-tenant-id",
            ),
            "internal-tenant-policy": _case(
                _DOMAIN,
                "test_non_tenant_domain_red_control_fails_closed",
                "internal-tenant-policy",
            ),
            "catalog-rls": _case(
                _DOMAIN,
                "test_non_tenant_domain_red_control_fails_closed",
                "catalog-rls",
            ),
            "uncontracted-source-tenant": _case(
                _DOMAIN,
                "test_source_tenant_id_is_typed_non_authoritative_provenance",
                "uncontracted-source-tenant",
                docker=True,
            ),
            "source-tenant-generated-unique-alias": _control(
                _DOMAIN, "source-tenant-generated-unique-alias", docker=True
            ),
        },
        EnumApplicationDatabaseEnforcementGate.ROLE_ACL: {
            "public-connect": _case(
                _ACL,
                "test_seeded_acl_red_control_fails_closed",
                "public-connect",
            ),
            "public-execute": _case(
                _ACL,
                "test_seeded_acl_red_control_fails_closed",
                "public-execute",
            ),
            "runtime-owner": _case(
                _ACL, "test_seeded_acl_red_control_fails_closed", "runtime-owner"
            ),
            "runtime-ddl": _case(
                _ACL,
                "test_seeded_acl_red_control_fails_closed",
                "runtime-ddl",
            ),
            "runtime-bypassrls": _control(_ACL, "runtime-bypassrls"),
            "cross-domain-grant": _control(_ACL, "cross-domain-grant"),
            "unsafe-default-privilege": _case(
                _ACL,
                "test_seeded_acl_red_control_fails_closed",
                "unsafe-default-privilege",
            ),
        },
        EnumApplicationDatabaseEnforcementGate.ONE_DATABASE: {
            "old-application-database": _control(
                _DOMAIN, "old-application-database", docker=True
            ),
            "duplicate-pool-user": _control(
                _DOMAIN, "duplicate-pool-user", docker=True
            ),
            "wrong-pool-user": _control(_DOMAIN, "wrong-pool-user"),
            "missing-pool-binding": _control(_DOMAIN, "missing-pool-binding"),
        },
        EnumApplicationDatabaseEnforcementGate.ADAPTER: {
            "untrusted-tenant-selection": _control(
                _ADAPTER, "untrusted-tenant-selection"
            ),
            "mismatched-signer-binding": _control(
                _ADAPTER, "mismatched-signer-binding"
            ),
            "nonlocal-tenant-guc": _control(_ADAPTER, "nonlocal-tenant-guc"),
            "leaked-tenant-guc": _control(_ADAPTER, "leaked-tenant-guc"),
            "internal-resolver-call": _control(_ADAPTER, "internal-resolver-call"),
            "domain-blind-upsert": _control(_ADAPTER, "domain-blind-upsert"),
        },
        EnumApplicationDatabaseEnforcementGate.TOPOLOGY_PARITY: {
            "profile-instance-drift": _control(_TOPOLOGY, "profile-instance-drift"),
            "database-user-drift": _case(
                _TOPOLOGY,
                "test_seeded_database_user_schema_and_dsn_drift_fail_closed",
                "database-user-drift",
            ),
            "docker-profile-injection-drift": _control(
                _TOPOLOGY, "docker-profile-injection-drift"
            ),
            "docker-dsn-consumer-drift": _case(
                _TOPOLOGY,
                "test_seeded_docker_database_or_dsn_drift_fails_closed",
                "docker-dsn-consumer-drift",
            ),
        },
    }
)


__all__ = [
    "APPLICATION_DATABASE_RED_CONTROL_REGISTRY",
    "ApplicationDatabaseRedControlBinding",
    "ApplicationDatabaseRedControlRegistry",
]
