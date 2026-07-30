# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed, generated application-database ACL matrix."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_infra.validation.enums.enum_application_database_acl_authorization_scope import (
    EnumApplicationDatabaseAclAuthorizationScope,
)
from omnibase_infra.validation.models.model_application_database_acl_object import (
    ModelApplicationDatabaseAclObject,
)
from omnibase_infra.validation.models.model_application_database_acl_row import (
    ModelApplicationDatabaseAclRow,
)
from omnibase_infra.validation.models.model_application_database_acl_source import (
    ModelApplicationDatabaseAclSource,
)
from omnibase_infra.validation.models.model_application_database_catalog_object_evidence import (
    ModelApplicationDatabaseCatalogObjectEvidence,
)
from omnibase_infra.validation.models.model_application_database_default_acl_row import (
    ModelApplicationDatabaseDefaultAclRow,
)
from omnibase_infra.validation.models.model_application_database_observed_role_state import (
    ModelApplicationDatabaseObservedRoleState,
)
from omnibase_infra.validation.models.model_application_database_role_membership import (
    ModelApplicationDatabaseRoleMembership,
)
from omnibase_infra.validation.models.model_application_database_role_state import (
    ModelApplicationDatabaseRoleState,
)


class ModelApplicationDatabaseAclMatrix(BaseModel):
    """Complete generated role/object/default-privilege projection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    authorization_scope: EnumApplicationDatabaseAclAuthorizationScope
    scaffold_status: Literal["READY", "BLOCKED"]
    scaffold_blockers: tuple[str, ...] = ()
    status: Literal["READY", "BLOCKED"]
    sources: tuple[ModelApplicationDatabaseAclSource, ...] = Field(..., min_length=1)
    verified_evidence_source_keys: tuple[str, ...] = ()
    declared_principals: dict[str, tuple[str, ...]]
    observed_principals: dict[str, tuple[str, ...]]
    absent_principals: dict[str, tuple[str, ...]]
    observed_owner_roles: tuple[str, ...]
    absent_owner_roles: tuple[str, ...]
    observed_role_states: tuple[ModelApplicationDatabaseObservedRoleState, ...]
    governed_role_states: tuple[ModelApplicationDatabaseRoleState, ...]
    retained_administrative_principals: tuple[str, ...]
    database_owners: dict[str, str]
    required_connect_databases: tuple[str, ...]
    observed_connect_database_owners: dict[str, str]
    allowed_connect_principals: dict[str, tuple[str, ...]]
    observed_connect_principals: dict[str, tuple[str, ...]]
    absent_connect_principals: dict[str, tuple[str, ...]]
    schema_domains: dict[str, dict[str, EnumDatabaseSchemaDomain]]
    observed_schema_owners: dict[str, dict[str, str]]
    absent_schemas: dict[str, tuple[str, ...]]
    principal_domains: dict[str, tuple[EnumDatabaseSchemaDomain, ...]]
    allowed_memberships: tuple[ModelApplicationDatabaseRoleMembership, ...]
    observed_objects: tuple[ModelApplicationDatabaseCatalogObjectEvidence, ...]
    objects: tuple[ModelApplicationDatabaseAclObject, ...]
    rows: tuple[ModelApplicationDatabaseAclRow, ...]
    default_privileges: tuple[ModelApplicationDatabaseDefaultAclRow, ...]
    blockers: tuple[str, ...] = ()
    excluded_objects: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_matrix(self) -> ModelApplicationDatabaseAclMatrix:
        """Keep the artifact deterministic, unique, and honest about blockers."""
        source_keys = [source.source_key for source in self.sources]
        if len(set(source_keys)) != len(source_keys):
            raise ValueError("ACL matrix source keys must be unique")
        if len(set(self.verified_evidence_source_keys)) != len(
            self.verified_evidence_source_keys
        ):
            raise ValueError("verified evidence source keys must be unique")
        source_by_key = {source.source_key: source for source in self.sources}
        unknown_verified = sorted(
            set(self.verified_evidence_source_keys) - set(source_by_key)
        )
        if unknown_verified:
            raise ValueError(
                f"verified evidence contains unknown source keys: {unknown_verified!r}"
            )
        invalid_verified = sorted(
            source_key
            for source_key in self.verified_evidence_source_keys
            if source_by_key[source_key].purpose
            not in {"catalog_result_evidence", "activity_result_evidence"}
        )
        if invalid_verified:
            raise ValueError(
                "only catalog/activity result blobs can be semantically verified: "
                f"{invalid_verified!r}"
            )
        if (
            self.authorization_scope
            is EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT
            and (self.status == "READY" or self.scaffold_status == "READY")
        ):
            locked_result_keys = {
                source.source_key
                for source in self.sources
                if source.purpose
                in {"catalog_result_evidence", "activity_result_evidence"}
            }
            if not locked_result_keys or locked_result_keys != set(
                self.verified_evidence_source_keys
            ):
                raise ValueError(
                    "READY deployment phase requires every locked catalog/activity "
                    "result source to be semantically verified"
                )
        object_ids = [item.identity for item in self.objects]
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("ACL matrix object identities must be unique")
        row_ids = [row.identity for row in self.rows]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("ACL matrix row identities must be unique")
        default_ids = [
            (
                row.owner,
                row.database_ref,
                row.schema_ref,
                row.object_type,
                row.grantee,
            )
            for row in self.default_privileges
        ]
        if len(set(default_ids)) != len(default_ids):
            raise ValueError("Default ACL row identities must be unique")
        membership_ids = [item.identity for item in self.allowed_memberships]
        if len(set(membership_ids)) != len(membership_ids):
            raise ValueError("Role membership identities must be unique")
        if set(self.declared_principals) != set(self.observed_principals) or set(
            self.declared_principals
        ) != set(self.absent_principals):
            raise ValueError(
                "declared/observed/absent principal database keys disagree"
            )
        if set(self.schema_domains) != set(self.declared_principals):
            raise ValueError("schema_domains and principal database keys disagree")
        if set(self.observed_schema_owners) != set(self.schema_domains) or set(
            self.absent_schemas
        ) != set(self.schema_domains):
            raise ValueError("schema presence/absence database keys are incomplete")
        for database_ref, schemas in self.schema_domains.items():
            observed_schemas = set(self.observed_schema_owners[database_ref])
            absent_schemas = set(self.absent_schemas[database_ref])
            if observed_schemas & absent_schemas:
                raise ValueError(
                    f"{database_ref}: schema presence/absence evidence overlaps"
                )
            if self.scaffold_status == "READY" and set(schemas) != (
                observed_schemas.union(absent_schemas)
            ):
                raise ValueError(
                    f"{database_ref}: READY scaffold schema evidence is incomplete"
                )
        observed_object_ids = [obj.identity for obj in self.observed_objects]
        if len(set(observed_object_ids)) != len(observed_object_ids):
            raise ValueError("observed catalog object identities must be unique")
        expected_object_ids = {
            (
                obj.catalog_kind,
                obj.schema_ref,
                obj.object_ref,
                obj.function_signature or "",
            )
            for obj in self.objects
        }
        if self.status == "READY" and set(observed_object_ids) != expected_object_ids:
            raise ValueError(
                "READY matrix requires exact live-catalog object identity coverage"
            )
        if len(set(self.required_connect_databases)) != len(
            self.required_connect_databases
        ):
            raise ValueError("required_connect_databases must be unique")
        invalid_databases = sorted(
            database
            for database in self.required_connect_databases
            if re.fullmatch(r"[a-z_][a-z0-9_]*", database) is None
        )
        if invalid_databases:
            raise ValueError(
                f"required_connect_databases contain unsafe names: {invalid_databases!r}"
            )
        if set(self.allowed_connect_principals) != set(
            self.observed_connect_principals
        ) or set(self.allowed_connect_principals) != set(
            self.absent_connect_principals
        ):
            raise ValueError("allowed/observed/absent CONNECT database keys disagree")
        if set(self.observed_connect_database_owners) != set(
            self.allowed_connect_principals
        ):
            raise ValueError(
                "observed CONNECT owner and principal database keys disagree"
            )
        if not set(self.allowed_connect_principals) <= set(
            self.required_connect_databases
        ):
            raise ValueError("CONNECT policies contain unrequired databases")
        if self.status == "READY" and set(self.allowed_connect_principals) != set(
            self.required_connect_databases
        ):
            raise ValueError("READY matrix requires CONNECT policy for every database")
        for database, allowed in self.allowed_connect_principals.items():
            observed = self.observed_connect_principals[database]
            absent = self.absent_connect_principals[database]
            if set(observed) & set(absent):
                raise ValueError(
                    f"{database}: CONNECT presence and absence evidence overlap"
                )
            if not set(allowed) <= set(observed).union(absent):
                raise ValueError(
                    f"{database}: allowed CONNECT principals lack presence/absence evidence"
                )
        globally_observed = {
            principal
            for principals in (
                *self.observed_principals.values(),
                *self.observed_connect_principals.values(),
                self.observed_owner_roles,
                tuple(self.observed_connect_database_owners.values()),
            )
            for principal in principals
        }
        globally_absent = {
            principal
            for principals in (
                *self.absent_principals.values(),
                *self.absent_connect_principals.values(),
                self.absent_owner_roles,
            )
            for principal in principals
        }
        global_evidence_conflicts = sorted(globally_observed & globally_absent)
        if global_evidence_conflicts:
            raise ValueError(
                "cluster-global role presence/absence evidence conflicts: "
                f"{global_evidence_conflicts!r}"
            )
        managed_owner_evidence = set(self.observed_owner_roles).union(
            self.absent_owner_roles
        )
        workload_evidence = {
            principal
            for principals in (
                *self.declared_principals.values(),
                *self.observed_principals.values(),
                *self.absent_principals.values(),
                *self.allowed_connect_principals.values(),
                *self.observed_connect_principals.values(),
                *self.absent_connect_principals.values(),
            )
            for principal in principals
        }
        role_kind_overlap = sorted(managed_owner_evidence & workload_evidence)
        if role_kind_overlap:
            raise ValueError(
                "managed owner and workload role evidence overlap: "
                f"{role_kind_overlap!r}"
            )
        invalid_owner_evidence = sorted(
            role
            for role in managed_owner_evidence.union(
                self.observed_connect_database_owners.values()
            )
            if re.fullmatch(r"[a-z_][a-z0-9_]*", role) is None
        )
        if invalid_owner_evidence:
            raise ValueError(
                f"owner evidence contains unsafe role names: {invalid_owner_evidence!r}"
            )
        desired_managed_owners = (
            set(self.database_owners.values())
            .union(obj.owner for obj in self.objects)
            .union(row.owner for row in self.default_privileges)
        )
        if self.scaffold_status == "READY" and not desired_managed_owners <= (
            managed_owner_evidence
        ):
            raise ValueError(
                "READY scaffold managed owners lack presence/absence evidence: "
                f"{sorted(desired_managed_owners - managed_owner_evidence)!r}"
            )
        governed_roles = [state.role for state in self.governed_role_states]
        if len(set(governed_roles)) != len(governed_roles):
            raise ValueError("governed role states must have unique role names")
        if self.scaffold_status == "READY" and not set(governed_roles) <= (
            globally_observed.union(globally_absent)
        ):
            raise ValueError(
                "READY scaffold governed roles lack presence/absence evidence"
            )
        observed_state_roles = [state.role for state in self.observed_role_states]
        if len(set(observed_state_roles)) != len(observed_state_roles):
            raise ValueError("observed role states must have unique role names")
        observed_principal_roles = {
            principal
            for principals in (
                *self.observed_principals.values(),
                *self.observed_connect_principals.values(),
                self.observed_owner_roles,
            )
            for principal in principals
        }
        if set(observed_state_roles) != observed_principal_roles:
            raise ValueError(
                "observed role states must cover the global observed principal and "
                "managed-owner census"
            )
        retained_admins = set(self.retained_administrative_principals)
        if len(retained_admins) != len(self.retained_administrative_principals):
            raise ValueError("retained administrative principals must be unique")
        if not retained_admins <= observed_principal_roles:
            raise ValueError(
                "retained administrative principals must be in the observed census"
            )
        if retained_admins & set(governed_roles):
            raise ValueError(
                "retained administrative principals cannot be governed roles"
            )
        if not set(self.database_owners) <= set(self.declared_principals):
            raise ValueError("database_owners contain unknown database keys")
        if self.status == "READY" and set(self.database_owners) != set(
            self.declared_principals
        ):
            raise ValueError("READY matrix requires an owner for every database")
        if any(
            len(set(principals)) != len(principals)
            for principals in (
                *self.declared_principals.values(),
                *self.observed_principals.values(),
                *self.absent_principals.values(),
            )
        ):
            raise ValueError("principal collections must be unique")
        if (self.status == "READY") != (not self.blockers):
            raise ValueError("READY status and blockers disagree")
        if (self.scaffold_status == "READY") != (not self.scaffold_blockers):
            raise ValueError("READY scaffold_status and scaffold_blockers disagree")
        return self


__all__ = ["ModelApplicationDatabaseAclMatrix"]
