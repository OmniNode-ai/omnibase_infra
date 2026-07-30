# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Generate the one-database ACL matrix from immutable Git blobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_infra.validation.application_database_acl import (
    build_application_database_acl_matrix,
    render_application_database_acl_sql,
)
from omnibase_infra.validation.enums.enum_application_database_acl_authorization_scope import (
    EnumApplicationDatabaseAclAuthorizationScope,
)
from omnibase_infra.validation.enums.enum_application_database_acl_render_phase import (
    EnumApplicationDatabaseAclRenderPhase,
)
from omnibase_infra.validation.models.model_application_database_acl_matrix import (
    ModelApplicationDatabaseAclSource,
)
from omnibase_infra.validation.models.model_application_database_acl_policy import (
    ModelApplicationDatabaseAclPolicy,
)
from omnibase_infra.validation.models.model_application_database_activity_result_evidence import (
    ModelApplicationDatabaseActivityResultEvidence,
)
from omnibase_infra.validation.models.model_application_database_catalog_result_evidence import (
    ModelApplicationDatabaseCatalogResultEvidence,
)
from omnibase_infra.validation.models.model_application_database_principal_inventory import (
    ModelApplicationDatabasePrincipalInventory,
)
from omnibase_infra.validation.models.model_application_relation_evidence_inventory import (
    ModelApplicationRelationEvidenceInventory,
)
from omnibase_infra.validation.models.model_migration_ownership_manifest import (
    ModelMigrationOwnershipManifest,
)


class _SourceLock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"]
    required_connect_databases: tuple[str, ...]
    sources: tuple[ModelApplicationDatabaseAclSource, ...]


class _IndentedSafeDumper(yaml.SafeDumper):
    """Match the repository YAML formatter's nested-sequence indentation."""

    def increase_indent(
        self,
        flow: bool = False,
        indentless: bool = False,
    ) -> None:
        return super().increase_indent(flow=flow, indentless=False)


def _parse_repository_roots(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        repository, separator, raw_path = value.partition("=")
        if not separator or not repository or not raw_path:
            raise ValueError("--repository-root must be REPOSITORY=/absolute/git/clone")
        root = Path(raw_path).resolve()
        if not root.is_dir():
            raise ValueError(f"Repository root does not exist: {root}")
        result[repository] = root
    return result


def _git_blob(root: Path, source: ModelApplicationDatabaseAclSource) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"{source.revision}:{source.path}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            f"Cannot read {source.source_key} from {source.repository}@"
            f"{source.revision}: {detail}"
        )
    digest = hashlib.sha256(result.stdout).hexdigest()
    if digest != source.sha256:
        raise ValueError(
            f"Source-lock digest mismatch for {source.source_key}: "
            f"expected {source.sha256}, got {digest}"
        )
    return result.stdout


def _mapping(blob: bytes, source_id: str) -> object:
    try:
        if blob.lstrip().startswith((b"{", b"[")):
            return json.loads(blob)
        return yaml.safe_load(blob)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot parse locked source {source_id}: {exc}") from exc


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(
            value,
            Dumper=_IndentedSafeDumper,
            default_flow_style=False,
            sort_keys=True,
            width=1_000_000,
        ),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=Path("docker/application-acl-proof/source-lock.yaml"),
    )
    parser.add_argument(
        "--repository-root",
        action="append",
        default=[],
        metavar="REPOSITORY=PATH",
        help="Local Git clone containing every locked revision (repeatable)",
    )
    parser.add_argument("--matrix-output", type=Path, required=True)
    parser.add_argument("--sql-output", type=Path)
    parser.add_argument(
        "--render-phase",
        type=EnumApplicationDatabaseAclRenderPhase,
        choices=tuple(EnumApplicationDatabaseAclRenderPhase),
        default=EnumApplicationDatabaseAclRenderPhase.FULL,
        help="Render the additive scaffold before full object materialization",
    )
    parser.add_argument(
        "--allow-blocked-matrix",
        action="store_true",
        help="Write candidate evidence but never SQL when dependencies are incomplete",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    lock = _SourceLock.model_validate(
        yaml.safe_load(args.source_lock.read_text(encoding="utf-8"))
    )
    roots = _parse_repository_roots(args.repository_root)
    missing_roots = sorted(
        {source.repository for source in lock.sources} - roots.keys()
    )
    if missing_roots:
        raise ValueError(f"Missing --repository-root values: {missing_roots!r}")

    blobs = {
        source.source_key: _git_blob(roots[source.repository], source)
        for source in lock.sources
    }
    topology_sources = [
        source for source in lock.sources if source.purpose == "topology"
    ]
    if len(topology_sources) != 1:
        raise ValueError("Source lock must contain exactly one topology source")
    topology_source = topology_sources[0]
    topology = ModelDeploymentTopology.model_validate(
        _mapping(blobs[topology_source.source_key], topology_source.source_key)
    )
    inventories = {
        source.source_key: ModelApplicationRelationEvidenceInventory.model_validate(
            _mapping(blobs[source.source_key], source.source_key)
        )
        for source in lock.sources
        if source.purpose == "relation_inventory"
    }
    manifests = {
        source.source_key: ModelMigrationOwnershipManifest.model_validate(
            _mapping(blobs[source.source_key], source.source_key)
        )
        for source in lock.sources
        if source.purpose == "service_ownership"
    }
    principal_inventories = {
        source.source_key: ModelApplicationDatabasePrincipalInventory.model_validate(
            _mapping(blobs[source.source_key], source.source_key)
        )
        for source in lock.sources
        if source.purpose == "principal_inventory"
    }
    catalog_results = {
        source.source_key: ModelApplicationDatabaseCatalogResultEvidence.model_validate(
            _mapping(blobs[source.source_key], source.source_key)
        )
        for source in lock.sources
        if source.purpose == "catalog_result_evidence"
    }
    activity_results = {
        source.source_key: ModelApplicationDatabaseActivityResultEvidence.model_validate(
            _mapping(blobs[source.source_key], source.source_key)
        )
        for source in lock.sources
        if source.purpose == "activity_result_evidence"
    }
    acl_policies = {
        source.source_key: ModelApplicationDatabaseAclPolicy.model_validate(
            _mapping(blobs[source.source_key], source.source_key)
        )
        for source in lock.sources
        if source.purpose == "acl_policy"
    }
    matrix = build_application_database_acl_matrix(
        topology=topology,
        sources=lock.sources,
        relation_inventories=inventories,
        service_manifests=manifests,
        principal_inventories=principal_inventories,
        acl_policies=acl_policies,
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT,
        required_connect_databases=lock.required_connect_databases,
        catalog_results=catalog_results,
        activity_results=activity_results,
    )
    _write_yaml(args.matrix_output, matrix.model_dump(mode="json"))
    print(
        f"matrix_status={matrix.status} scaffold_status={matrix.scaffold_status} "
        f"objects={len(matrix.objects)} "
        f"rows={len(matrix.rows)} defaults={len(matrix.default_privileges)} "
        f"blockers={len(matrix.blockers)} "
        f"scaffold_blockers={len(matrix.scaffold_blockers)}"
    )
    phase_blocked = (
        matrix.status == "BLOCKED"
        if args.render_phase is EnumApplicationDatabaseAclRenderPhase.FULL
        else matrix.scaffold_status == "BLOCKED"
    )
    phase_blockers = (
        matrix.blockers
        if args.render_phase is EnumApplicationDatabaseAclRenderPhase.FULL
        else matrix.scaffold_blockers
    )
    if phase_blocked:
        for blocker in phase_blockers:
            print(f"blocker={blocker}")
        if args.sql_output is not None and args.sql_output.exists():
            raise ValueError(
                "Refusing to overwrite an existing SQL output from a blocked phase"
            )
        return 0 if args.allow_blocked_matrix else 2
    if args.sql_output is not None:
        args.sql_output.parent.mkdir(parents=True, exist_ok=True)
        args.sql_output.write_text(
            render_application_database_acl_sql(matrix, phase=args.render_phase),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"application ACL generation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
