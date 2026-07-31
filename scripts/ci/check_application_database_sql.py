# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail CI when changed deployable SQL uses unsafe application targets."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from pathlib import Path

from omnibase_infra.topology.application_database import load_topology_profile
from omnibase_infra.validation.application_database_domain_enforcement import (
    application_database_created_catalog_identities,
    application_database_sql_target_requirements,
    lint_application_database_sql,
    load_application_database_ownership_identities,
)

_NON_DEPLOYABLE_SQL_PROOF_FIXTURES = frozenset(
    {Path("docker/application-domain-enforcement/seed.sql")}
)


def changed_sql_paths(
    repository: Path,
    base_revision: str,
    head_revision: str,
) -> tuple[Path, ...]:
    """Return changed deployable SQL, excluding the exact ephemeral proof seed."""
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
            f"{base_revision}...{head_revision}",
            "--",
            "*.sql",
        ],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot resolve changed SQL range: {detail}")
    relative_paths = tuple(
        Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw
    )
    resolved: list[Path] = []
    repository_root = repository.resolve()
    for relative_path in relative_paths:
        if relative_path in _NON_DEPLOYABLE_SQL_PROOF_FIXTURES:
            # This exact SQL file initializes only the rebuilt, ephemeral PG16
            # proof container. Its synthetic ownership universe is validated
            # independently and must never be composed with production owners.
            continue
        candidate = (repository_root / relative_path).resolve()
        if not candidate.is_relative_to(repository_root):
            raise RuntimeError(f"changed SQL path escapes repository: {relative_path}")
        if candidate.is_file():
            resolved.append(candidate)
    return tuple(resolved)


def validate_changed_sql(
    repository: Path,
    base_revision: str,
    head_revision: str,
    *,
    ownership_manifest_paths: Sequence[Path],
) -> tuple[str, ...]:
    """Lint every changed deployable SQL file against typed topology authority."""
    topology = load_topology_profile("local")
    violations: list[str] = []
    try:
        ownership_identities = load_application_database_ownership_identities(
            ownership_manifest_paths
        )
    except ValueError as exc:
        ownership_identities = ()
        violations.append(f"ownership manifests: {exc}")
    declared_identities = {identity.identity for identity in ownership_identities}
    for path in changed_sql_paths(repository, base_revision, head_revision):
        relative_path = path.relative_to(repository.resolve())
        sql = path.read_text(encoding="utf-8")
        for violation in lint_application_database_sql(sql, topology):
            violations.append(f"{relative_path}: {violation}")
        for requirement in application_database_sql_target_requirements(sql, topology):
            location_matches = tuple(
                identity
                for identity in ownership_identities
                if identity.schema == requirement.schema
                and identity.name == requirement.name
            )
            kind_matches = tuple(
                identity
                for identity in location_matches
                if identity.kind in requirement.allowed_kinds
            )
            if not location_matches:
                violations.append(
                    f"{relative_path}: application target "
                    f"{requirement.schema}.{requirement.name} requires exactly one "
                    "ownership declaration"
                )
            elif not kind_matches:
                violations.append(
                    f"{relative_path}: application target "
                    f"{requirement.schema}.{requirement.name} requires an exact "
                    "object-kind ownership declaration"
                )
            elif requirement.function_signature is not None and all(
                identity.function_signature != requirement.function_signature
                for identity in kind_matches
            ):
                violations.append(
                    f"{relative_path}: application target "
                    f"{requirement.schema}.{requirement.name}"
                    f"{requirement.function_signature} requires an exact routine "
                    "ownership declaration"
                )
        for identity in application_database_created_catalog_identities(sql):
            if identity.identity not in declared_identities:
                violations.append(
                    f"{relative_path}: created application object "
                    f"{identity.identity!r} lacks an authoritative ownership declaration"
                )
    return tuple(sorted(set(violations)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--head-revision", default="HEAD")
    parser.add_argument(
        "--ownership-manifest",
        action="append",
        type=Path,
        required=True,
        help="Typed ownership manifest; repeat for every authoritative source",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    violations = validate_changed_sql(
        args.repository,
        args.base_revision,
        args.head_revision,
        ownership_manifest_paths=tuple(
            path if path.is_absolute() else args.repository / path
            for path in args.ownership_manifest
        ),
    )
    if violations:
        print("application_database_sql_gate=FAIL")
        for violation in violations:
            print(violation)
        return 1
    print("application_database_sql_gate=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
