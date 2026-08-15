#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Generate and drift-check application-database ``TABLE`` grants (OMN-15656).

The typed topology instances under ``src/omnibase_infra/topology/instances`` are
the platform authority for which principal may touch which relation. Their
``object_type: TABLE`` grants are a **projection of node contract**
``db_io.db_tables`` **declarations** — this script is the only sanctioned way to
write them. Hand-listing is what ADR-0027 exists to remove.

Modes
-----
``--write``
    Regenerate the TABLE grants in every instance from the contracts.
``--check``
    Fail when the checked-in grants differ from the derivation, in either
    direction: a contract-declared table with no grant, or a granted table no
    contract declares.
``--prove``
    Resolve every contract declaration through the real
    ``_resolve_projection_database_target`` against the real shipped
    ``load_topology_profile(profile)`` for every supported profile, and report
    ``PASS``/``FAIL`` per profile. This is the end-to-end assertion that the
    grants actually satisfy the OMN-15418 validator.

``--check`` and ``--prove`` compose; CI runs both.

After ``--write`` the rendered catalogs must be regenerated with
``scripts/render_application_database_topology.py`` — the topology unit tests
fail closed if they drift.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.models.core import ModelDeploymentTopology
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import (
    SUPPORTED_TOPOLOGY_PROFILES,
    TOPOLOGY_PROFILE_INSTANCE_MAP,
)
from omnibase_infra.topology.table_grant_derivation import (
    STATE_IO_TABLE_DECLARATIONS,
    ContractTableDeclaration,
    TopologyTableGrants,
    derive_topology_table_grants,
    load_contract_declarations,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_INSTANCE_ROOT = _REPOSITORY_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
DEFAULT_CONTRACTS_ROOT = (
    _REPOSITORY_ROOT
    / ".proof-dependencies"
    / "omnimarket"
    / "src"
    / "omnimarket"
    / "nodes"
)

# Every instance that must carry the derived grants. The three files are
# byte-identical in their principal blocks by construction; this script is the
# mechanism that keeps them so, because the topology schema has no include or
# inheritance surface.
INSTANCE_NAMES = tuple(sorted(set(TOPOLOGY_PROFILE_INSTANCE_MAP.values())))


class _FlowList(list[object]):
    """Sequence rendered inline to preserve the hand-authored grant style."""


class _InstanceDumper(yaml.SafeDumper):
    """Dumper that reproduces the checked-in instance formatting exactly."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)


def _represent_flow_list(dumper: yaml.SafeDumper, data: _FlowList) -> yaml.Node:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
    """Double-quote scalars that would otherwise stop resolving as strings.

    Keeps ``schema_version: "2.0"`` intact instead of round-tripping it into
    the single-quoted form, so a regeneration diff shows only real changes.
    """
    style: str | None = None
    try:
        if not isinstance(yaml.safe_load(data), str):
            style = '"'
    except yaml.YAMLError:
        style = '"'
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_InstanceDumper.add_representer(_FlowList, _represent_flow_list)
_InstanceDumper.add_representer(str, _represent_str)


def _mark_flow(node: object) -> object:
    """Render ``privileges`` inline and everything else in block style."""
    if isinstance(node, dict):
        return {
            key: _FlowList(value) if key == "privileges" else _mark_flow(value)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_mark_flow(item) for item in node]
    return node


def _grant_to_document(grant: object) -> dict[str, object]:
    payload = grant.model_dump(mode="json", exclude_defaults=False)  # type: ignore[attr-defined]
    document: dict[str, object] = {"object_type": payload["object_type"]}
    if payload.get("objects"):
        document["objects"] = list(payload["objects"])
    document["privileges"] = list(payload["privileges"])
    if payload.get("schema") is not None:
        document["schema"] = payload["schema"]
    return document


def _render_instance(path: Path, derived: TopologyTableGrants) -> str:
    """Return the instance text with derived TABLE grants substituted in.

    Every logical database in the instance is rendered, not just
    ``application``: the omniintelligence service database (OMN-15655 AC-2)
    carries its own principal, and a renderer scoped to one database would
    leave that principal permanently grant-less while ``--check`` reported
    green.
    """
    original = path.read_text(encoding="utf-8")
    document = yaml.safe_load(original)
    for database_ref, database in document["databases"].items():
        database_grants = derived.per_database.get(database_ref)
        for principal_name, principal in database["principals"].items():
            # Drop every existing TABLE grant: this script owns that subset
            # entirely, so a stale entry must not survive a regeneration.
            retained = [
                grant
                for grant in principal.get("grants", [])
                if grant.get("object_type") != EnumDatabaseGrantObjectType.TABLE.value
            ]
            generated = (
                []
                if database_grants is None
                else [
                    _grant_to_document(grant)
                    for grant in database_grants.grants.get(principal_name, ())
                ]
            )
            principal["grants"] = retained + generated
    header = "".join(
        line for line in original.splitlines(keepends=True) if line.startswith("#")
    )
    body = yaml.dump(
        _mark_flow(document),
        Dumper=_InstanceDumper,
        sort_keys=True,
        default_flow_style=False,
        width=4096,
        allow_unicode=True,
    )
    return header + body


def _derivation_for_instance(
    instance_name: str, declarations: Sequence[ContractTableDeclaration]
) -> TopologyTableGrants:
    topology = ModelDeploymentTopology.from_yaml(
        _INSTANCE_ROOT / f"{instance_name}.yaml"
    )
    return derive_topology_table_grants(topology, declarations)


def _run_write(declarations: Sequence[ContractTableDeclaration]) -> int:
    changed = []
    for instance_name in INSTANCE_NAMES:
        path = _INSTANCE_ROOT / f"{instance_name}.yaml"
        rendered = _render_instance(
            path, _derivation_for_instance(instance_name, declarations)
        )
        if rendered != path.read_text(encoding="utf-8"):
            path.write_text(rendered, encoding="utf-8")
            changed.append(instance_name)
    print(f"instances updated: {', '.join(changed) if changed else '(none)'}")
    print(
        "reminder: regenerate the rendered catalogs with "
        "scripts/render_application_database_topology.py"
    )
    return 0


def _run_check(declarations: Sequence[ContractTableDeclaration]) -> int:
    failures: list[str] = []
    for instance_name in INSTANCE_NAMES:
        path = _INSTANCE_ROOT / f"{instance_name}.yaml"
        expected = _render_instance(
            path, _derivation_for_instance(instance_name, declarations)
        )
        if expected != path.read_text(encoding="utf-8"):
            failures.append(
                f"{path.relative_to(_REPOSITORY_ROOT)} drifted from the contract "
                "derivation; regenerate with "
                "scripts/generate_application_database_table_grants.py --write"
            )
    for failure in failures:
        print(f"::error::{failure}", file=sys.stderr)
    if not failures:
        print(f"grant derivation check: {len(INSTANCE_NAMES)} instance(s) in sync")
    return 1 if failures else 0


def _run_prove(declarations: Sequence[ContractTableDeclaration]) -> int:
    # Imported lazily so --write/--check do not depend on the wiring module's
    # import cost, and so the private resolver import stays localised.
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _resolve_projection_database_target,
    )

    by_node: dict[str, list[ContractTableDeclaration]] = {}
    for declaration in declarations:
        by_node.setdefault(declaration.node, []).append(declaration)

    overall_failures = 0
    for profile in sorted(SUPPORTED_TOPOLOGY_PROFILES):
        topology = load_topology_profile(profile)
        derived = derive_topology_table_grants(topology, declarations)
        residual_keys = {residual.key for residual in derived.unmappable}
        passed = 0
        failed: list[str] = []
        allowed: list[str] = []
        for node, node_declarations in sorted(by_node.items()):
            tables = tuple(item.table for item in node_declarations)
            node_residuals = [
                (table.database_ref, table.schema, table.name)
                for table in tables
                if (table.database_ref, table.schema, table.name) in residual_keys
            ]
            try:
                _resolve_projection_database_target(tables, topology)
            except ValueError as exc:
                if node_residuals:
                    allowed.append(f"{node}: {exc}")
                else:
                    failed.append(f"{node}: {exc}")
            else:
                passed += 1
        overall_failures += len(failed)
        print(
            f"profile={profile:<15} PASS={passed:>3} "
            f"FAIL={len(failed):>3} RESIDUAL={len(allowed):>3}"
        )
        for message in failed:
            print(f"  ::error::{message}", file=sys.stderr)
    if overall_failures:
        print(
            f"::error::{overall_failures} contract/profile resolution failure(s)",
            file=sys.stderr,
        )
    return 1 if overall_failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contracts-root",
        type=Path,
        default=DEFAULT_CONTRACTS_ROOT,
        help="Directory containing node contract.yaml files (cross-repo checkout).",
    )
    parser.add_argument(
        "--write", action="store_true", help="Rewrite instance TABLE grants."
    )
    parser.add_argument(
        "--check", action="store_true", help="Fail on grant/contract drift."
    )
    parser.add_argument(
        "--prove",
        action="store_true",
        help="Resolve every contract against every shipped profile.",
    )
    args = parser.parse_args(argv)
    if not (args.write or args.check or args.prove):
        parser.error("one of --write, --check, or --prove is required")
    if args.write and args.check:
        parser.error("--write and --check are mutually exclusive")

    declarations = (
        load_contract_declarations(args.contracts_root) + STATE_IO_TABLE_DECLARATIONS
    )
    print(
        f"loaded {len(declarations) - len(STATE_IO_TABLE_DECLARATIONS)} "
        f"db_io.db_tables declaration(s) from {args.contracts_root} plus "
        f"{len(STATE_IO_TABLE_DECLARATIONS)} state_io declaration(s)"
    )

    exit_code = 0
    if args.write:
        exit_code |= _run_write(declarations)
    if args.check:
        exit_code |= _run_check(declarations)
    if args.prove:
        exit_code |= _run_prove(declarations)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
