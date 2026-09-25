#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Refuse tenant row-level security on a relation whose writer never sets the GUC.

OMN-17422 AC5.

THE DEFECT CLASS
----------------
A policy predicated on ``current_setting('app.tenant_id', ...)`` is satisfiable
only if the writer that reaches the relation SETS that GUC. Exactly one code
path does: ``ProjectionBindingConnections.tenant_transaction`` in
``src/omnibase_infra/runtime/auto_wiring/handler_wiring.py``, reached from the
two ``_execute_upsert`` / ``_execute_query`` sites and nowhere else in the tree.
Both resolve their scope through ``_statement_tenant_scope``, which is fed a
``recorded_scope`` only by ``TenantProjectionTableOperation``.

The sibling operations do not, and say so themselves:

  * ``InternalProjectionTableOperation`` -- "Internal operation that never
    resolves or sets tenant context." It also refuses a ``tenant_id`` key via
    ``_reject_canonical_tenant_field``.
  * ``CatalogProjectionTableOperation`` -- no tenant handling at all.

The kernel picks the operation class from the relation's declared DOMAIN. So a
relation declared ``omninode_internal`` or ``platform_catalog`` that carries a
GUC-predicated policy is in a state no declared writer can satisfy: the
predicate compares against a setting the writer never issues. Such writes
succeed today only where the connection owns the table and FORCE is off, and
fail closed the moment the runtime uses a non-owning, non-BYPASSRLS login --
which is what lab security parity requires (OMN-18256 AC1/AC2).

This is the fourth RLS-shape defect on this family: OMN-17288 (policy dropped
and recreated across a commit boundary), OMN-17298 (RLS on with no policy),
OMN-17315 (policy present, enforcement off), OMN-18774 (``generation_events``
and ``node_service_registry``, both internal-declared, carrying
``tenant_isolation``). Detection that is not enforcement gets ignored
(Operating Rule 5), so this is a gate.

THE RULE
--------
For every relation whose LOGICAL domain resolves to ``omninode_internal`` or
``platform_catalog``, the forward-migration corpus must leave it, IN NET, with
no GUC-predicated policy and with row-level security not enabled.

NET, not textual. The corpus is append-only -- ``check_migration_append_only.py``
(OMN-16705) refuses to edit a landed migration -- so the ``CREATE POLICY`` that
OMN-18774 remediated is still in ``node_projection_delegation/0027`` and
``node_projection_registration/0002`` verbatim, permanently. What retires it is
a LATER migration (``0043`` and ``0007``). A gate that greps for ``CREATE
POLICY`` therefore reports permanently-fixed defects as live. This one replays
create/drop and enable/disable in apply order and judges only the end state.

Ordering is (node lineage, numeric ordinal, filename, BYTE OFFSET within the
file). The byte offset is load-bearing: the canonical atomic restatement is
``DROP POLICY IF EXISTS x; CREATE POLICY x ...`` inside one file, so ordering
that resolves only to file granularity lets the DROP win and erases every
policy in the corpus.

HOW THE LOGICAL DOMAIN IS RECOVERED, AND WHY NOT FROM THE OBVIOUS PLACE
-----------------------------------------------------------------------
Node contracts are the authority, but they live in omnimarket and reach CI only
as ``.proof-dependencies/omnimarket`` at the pin in
``.github/omnimarket-contract-pin.yaml``. Pre-commit has no such checkout, which
is why ``check_topology_grant_delivery.py`` reads the topology instead.

The topology cannot answer this question on its own: its ``schema`` field is the
PHYSICAL schema, applied by ``physical_grant_schema_for_table``. Measured on
``onex-dev.yaml``, ``generation_events`` and ``node_service_registry`` both read
``schema: public`` there while their contracts declare ``omninode_internal`` --
so a gate keyed on the topology's ``schema`` would have missed the exact pair
OMN-18774 was filed for.

The inverse map is checked in: the
``INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359`` frozenset in
``topology/physical_schema_mapping.py``. Since OMN-17887 the TENANT domain's
schema is ``public`` itself -- each database's ``schemas`` block in the topology
states the domain of every schema it declares -- so no tenant bridge exists.
Resolution is therefore, per database:

  topology schema ``omninode_internal`` / ``platform_catalog``  -> that domain
  topology schema in the INTERNAL bridge                        -> internal
  topology schema whose declared domain is TENANT               -> tenant
  anything else                                                 -> UNRESOLVED

The frozensets are read with ``ast``, not imported: every script in
``scripts/validation`` is stdlib-only so that a gate cannot fail open on an
import error, and not regex-parsed so that a change to their shape raises
rather than silently matching nothing.

UNRESOLVED IS REPORTED, NOT FAILED
-----------------------------------
A ``public``-declared relation in neither bridge has no checked-in statement of
its logical domain. ``hook_events`` is the live instance: it declares
``schema: public`` as a PHYSICAL value and its owning contract records in prose
that it is logically TENANT. Failing on it would red the gate over a
classification gap rather than a defect, and guessing either way is how this
class keeps recurring. They are counted and named on every run so the gap stays
visible, and they are not violations.

WHAT IS DELIBERATELY NOT CHECKED
--------------------------------
Whether the predicate admits the writer at RUNTIME -- a live-database property.
The authority for predicate SHAPE is already
``application_database_domain_enforcement``, and policy presence and atomicity
are ``check_migration_rls_policy_atomicity.py``. This gate answers only the
static question those leave open: is a GUC-predicated policy attached to a
relation whose declared writer provably never sets that GUC.

NO ALLOWLIST
------------
Measured on 2026-09-22 against ``dev`` the corpus has zero violations: OMN-18774
drained both via ``0043`` and ``0007``, each doing DROP POLICY + DISABLE ROW
LEVEL SECURITY + DROP COLUMN tenant_id. So this lands clean. A violation that
ever has to be admitted belongs here as a dated, ticket-bound entry to be
drained -- never an allowlist that grows.

USAGE
  python3 scripts/validation/check_tenant_guc_domain_parity.py
  python3 scripts/validation/check_tenant_guc_domain_parity.py --root <dir>

EXIT CODES
  0 -- no internal- or catalog-declared relation carries tenant RLS
  1 -- at least one violation, printed one per line
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

_CORPUS_ROOT = Path("docker/migrations/forward")
_TOPOLOGY_ROOT = Path("src/omnibase_infra/topology/instances")
_PHYSICAL_MAP = Path("src/omnibase_infra/topology/physical_schema_mapping.py")

_INTERNAL_BRIDGE = "INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359"

# Domains whose operation class never issues ``set_config('app.tenant_id', ...)``.
_NO_GUC_DOMAINS = frozenset({"omninode_internal", "platform_catalog"})

_IDENT = r'(?:"(?:[^"]|"")+"|[A-Za-z_][A-Za-z0-9_$]*)'
_QUALIFIED = rf"(?:{_IDENT}\s*\.\s*)?{_IDENT}"

_CREATE_POLICY = re.compile(
    rf"\bCREATE\s+POLICY\s+({_IDENT})\s+ON\s+(?:ONLY\s+)?({_QUALIFIED})(.*?);",
    re.IGNORECASE | re.DOTALL,
)
_DROP_POLICY = re.compile(
    rf"\bDROP\s+POLICY\s+(?:IF\s+EXISTS\s+)?({_IDENT})\s+ON\s+(?:ONLY\s+)?({_QUALIFIED})",
    re.IGNORECASE,
)
_RLS_ENABLE = re.compile(
    rf"\bALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?({_QUALIFIED})\s+"
    r"(?!NO\s+FORCE)(?:ENABLE|FORCE)\s+ROW\s+LEVEL\s+SECURITY\b",
    re.IGNORECASE,
)
_RLS_DISABLE = re.compile(
    rf"\bALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?({_QUALIFIED})\s+"
    r"DISABLE\s+ROW\s+LEVEL\s+SECURITY\b",
    re.IGNORECASE,
)
_TENANT_GUC_READ = re.compile(r"current_setting\s*\(\s*'app\.tenant_id'", re.IGNORECASE)
_DOLLAR_TAG = re.compile(r"\$[A-Za-z_]*\$")


@dataclass(frozen=True, slots=True)
class Violation:
    relation: str
    domain: str
    reason: str
    source: str

    def render(self) -> str:
        return f"{self.relation} ({self.domain}): {self.reason} [{self.source}]"


def _unquote(identifier: str) -> str:
    stripped = identifier.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        return stripped[1:-1].replace('""', '"')
    return stripped


def _relation_name(qualified: str) -> str:
    return _unquote(qualified.split(".")[-1]).lower()


def _executable_sql(text: str) -> str:
    """``text`` with comments blanked out, so prose is never read as code.

    Migration headers in this corpus quote the defective shape they remove --
    ``node_projection_delegation/0043`` does exactly that -- so a gate matching
    raw bytes reports a policy that does not exist and reds CI on a correct
    migration. Replaced with spaces rather than deleted so byte offsets, which
    carry the apply order within a file, are preserved.

    A ``--`` inside a string literal or a dollar-quoted body is not a comment,
    so both are skipped rather than scanned.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if text.startswith("--", i):
            end = text.find("\n", i)
            end = n if end == -1 else end
            out.append(" " * (end - i))
            i = end
        elif text.startswith("/*", i):
            depth, j = 1, i + 2  # Postgres block comments nest
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            out.append(" " * (j - i))
            i = j
        elif ch == "'":
            j = i + 1
            while j < n:
                if text[j] == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            out.append(text[i:j])
            i = j
        elif ch == "$":
            match = _DOLLAR_TAG.match(text, i)
            if match:
                close = text.find(match.group(0), match.end())
                j = n if close == -1 else close + len(match.group(0))
                out.append(text[i:j])
                i = j
            else:
                out.append(ch)
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _file_order(path: Path, corpus_root: Path) -> tuple[int, str, str]:
    relative = path.relative_to(corpus_root)
    if len(relative.parts) == 1:
        return (0, "", relative.name)
    return (1, "/".join(relative.parts[:-1]), relative.name)


def net_corpus_state(
    corpus_root: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, tuple[bool, str]]]:
    """Replay the corpus in apply order; return net GUC policies and net RLS state."""
    if not corpus_root.is_dir():
        raise FileNotFoundError(f"corpus root {corpus_root} does not exist")
    events: list[tuple[tuple[int, str, str, int], str, str, str, str]] = []
    for path in sorted(
        corpus_root.rglob("*.sql"), key=lambda p: _file_order(p, corpus_root)
    ):
        text = _executable_sql(path.read_text(encoding="utf-8", errors="replace"))
        key = _file_order(path, corpus_root)
        name = str(path.relative_to(corpus_root))
        for match in _CREATE_POLICY.finditer(text):
            relation = _relation_name(match.group(2))
            policy = _unquote(match.group(1)).lower()
            # A restatement whose predicate no longer reads the GUC retires the
            # earlier GUC-predicated policy of the same name, so it is a DROP here.
            action = "CREATE" if _TENANT_GUC_READ.search(match.group(3)) else "DROP"
            events.append(((*key, match.start()), action, relation, policy, name))
        for match in _DROP_POLICY.finditer(text):
            events.append(
                (
                    (*key, match.start()),
                    "DROP",
                    _relation_name(match.group(2)),
                    _unquote(match.group(1)).lower(),
                    name,
                )
            )
        for match in _RLS_ENABLE.finditer(text):
            events.append(
                (
                    (*key, match.start()),
                    "RLS_ON",
                    _relation_name(match.group(1)),
                    "",
                    name,
                )
            )
        for match in _RLS_DISABLE.finditer(text):
            events.append(
                (
                    (*key, match.start()),
                    "RLS_OFF",
                    _relation_name(match.group(1)),
                    "",
                    name,
                )
            )

    policies: dict[str, dict[str, str]] = {}
    rls: dict[str, tuple[bool, str]] = {}
    for _, action, relation, policy, source in sorted(events, key=lambda e: e[0]):
        if action == "CREATE":
            policies.setdefault(relation, {})[policy] = source
        elif action == "DROP":
            policies.get(relation, {}).pop(policy, None)
        elif action == "RLS_ON":
            rls[relation] = (True, source)
        elif action == "RLS_OFF":
            rls[relation] = (False, source)
    return ({r: p for r, p in policies.items() if p}, rls)


def _bridge(path: Path, name: str) -> frozenset[str]:
    """Read one ``*_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359`` frozenset with ``ast``.

    Parsed rather than imported (every gate here is stdlib-only, so an import
    error cannot make one fail open) and rather than regexed (a change to the
    literal's shape must raise, not match nothing).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = (
            [node.target]
            if isinstance(node, ast.AnnAssign)
            else getattr(node, "targets", [])
        )
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "frozenset"
            and value.args
        ):
            value = value.args[0]
        members = ast.literal_eval(value)
        return frozenset(str(m).lower() for m in members)
    raise ValueError(f"{name} not found as a module-level literal in {path}")


def _walk_table_blocks(node: object) -> Iterable[tuple[str, str]]:
    """Yield (schema, relation) for every ``object_type: TABLE`` grant block."""
    if isinstance(node, dict):
        schema = node.get("schema")
        objects = node.get("objects")
        if (
            str(node.get("object_type", "")).upper() == "TABLE"
            and isinstance(schema, str)
            and isinstance(objects, list)
        ):
            for relation in objects:
                if isinstance(relation, str):
                    yield (schema.lower(), relation.lower())
        for value in node.values():
            yield from _walk_table_blocks(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_table_blocks(value)


def _walk_database_table_blocks(
    document: object,
) -> Iterable[tuple[dict[str, str], str, str]]:
    """Yield (that database's schema -> domain map, schema, relation) for every
    ``object_type: TABLE`` grant block, database by database.

    The domain map is what the topology itself declares under each database's
    ``schemas`` block, so ``public`` resolves to TENANT only where the topology
    says it is (the ``application`` database) and not in a service-owned
    database whose ``public`` is internal.
    """
    databases = document.get("databases") if isinstance(document, dict) else None
    if not isinstance(databases, dict):
        return
    for database in databases.values():
        if not isinstance(database, dict):
            continue
        schemas = database.get("schemas")
        schema_domains = (
            {
                str(name).lower(): str((spec or {}).get("domain", "")).upper()
                for name, spec in schemas.items()
                if isinstance(spec, dict) or spec is None
            }
            if isinstance(schemas, dict)
            else {}
        )
        for schema, relation in _walk_table_blocks(database):
            yield schema_domains, schema, relation


def logical_domains(repo_root: Path) -> dict[str, str]:
    """Relation -> logical domain, or ``unresolved`` when nothing states it."""
    topology_root = repo_root / _TOPOLOGY_ROOT
    if not topology_root.is_dir():
        raise FileNotFoundError(f"topology root {topology_root} does not exist")
    internal_bridge = _bridge(repo_root / _PHYSICAL_MAP, _INTERNAL_BRIDGE)

    seen: dict[str, set[str]] = {}
    for path in sorted(topology_root.glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for schema_domains, schema, relation in _walk_database_table_blocks(document):
            if schema in _NO_GUC_DOMAINS:
                resolved = schema
            elif relation in internal_bridge:
                resolved = "omninode_internal"
            elif schema_domains.get(schema) == "TENANT":
                resolved = "tenant"
            else:
                resolved = "unresolved"
            seen.setdefault(relation, set()).add(resolved)

    # Two instances disagreeing is a real condition, not a tie to break: a
    # topology regenerated for one instance and not another would otherwise
    # decide the verdict by glob order. Resolve to "conflict" and say so.
    declared: dict[str, str] = {}
    for relation, domains in seen.items():
        stated = domains - {"unresolved"}
        if len(stated) > 1:
            declared[relation] = "conflict:" + ",".join(sorted(stated))
        elif stated:
            declared[relation] = stated.pop()
        else:
            declared[relation] = "unresolved"
    if not declared:
        raise ValueError(
            f"no TABLE grant blocks found under {topology_root}; refusing a vacuous pass"
        )
    return declared


def violations(repo_root: Path) -> tuple[list[Violation], list[str]]:
    """Return (violations, unresolved relations that carry a GUC policy)."""
    policies, rls = net_corpus_state(repo_root / _CORPUS_ROOT)
    domains = logical_domains(repo_root)

    found: list[Violation] = []
    unresolved: list[str] = []
    for relation in sorted(policies):
        domain = domains.get(relation, "unresolved")
        if domain == "unresolved":
            unresolved.append(relation)
            continue
        if domain.startswith("conflict:"):
            found.append(
                Violation(
                    relation,
                    domain,
                    "carries a GUC-predicated policy while topology instances "
                    "disagree on its domain, so no verdict is derivable",
                    sorted(policies[relation].values())[0],
                )
            )
            continue
        if domain not in _NO_GUC_DOMAINS:
            continue
        for policy, source in sorted(policies[relation].items()):
            found.append(
                Violation(
                    relation,
                    domain,
                    f"policy {policy} reads current_setting('app.tenant_id'), "
                    "which its declared writer never sets",
                    source,
                )
            )
    for relation, (enabled, source) in sorted(rls.items()):
        if enabled and domains.get(relation, "unresolved") == "unresolved":
            # RLS on, and nothing checked in says what this relation is. Not a
            # violation -- guessing is how this class recurs -- but it must not
            # vanish from the output and read as clean.
            if relation not in unresolved:
                unresolved.append(relation)
            continue
        if enabled and str(domains.get(relation, "")).startswith("conflict:"):
            found.append(
                Violation(
                    relation,
                    domains[relation],
                    "row-level security enabled while instances disagree on the domain",
                    source,
                )
            )
            continue
        if enabled and domains.get(relation) in _NO_GUC_DOMAINS:
            found.append(
                Violation(
                    relation,
                    domains[relation],
                    "row-level security left enabled on a non-tenant domain",
                    source,
                )
            )
    return found, unresolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    found, unresolved = violations(args.root)
    print(f"tenant GUC / declared-domain parity: {len(found)} violation(s)")
    for violation in found:
        print(f"  VIOLATION  {violation.render()}")
    if unresolved:
        print(
            f"  note: {len(unresolved)} relation(s) carry a GUC-predicated policy with no "
            "checked-in logical domain, so they are out of scope rather than clean: "
            + ", ".join(unresolved)
        )
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
