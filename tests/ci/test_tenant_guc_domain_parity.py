# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Boundary tests for the OMN-17422 AC5 ratchet.

A gate is defined as much by what it declines to match as by what it matches,
so both sides are here: the falsifier AC5 names (an internal-declared relation
acquiring a GUC-predicated policy must exit non-zero) and its paired positive
control (a tenant-declared relation carrying the same policy must pass).

Three of these exist because the instrument was wrong during development, not
because the shape was theorised:

* ``test_a_later_migration_that_drops_the_policy_clears_it`` -- a text scan for
  ``CREATE POLICY`` over an append-only corpus reports OMN-18774's remediated
  relations as live defects, because the migration that fixed them cannot edit
  the one that caused them.
* ``test_same_file_drop_then_create_is_still_present`` -- ordering that resolves
  only to file granularity lets a file's own ``DROP POLICY IF EXISTS x;`` beat
  the ``CREATE POLICY x`` two lines below it, which erases every policy in the
  corpus and returns a clean sweep.
* ``test_bridge_parse_raises_when_the_literal_is_gone`` -- a regex read of the
  two bridge frozensets silently undercounted them 6/24 and 39/43, which
  misclassified ``hook_events``. Parsing must raise, never match nothing.

OMN-17887 (operator ruling 2026-09-24) retired the ``tenant`` Postgres schema:
the TENANT domain's schema is ``public``, and the tenant bridge frozenset is
gone. A relation is tenant-domain because the database it is granted in
declares that schema ``domain: TENANT`` in its own ``schemas`` block, so the
synthetic topologies below carry that block, per database.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "validation" / "check_tenant_guc_domain_parity.py"

_GUC_POLICY = (
    "CREATE POLICY tenant_isolation ON public.{rel}\n"
    "  USING (tenant_id = current_setting('app.tenant_id', true));\n"
)
_RLS_ON = "ALTER TABLE public.{rel} ENABLE ROW LEVEL SECURITY;\n"
_RLS_OFF = "ALTER TABLE public.{rel} DISABLE ROW LEVEL SECURITY;\n"
_DROP = "DROP POLICY IF EXISTS tenant_isolation ON public.{rel};\n"


def _load() -> object:
    spec = importlib.util.spec_from_file_location(
        "check_tenant_guc_domain_parity", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate() -> object:
    return _load()


# The application database's real schema -> domain map (OMN-17887): `public`
# IS the TENANT domain's schema.
_APPLICATION_SCHEMAS = {
    "action_authorization_claim": "OMNINODE_INTERNAL",
    "omninode_internal": "OMNINODE_INTERNAL",
    "platform_catalog": "PLATFORM_CATALOG",
    "public": "TENANT",
}


def _database(
    topology: dict[str, list[str]], schemas: dict[str, str] | None
) -> dict[str, object]:
    """One topology database: its TABLE grants and, if given, its `schemas`."""
    database: dict[str, object] = {
        "principals": {
            "p": {
                "grants": [
                    {"object_type": "TABLE", "schema": schema, "objects": list(rels)}
                    for schema, rels in topology.items()
                ]
            }
        }
    }
    if schemas is not None:
        database["schemas"] = {
            name: {"domain": domain, "owner": f"owner_{name}"}
            for name, domain in schemas.items()
        }
    return database


def _instance_yaml(
    topology: dict[str, list[str]],
    schemas: dict[str, str] | None,
    extra_databases: dict[str, tuple[dict[str, list[str]], dict[str, str] | None]]
    | None = None,
) -> str:
    databases = {"application": _database(topology, schemas)}
    for name, (extra_topology, extra_schemas) in (extra_databases or {}).items():
        databases[name] = _database(extra_topology, extra_schemas)
    return yaml.safe_dump({"databases": databases}, sort_keys=True)


def _root(
    tmp_path: Path,
    *,
    migrations: dict[str, str],
    topology: dict[str, list[str]],
    schemas: dict[str, str] | None = None,
    extra_databases: dict[str, tuple[dict[str, list[str]], dict[str, str] | None]]
    | None = None,
    internal_bridge: tuple[str, ...] = (),
) -> Path:
    """Build a throwaway repo root: corpus, topology and the physical map.

    ``topology`` is the ``application`` database's TABLE grants (schema ->
    relations); ``schemas`` is that database's declared schema -> domain map
    (omitted entirely when ``None``, so nothing states a domain);
    ``extra_databases`` adds further databases, each with its own pair.
    """
    forward = tmp_path / "docker" / "migrations" / "forward" / "nodes" / "n"
    forward.mkdir(parents=True)
    for name, sql in migrations.items():
        (forward / name).write_text(sql, encoding="utf-8")

    instances = tmp_path / "src" / "omnibase_infra" / "topology" / "instances"
    instances.mkdir(parents=True)
    (instances / "local.yaml").write_text(
        _instance_yaml(topology, schemas, extra_databases), encoding="utf-8"
    )

    # OMN-17887: the physical map carries only the INTERNAL bridge now.
    mapping = tmp_path / "src" / "omnibase_infra" / "topology"
    (mapping / "physical_schema_mapping.py").write_text(
        "INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359: frozenset[str] = frozenset(\n"
        f"    {list(internal_bridge)!r}\n)\n",
        encoding="utf-8",
    )
    return tmp_path


# ---------------------------------------------------------------------------
# The real tree, and the control proving that result is not vacuous.
# ---------------------------------------------------------------------------


def test_the_checked_in_tree_is_clean(gate: object) -> None:
    assert gate.main(["--root", str(_REPO_ROOT)]) == 0  # type: ignore[attr-defined]


def test_the_clean_result_is_not_vacuous(gate: object) -> None:
    """A zero is not a finding until the instrument is shown to see anything."""
    policies, _ = gate.net_corpus_state(  # type: ignore[attr-defined]
        _REPO_ROOT / "docker" / "migrations" / "forward"
    )
    domains = gate.logical_domains(_REPO_ROOT)  # type: ignore[attr-defined]
    assert policies, "no GUC-predicated policy found at all -- the scan is broken"
    assert "omninode_internal" in domains.values()
    assert "tenant" in domains.values()


# ---------------------------------------------------------------------------
# AC5's falsifier, and its paired positive control.
# ---------------------------------------------------------------------------


def test_internal_declared_relation_with_a_guc_policy_is_a_violation(
    gate: object, tmp_path: Path
) -> None:
    root = _root(
        tmp_path,
        migrations={
            "0001_x.sql": _RLS_ON.format(rel="gen") + _GUC_POLICY.format(rel="gen")
        },
        topology={"omninode_internal": ["gen"]},
    )
    assert gate.main(["--root", str(root)]) == 1  # type: ignore[attr-defined]


def test_tenant_declared_relation_with_the_same_policy_passes(
    gate: object, tmp_path: Path
) -> None:
    """The control without which the test above proves nothing.

    OMN-17887: tenant-declared means ``schema: public`` in a database whose
    ``schemas`` block declares ``public: {domain: TENANT}``. The relation must
    resolve to ``tenant`` -- not merely pass as ``unresolved``, which would
    also exit 0 and prove nothing.
    """
    root = _root(
        tmp_path,
        migrations={
            "0001_x.sql": _RLS_ON.format(rel="d") + _GUC_POLICY.format(rel="d")
        },
        topology={"public": ["d"]},
        schemas=_APPLICATION_SCHEMAS,
    )
    assert gate.main(["--root", str(root)]) == 0  # type: ignore[attr-defined]
    assert gate.logical_domains(root)["d"] == "tenant"  # type: ignore[attr-defined]
    assert gate.violations(root) == ([], [])  # type: ignore[attr-defined]


def test_public_resolves_to_tenant_only_where_its_database_declares_it(
    gate: object, tmp_path: Path
) -> None:
    """The domain map is per database. ``public`` is TENANT in ``application``
    but OMNINODE_INTERNAL in a service-owned database; a relation granted in
    the latter's ``public`` must be ``unresolved``, never ``tenant``."""
    root = _root(
        tmp_path,
        migrations={
            "0001_x.sql": _GUC_POLICY.format(rel="d") + _GUC_POLICY.format(rel="svc")
        },
        topology={"public": ["d"]},
        schemas=_APPLICATION_SCHEMAS,
        extra_databases={
            "omnibase_infra": ({"public": ["svc"]}, {"public": "OMNINODE_INTERNAL"})
        },
    )
    domains = gate.logical_domains(root)  # type: ignore[attr-defined]
    assert domains["d"] == "tenant"
    assert domains["svc"] == "unresolved"
    assert gate.main(["--root", str(root)]) == 0  # type: ignore[attr-defined]
    assert gate.violations(root) == ([], ["svc"])  # type: ignore[attr-defined]


def test_a_retired_tenant_schema_grant_no_longer_resolves_to_tenant(
    gate: object, tmp_path: Path
) -> None:
    """OMN-17887: ``schema: tenant`` is no longer special-cased. A database that
    does not declare it cannot make it TENANT, so it is reported unresolved."""
    root = _root(
        tmp_path,
        migrations={"0001_x.sql": _GUC_POLICY.format(rel="d")},
        topology={"tenant": ["d"]},
        schemas=_APPLICATION_SCHEMAS,
    )
    assert gate.logical_domains(root)["d"] == "unresolved"  # type: ignore[attr-defined]
    assert gate.main(["--root", str(root)]) == 0  # type: ignore[attr-defined]
    assert gate.violations(root) == ([], ["d"])  # type: ignore[attr-defined]


def test_the_internal_bridge_classifies_a_public_declared_relation(
    gate: object, tmp_path: Path
) -> None:
    """`generation_events` reads `schema: public` in the topology; the bridge is
    the only checked-in thing that says it is internal. It must win over the
    database's `public: {domain: TENANT}` declaration (OMN-17887)."""
    root = _root(
        tmp_path,
        migrations={"0001_x.sql": _GUC_POLICY.format(rel="gen")},
        topology={"public": ["gen"]},
        schemas=_APPLICATION_SCHEMAS,
        internal_bridge=("gen",),
    )
    assert gate.logical_domains(root)["gen"] == "omninode_internal"  # type: ignore[attr-defined]
    assert gate.main(["--root", str(root)]) == 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Net end state over an append-only corpus.
# ---------------------------------------------------------------------------


def test_a_later_migration_that_drops_the_policy_clears_it(
    gate: object, tmp_path: Path
) -> None:
    root = _root(
        tmp_path,
        migrations={
            "0001_x.sql": _RLS_ON.format(rel="gen") + _GUC_POLICY.format(rel="gen"),
            "0002_drop.sql": _DROP.format(rel="gen") + _RLS_OFF.format(rel="gen"),
        },
        topology={"omninode_internal": ["gen"]},
    )
    assert gate.main(["--root", str(root)]) == 0  # type: ignore[attr-defined]


def test_same_file_drop_then_create_is_still_present(
    gate: object, tmp_path: Path
) -> None:
    """The atomic restatement shape. File-granular ordering would erase it."""
    root = _root(
        tmp_path,
        migrations={
            "0001_x.sql": _DROP.format(rel="gen")
            + _GUC_POLICY.format(rel="gen")
            + _RLS_ON.format(rel="gen")
        },
        topology={"omninode_internal": ["gen"]},
    )
    assert gate.main(["--root", str(root)]) == 1  # type: ignore[attr-defined]


def test_a_restatement_without_the_guc_retires_the_earlier_policy(
    gate: object, tmp_path: Path
) -> None:
    root = _root(
        tmp_path,
        migrations={
            "0001_x.sql": _GUC_POLICY.format(rel="gen"),
            "0002_y.sql": _DROP.format(rel="gen")
            + "CREATE POLICY tenant_isolation ON public.gen USING (true);\n",
        },
        topology={"omninode_internal": ["gen"]},
    )
    assert gate.main(["--root", str(root)]) == 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# What it declines to match.
# ---------------------------------------------------------------------------


def test_a_relation_in_neither_bridge_is_unresolved_not_a_violation(
    gate: object, tmp_path: Path
) -> None:
    """A `public` relation outside the internal bridge, in a database whose
    `schemas` block states no domain for `public`. Guessing its domain either
    way is how this class keeps recurring, so it is reported, not failed.
    (OMN-17887: with no tenant bridge, only the database's own declaration can
    make `public` TENANT; here nothing states it.)"""
    root = _root(
        tmp_path,
        migrations={"0001_x.sql": _GUC_POLICY.format(rel="hook")},
        topology={"public": ["hook"]},
    )
    assert gate.logical_domains(root)["hook"] == "unresolved"  # type: ignore[attr-defined]
    assert gate.main(["--root", str(root)]) == 0  # type: ignore[attr-defined]
    assert gate.violations(root)[1] == ["hook"]  # type: ignore[attr-defined]


def test_no_force_row_level_security_is_not_enforcement_on(
    gate: object, tmp_path: Path
) -> None:
    root = _root(
        tmp_path,
        migrations={
            "0001_x.sql": "ALTER TABLE public.gen NO FORCE ROW LEVEL SECURITY;\n"
        },
        topology={"omninode_internal": ["gen"]},
    )
    assert gate.main(["--root", str(root)]) == 0  # type: ignore[attr-defined]


def test_a_policy_that_does_not_read_the_guc_is_not_a_violation(
    gate: object, tmp_path: Path
) -> None:
    root = _root(
        tmp_path,
        migrations={
            "0001_x.sql": "CREATE POLICY p ON public.gen USING (owner = current_user);\n"
        },
        topology={"omninode_internal": ["gen"]},
    )
    assert gate.main(["--root", str(root)]) == 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# The instrument must fail loudly rather than match nothing.
# ---------------------------------------------------------------------------


def test_a_policy_quoted_in_a_line_comment_is_not_read_as_code(
    gate: object, tmp_path: Path
) -> None:
    """Found by self-review, not theorised: this returned a violation.

    Migration headers here quote the shape they remove -- 0043 does -- so a
    gate matching raw bytes reds CI on a correct migration.
    """
    root = _root(
        tmp_path,
        migrations={
            "0001_x.sql": "-- the shape it removes was:\n"
            "--   " + _GUC_POLICY.format(rel="gen").replace("\n", "\n--   ")
        },
        topology={"omninode_internal": ["gen"]},
    )
    assert gate.main(["--root", str(root)]) == 0  # type: ignore[attr-defined]


def test_a_policy_quoted_in_a_block_comment_is_not_read_as_code(
    gate: object, tmp_path: Path
) -> None:
    root = _root(
        tmp_path,
        migrations={
            "0001_x.sql": "/* was:\n"
            + _GUC_POLICY.format(rel="gen")
            + "*/\nSELECT 1;\n"
        },
        topology={"omninode_internal": ["gen"]},
    )
    assert gate.main(["--root", str(root)]) == 0  # type: ignore[attr-defined]


def test_two_dashes_inside_a_string_literal_do_not_start_a_comment(
    gate: object, tmp_path: Path
) -> None:
    """The control for the two above: stripping must not eat real statements."""
    root = _root(
        tmp_path,
        migrations={
            "0001_x.sql": "SELECT 'not -- a comment';\n" + _GUC_POLICY.format(rel="gen")
        },
        topology={"omninode_internal": ["gen"]},
    )
    assert gate.main(["--root", str(root)]) == 1  # type: ignore[attr-defined]


def test_instances_disagreeing_on_a_domain_is_reported_not_tie_broken(
    gate: object, tmp_path: Path
) -> None:
    root = _root(
        tmp_path,
        migrations={"0001_x.sql": _GUC_POLICY.format(rel="gen")},
        topology={"omninode_internal": ["gen"]},
    )
    instances = root / "src" / "omnibase_infra" / "topology" / "instances"
    # OMN-17887: onex-dev states `gen` tenant-domain the new way -- `public`
    # in a database declaring `public: {domain: TENANT}`.
    (instances / "onex-dev.yaml").write_text(
        _instance_yaml({"public": ["gen"]}, _APPLICATION_SCHEMAS),
        encoding="utf-8",
    )
    assert gate.main(["--root", str(root)]) == 1  # type: ignore[attr-defined]
    found, _ = gate.violations(root)  # type: ignore[attr-defined]
    assert found and found[0].domain.startswith("conflict:")


def test_rls_on_with_no_resolvable_domain_is_reported_not_dropped(
    gate: object, tmp_path: Path
) -> None:
    """It used to vanish: not a violation, and not in the note either."""
    root = _root(
        tmp_path,
        migrations={"0001_x.sql": _RLS_ON.format(rel="orphan")},
        topology={"public": ["orphan"]},
    )
    assert gate.main(["--root", str(root)]) == 0  # type: ignore[attr-defined]
    assert gate.violations(root)[1] == ["orphan"]  # type: ignore[attr-defined]


def test_bridge_parse_raises_when_the_literal_is_gone(
    gate: object, tmp_path: Path
) -> None:
    stub = tmp_path / "physical_schema_mapping.py"
    stub.write_text("SOMETHING_ELSE = frozenset()\n", encoding="utf-8")
    with pytest.raises(ValueError):
        gate._bridge(stub, gate._INTERNAL_BRIDGE)  # type: ignore[attr-defined]
    # OMN-17887: the tenant bridge is gone, not merely unused.
    assert not hasattr(gate, "_TENANT_BRIDGE")


def test_an_empty_topology_refuses_a_vacuous_pass(gate: object, tmp_path: Path) -> None:
    root = _root(tmp_path, migrations={"0001_x.sql": "SELECT 1;\n"}, topology={})
    with pytest.raises(ValueError):
        gate.logical_domains(root)  # type: ignore[attr-defined]


def test_a_missing_corpus_raises_rather_than_reporting_clean(
    gate: object, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        gate.net_corpus_state(tmp_path / "absent")  # type: ignore[attr-defined]
