# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26444: the tenant projection writer is declared to read and write
``public.savings_priced_models`` in every deployment topology instance.

omnimarket's metering summary refresh refills that table (migration 102 grants
the writer SELECT, INSERT, UPDATE, DELETE on it) before it reads runs; the fill
deletes the rows the pricing manifest no longer names, so DELETE is declared. The
runtime resolver proves a projection binding against the topology's declared
grants, not the database's, so without this declaration the binding raises
"lacks declared read privileges: SELECT on table public.savings_priced_models"
under every profile.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_INSTANCES = (
    Path(__file__).resolve().parents[3] / "src/omnibase_infra/topology/instances"
)
_TABLE = "savings_priced_models"
_REQUIRED = {"SELECT", "INSERT", "UPDATE", "DELETE"}


def _writer_table_privileges(instance: str) -> set[str]:
    doc = yaml.safe_load((_INSTANCES / f"{instance}.yaml").read_text(encoding="utf-8"))
    granted: set[str] = set()
    found_writer = False

    def walk(node: object) -> None:
        nonlocal found_writer
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "tenant_projection_writer" and isinstance(value, dict):
                    found_writer = True
                    for grant in value.get("grants", []):
                        if (
                            grant.get("object_type") == "TABLE"
                            and grant.get("schema") == "public"
                            and _TABLE in grant.get("objects", [])
                        ):
                            granted.update(grant.get("privileges", []))
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(doc)
    assert found_writer, (
        f"{instance}.yaml declares no tenant_projection_writer principal"
    )
    return granted


# Every instance that declares the writer, read from the tree rather than named.
_WRITER_INSTANCES = sorted(
    path.stem
    for path in _INSTANCES.glob("*.yaml")
    if "tenant_projection_writer:" in path.read_text(encoding="utf-8")
)


def test_the_writer_is_declared_in_at_least_three_instances() -> None:
    assert len(_WRITER_INSTANCES) >= 3, _WRITER_INSTANCES


@pytest.mark.parametrize("instance", _WRITER_INSTANCES)
def test_writer_declares_read_write_on_savings_priced_models(instance: str) -> None:
    missing = _REQUIRED - _writer_table_privileges(instance)
    assert not missing, (
        f"{instance}.yaml: tenant_projection_writer lacks {sorted(missing)} "
        f"on public.{_TABLE}"
    )
