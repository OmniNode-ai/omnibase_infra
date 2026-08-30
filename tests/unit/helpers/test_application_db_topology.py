# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Anti-recurrence proof for the topology-derived projection DSN helper [OMN-17152].

OMN-15425 added the ``tenant_projection`` binding to every shipped topology
profile. Nine test modules that hand-listed the binding DSN env vars needed to
construct ``_make_projection_dispatch_callback`` were updated by hand; two were
missed, and the miss reddened every omnibase_infra PR (OMN-17142) because the
gate is a required test split. ``projection_binding_dsn_envs()`` /
``configure_projection_dsns()`` (``tests/helpers/application_db_topology.py``)
exist so the next binding addition needs no per-module edit at all.

This module pins that property directly: it fails the moment
``projection_binding_dsn_envs()`` stops deriving from the shipped topology and
starts hand-listing again, which is exactly the regression class this ticket
closes.
"""

from __future__ import annotations

import os

import pytest

from tests.helpers.application_db_topology import (
    application_topology,
    configure_projection_dsns,
    projection_binding_dsn_envs,
)

pytestmark = pytest.mark.unit


def test_projection_binding_dsn_envs_covers_every_shipped_binding() -> None:
    """Every ``dsn_env`` any shipped topology binding declares is covered.

    ``all_dsn_envs`` below is computed directly from ``application_topology()``,
    independently of the helper's own implementation. If the helper is ever
    rewritten back into a hardcoded literal list -- the exact shape that let
    OMN-15425's ``tenant_projection`` binding go uncovered in two modules --
    this assertion diverges from the independently-computed set and fails.
    """
    topology = application_topology()
    all_dsn_envs = {
        binding.dsn_env
        for database in topology.databases.values()
        for binding in database.bindings.values()
    }

    assert all_dsn_envs, "shipped topology must declare at least one binding"
    assert set(projection_binding_dsn_envs()) == all_dsn_envs


def test_projection_binding_dsn_envs_has_no_duplicates_and_is_sorted() -> None:
    """The returned tuple is a stable, de-duplicated, sorted sequence."""
    envs = projection_binding_dsn_envs()

    assert len(envs) == len(set(envs))
    assert list(envs) == sorted(envs)


def test_configure_projection_dsns_sets_every_declared_binding_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``configure_projection_dsns`` sets ALL declared bindings, not a subset.

    A helper that silently covered only some bindings would reintroduce the
    OMN-17142 failure class one binding at a time. Every name the derivation
    returns must actually be set in the environment after the call.
    """
    for env in projection_binding_dsn_envs():
        monkeypatch.delenv(env, raising=False)

    configure_projection_dsns(monkeypatch, url="postgresql://anti-recurrence-fixture")

    for env in projection_binding_dsn_envs():
        assert os.environ.get(env) == "postgresql://anti-recurrence-fixture"
