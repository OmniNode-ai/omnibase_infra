# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A standalone projection writer must connect as its declared principal [OMN-18109].

The live defect, read off the ``.201`` dev lane 2026-09-10T00:56:40Z::

    omnimarket-projection-live-events-writer   411 x  Failed to update watermark:
    omnimarket-projection-registration-writer  191 x    permission denied for
                                                        schema omninode_internal

in the preceding 30 minutes, unbroken since 2026-09-08T23:22:56Z. Nothing newer
fixes it: it is a wiring defect, not staleness.

``docker/docker-compose.dev-lane.yml`` merges ``*dev_lane_analytics_env`` into
every standalone writer. That anchor's only key is ``OMNIDASH_ANALYTICS_DB_URL``,
whose DSN names ``role_omnidash``. ``BaseProjectionRunner`` resolves its DSN
through ``ModelProjectionRuntimeBinding.from_legacy_settings()``, which prefers
exactly that variable and is never handed the topology-resolved binding --
``bind_projection_database_url()`` runs only on the in-process dispatch path, and
by construction the runtime does not dispatch a standalone runner. Live schema
ACLs on that lane, names only::

    omninode_internal : postgres=UC, omninode_runtime=U, jake_ro=U
    public            : ... role_omnidash=UC, app_dashboard=U,
                        omninode_runtime=U, tenant_projection_writer=U ...

``role_omnidash`` holds no grant of any kind inside ``omninode_internal``, which
is the refusal. ``projection_watermarks`` lives there, so EVERY writer's
watermark write is refused; the two above are simply the two receiving traffic.

SATISFIABLE AND UNSATISFIABLE, AND WHY THIS FILE ONLY CLOSES THE FIRST HALF
---------------------------------------------------------------------------
A ``BaseProjectionRunner`` opens ONE connection as ONE principal. The topology
declares a principal per schema domain. For a writer whose declared relations
span two domains, no single DSN is satisfiable -- picking any one trades one
refusal for another. That is OMN-17454, which owns the multi-principal
mechanism, and this file must not paper over it: the split writers are asserted
to BE split rather than quietly left alone.

What is enforced
----------------
1. Every standalone writer's resolved principal is derived from the compose file
   as deployed -- the binding overlay when one is declared, the legacy env
   otherwise -- never assumed.
2. A single-domain writer resolves the principal the topology declares for that
   domain, read from ``omnibase_infra.topology.application_database`` rather than
   restated here.
3. A multi-domain writer is named, with its split, so the boundary is an
   assertion instead of an absence.
4. No writer's DSN carries a literal credential, and every one fails closed on an
   unset password variable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.topology.application_database import (
    _EXPECTED_BINDING_DSN_ENVS,
    _EXPECTED_BINDING_PRINCIPALS,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "docker"
DEV_LANE_OVERLAY = DOCKER_DIR / "docker-compose.dev-lane.yml"

# The schema domains a node contract may declare, mapped to the topology binding
# that owns them. Read from `db_io.db_tables[].schema` in each node's
# `contract.yaml` (omnimarket). A domain absent from this map has no single
# declared binding principal on this lane, which makes any writer touching it
# unsatisfiable by construction -- see `public` on the tenant-credentials writer.
DOMAIN_BINDING: dict[str, str] = {
    "omninode_internal": "omninode_runtime_service",
    "tenant": "tenant_projection",
}

# `BaseProjectionRunner._update_watermark` writes
# `omninode_internal.projection_watermarks` on every projected message, for every
# subclass. It is part of each writer's domain set whether the contract lists it
# or not, and it is the relation the live refusal is on.
RUNNER_BASE_DOMAINS: frozenset[str] = frozenset({"omninode_internal"})

# compose service -> the schema domains its node contract declares.
#
# Frozen here rather than read, because the contracts live in omnimarket and this
# repository has no import of them. Each entry is the `schema:` values of
# `db_io.db_tables` in
# `omnimarket/src/omnimarket/nodes/<node>/contract.yaml`, read 2026-09-10.
# Adding a writer to the compose file without adding it here is a red test.
WRITER_CONTRACT_DOMAINS: dict[str, frozenset[str]] = {
    # node_projection_registration: node_service_registry, projection_watermarks
    "projection-registration-writer": frozenset({"omninode_internal"}),
    # node_projection_tenant_registry: tenant_registry_mirror
    "projection-tenant-registry-writer": frozenset({"omninode_internal"}),
    # node_projection_live_events: live_events
    "projection-live-events-writer": frozenset({"omninode_internal"}),
    # node_projection_delegation: delegation_events, delegation_shadow_comparisons,
    # delegation_judge_verdict_events, delegation_budget_state (tenant) +
    # generation_events, tenant_registry_mirror (omninode_internal)
    "projection-delegation-writer": frozenset({"tenant", "omninode_internal"}),
    # node_projection_savings: savings_estimates (tenant) +
    # tenant_registry_mirror (omninode_internal)
    "projection-savings-writer": frozenset({"tenant", "omninode_internal"}),
    # node_projection_tenant_credentials: tenant_inference_credentials (public) +
    # delegation_routing_tenant_overlay (tenant)
    "projection-tenant-credentials-writer": frozenset({"public", "tenant"}),
}

# The env var the runner reads when no binding overlay is declared
# (`Settings.omnidash_analytics_db_url`, preferred by `from_legacy_settings`).
LEGACY_DSN_ENV = "OMNIDASH_ANALYTICS_DB_URL"
BINDING_OVERLAY_ENV = "OMNIMARKET_PROJECTION_RUNTIME_BINDING_OVERLAY"


class _TolerantLoader(yaml.SafeLoader):
    """compose uses `!override` / `!!merge`, which SafeLoader refuses."""


_TolerantLoader.add_multi_constructor(  # type: ignore[no-untyped-call]
    "",
    lambda loader, suffix, node: (
        loader.construct_mapping(node)
        if isinstance(node, yaml.MappingNode)
        else (
            loader.construct_sequence(node)
            if isinstance(node, yaml.SequenceNode)
            else loader.construct_scalar(node)
        )
    ),
)


def _compose() -> dict[str, Any]:
    # S506: _TolerantLoader subclasses SafeLoader; the only widening is a
    # multi-constructor for compose's `!override` / `!!merge` tags, which
    # constructs plain mappings, sequences and scalars and instantiates no
    # arbitrary object. Same suppression, same reason, as the OMN-18012 gate.
    text = DEV_LANE_OVERLAY.read_text(encoding="utf-8")
    loaded = yaml.load(text, Loader=_TolerantLoader)  # noqa: S506
    assert isinstance(loaded, dict), DEV_LANE_OVERLAY
    return loaded


def _standalone_writers() -> dict[str, dict[str, Any]]:
    """Services whose command runs an omnimarket projection node entrypoint.

    Derived from the compose file, not from a name list, so a writer added
    without an entry in ``WRITER_CONTRACT_DOMAINS`` fails rather than passing
    unexamined.
    """
    found: dict[str, dict[str, Any]] = {}
    for name, service in (_compose().get("services") or {}).items():
        command = service.get("command") or []
        if any("omnimarket.nodes.node_projection" in str(part) for part in command):
            found[name] = service
    return found


def _declared_domains(service_name: str) -> frozenset[str]:
    return WRITER_CONTRACT_DOMAINS[service_name] | RUNNER_BASE_DOMAINS


def _is_satisfiable(service_name: str) -> bool:
    return len(_declared_domains(service_name)) == 1


def _expected_principal(service_name: str) -> str:
    (domain,) = _declared_domains(service_name)
    return str(_EXPECTED_BINDING_PRINCIPALS[DOMAIN_BINDING[domain]])


def _expected_dsn_env(service_name: str) -> str:
    (domain,) = _declared_domains(service_name)
    return str(_EXPECTED_BINDING_DSN_ENVS["local"][DOMAIN_BINDING[domain]])


def _overlay_file_for(service: dict[str, Any], container_path: str) -> Path:
    """Resolve the host file compose mounts at ``container_path``."""
    for mount in service.get("volumes") or []:
        text = str(mount)
        host, _, rest = text.partition(":")
        target = rest.split(":", 1)[0]
        if target == container_path:
            return (DOCKER_DIR / host).resolve()
    raise AssertionError(
        f"{BINDING_OVERLAY_ENV} names {container_path} but no volume mounts it"
    )


def _resolved_dsn_env_name(service: dict[str, Any]) -> str:
    """The env var this service's runner will actually read for its DSN."""
    env = service.get("environment") or {}
    overlay_path = env.get(BINDING_OVERLAY_ENV)
    if not overlay_path:
        return LEGACY_DSN_ENV
    overlay = yaml.safe_load(
        _overlay_file_for(service, str(overlay_path)).read_text(encoding="utf-8")
    )
    ref = str(overlay.get("database_url_secret_ref") or "")
    assert ref.startswith("env:"), (
        "a binding overlay on this lane must name its DSN by env reference, "
        f"never carry a value; got {ref!r}"
    )
    return ref.removeprefix("env:").strip()


def _resolved_principal(service: dict[str, Any]) -> str:
    env = service.get("environment") or {}
    dsn_env = _resolved_dsn_env_name(service)
    dsn = str(env.get(dsn_env) or "")
    assert dsn, f"service resolves {dsn_env} but does not set it"
    match = re.match(r"^postgresql://(?P<user>[^:]+):", dsn)
    assert match is not None, f"unparseable DSN for {dsn_env}"
    return match.group("user")


class TestTheWritersAreFoundAtAll:
    """Positive control: an empty discovery must never read as agreement."""

    def test_every_known_writer_is_declared_in_the_compose_file(self) -> None:
        assert set(_standalone_writers()) == set(WRITER_CONTRACT_DOMAINS)

    def test_there_are_six_of_them(self) -> None:
        assert len(_standalone_writers()) == 6


class TestSingleDomainWritersUseTheirDeclaredPrincipal:
    """AC1/AC2. Red at the parent: all six resolved ``role_omnidash``."""

    @pytest.mark.parametrize(
        "service_name",
        sorted(name for name in WRITER_CONTRACT_DOMAINS if _is_satisfiable(name)),
    )
    def test_resolved_principal_is_the_declared_one(self, service_name: str) -> None:
        service = _standalone_writers()[service_name]
        expected = _expected_principal(service_name)
        actual = _resolved_principal(service)
        assert actual == expected, (
            f"{service_name} connects as {actual!r}; the topology declares "
            f"{expected!r} for its schema domain. Every write it makes into "
            "omninode_internal, the watermark included, is refused."
        )

    @pytest.mark.parametrize(
        "service_name",
        sorted(name for name in WRITER_CONTRACT_DOMAINS if _is_satisfiable(name)),
    )
    def test_it_reads_the_dsn_env_the_topology_binds(self, service_name: str) -> None:
        service = _standalone_writers()[service_name]
        assert _resolved_dsn_env_name(service) == _expected_dsn_env(service_name)


class TestMultiDomainWritersAreDeclaredUnsatisfiable:
    """AC3. The boundary is asserted, not left as an absence."""

    def test_exactly_the_three_known_splits_are_unsatisfiable(self) -> None:
        unsatisfiable = {
            name for name in WRITER_CONTRACT_DOMAINS if not _is_satisfiable(name)
        }
        assert unsatisfiable == {
            "projection-delegation-writer",
            "projection-savings-writer",
            "projection-tenant-credentials-writer",
        }, (
            "the satisfiable/unsatisfiable split changed. A writer that became "
            "single-domain should be wired to its declared principal here; a "
            "writer that became multi-domain belongs to OMN-17454."
        )

    @pytest.mark.parametrize(
        "service_name",
        sorted(name for name in WRITER_CONTRACT_DOMAINS if not _is_satisfiable(name)),
    )
    def test_a_split_writer_spans_more_than_one_declared_domain(
        self, service_name: str
    ) -> None:
        domains = _declared_domains(service_name)
        assert len(domains) > 1, domains

    @pytest.mark.parametrize(
        "service_name",
        sorted(name for name in WRITER_CONTRACT_DOMAINS if not _is_satisfiable(name)),
    )
    def test_a_split_writer_is_left_on_the_legacy_binding(
        self, service_name: str
    ) -> None:
        # Deliberate. Moving one of these to any single principal trades the
        # watermark refusal for a table refusal, which is strictly worse than
        # the state OMN-17454 is open on.
        service = _standalone_writers()[service_name]
        assert _resolved_dsn_env_name(service) == LEGACY_DSN_ENV


class TestNoCredentialMaterialAndFailClosed:
    """AC2's other half: by store reference, never a value."""

    @pytest.mark.parametrize("service_name", sorted(WRITER_CONTRACT_DOMAINS))
    def test_every_dsn_renders_its_password_from_a_fail_closed_variable(
        self, service_name: str
    ) -> None:
        service = _standalone_writers()[service_name]
        env = service.get("environment") or {}
        dsn = str(env.get(_resolved_dsn_env_name(service)) or "")
        password = dsn.split(":", 2)[2].split("@", 1)[0]
        assert password.startswith("${") and ":?" in password, (
            f"{service_name} must render its password from a `${{VAR:?...}}` "
            "reference so an unset variable fails at compose render, never "
            f"from a literal or a `:-` default; got {password[:24]!r}"
        )

    @pytest.mark.parametrize("service_name", sorted(WRITER_CONTRACT_DOMAINS))
    def test_no_writer_introduces_a_new_required_variable(
        self, service_name: str
    ) -> None:
        # Both password variables are already required by
        # docker/docker-compose.infra.yml's own `${VAR:?}` renders, so this
        # change adds no new operator-supplied value on any host.
        service = _standalone_writers()[service_name]
        env = service.get("environment") or {}
        dsn = str(env.get(_resolved_dsn_env_name(service)) or "")
        variables = set(re.findall(r"\$\{([A-Z0-9_]+)", dsn))
        assert variables <= {
            "ROLE_OMNIDASH_PASSWORD",
            "OMNINODE_RUNTIME_PASSWORD",
            "TENANT_PROJECTION_WRITER_PASSWORD",
        }, variables


class TestTheBindingOverlayIsCoherent:
    """The overlay is a second declaration site; it must not drift from the first."""

    @pytest.mark.parametrize(
        "service_name",
        sorted(name for name in WRITER_CONTRACT_DOMAINS if _is_satisfiable(name)),
    )
    def test_the_overlay_file_exists_and_is_mounted_read_only(
        self, service_name: str
    ) -> None:
        service = _standalone_writers()[service_name]
        env = service.get("environment") or {}
        container_path = str(env[BINDING_OVERLAY_ENV])
        host_file = _overlay_file_for(service, container_path)
        assert host_file.is_file(), host_file
        # A bind mount whose host path is missing is silently created by compose
        # as a DIRECTORY, and the runner then fails on a path it cannot read --
        # so the file's existence is the assertion, not its mount string alone.
        mounts = [str(m) for m in (service.get("volumes") or [])]
        assert any(m.endswith(f"{container_path}:ro") for m in mounts), mounts

    @pytest.mark.parametrize(
        "service_name",
        sorted(name for name in WRITER_CONTRACT_DOMAINS if _is_satisfiable(name)),
    )
    def test_the_overlay_group_matches_the_services_own_env(
        self, service_name: str
    ) -> None:
        # The overlay wins over KAFKA_CONSUMER_GROUP, so a disagreement would
        # silently move a writer onto a different group -- the failure the
        # compose file's own comment warns about, where partitions split
        # between a writer and a no-op consumer and half the input is lost
        # while everything looks healthy.
        service = _standalone_writers()[service_name]
        env = service.get("environment") or {}
        overlay = yaml.safe_load(
            _overlay_file_for(service, str(env[BINDING_OVERLAY_ENV])).read_text(
                encoding="utf-8"
            )
        )
        assert overlay["kafka_consumer_group"] == env["KAFKA_CONSUMER_GROUP"]

    @pytest.mark.parametrize(
        "service_name",
        sorted(name for name in WRITER_CONTRACT_DOMAINS if _is_satisfiable(name)),
    )
    def test_the_overlay_carries_no_inline_database_url(
        self, service_name: str
    ) -> None:
        service = _standalone_writers()[service_name]
        env = service.get("environment") or {}
        overlay = yaml.safe_load(
            _overlay_file_for(service, str(env[BINDING_OVERLAY_ENV])).read_text(
                encoding="utf-8"
            )
        )
        assert "database_url" not in overlay, (
            "a committed overlay must name its DSN by env reference only"
        )

    @pytest.mark.parametrize(
        "service_name",
        sorted(name for name in WRITER_CONTRACT_DOMAINS if not _is_satisfiable(name)),
    )
    def test_a_split_writer_declares_no_overlay(self, service_name: str) -> None:
        service = _standalone_writers()[service_name]
        env = service.get("environment") or {}
        assert BINDING_OVERLAY_ENV not in env
