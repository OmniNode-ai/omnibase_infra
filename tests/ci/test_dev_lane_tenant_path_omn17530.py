# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev-lane tenant-scoped control plane holds its shape (OMN-17530).

Each test here pins a property that, if it drifted, would produce a lane that
still comes up and still reports healthy while proving something weaker than it
claims — which is the failure class rule 24 exists for.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_LANE = REPO_ROOT / "docker" / "docker-compose.dev-lane.yml"
CATALOG = REPO_ROOT / "docker" / "catalog" / "services"
BUNDLES = REPO_ROOT / "docker" / "catalog" / "bundles.yaml"
SMOKE = REPO_ROOT / "scripts" / "smoke" / "smoke_delegation.sh"
CLOUD_RUNNER = REPO_ROOT / "docker" / "migrations" / "cloud" / "run-cloud-migrations.sh"
ENV_RENDER = (
    REPO_ROOT / "scripts" / "runtime_build" / "render_dev_lane_tenant_path_env.sh"
)

TENANT_PATH_SERVICES = ("cloud-migration-files", "cloud-migration", "onex-api")


def _load_dev_lane() -> dict:
    text = DEV_LANE.read_text().replace("!!merge ", "").replace("!override", "")
    return yaml.safe_load(text)


@pytest.mark.unit
def test_tenant_path_services_are_declared_on_the_dev_lane_only() -> None:
    """The plane lives in the overlay only this project loads.

    stability-test, prod and judge each merge docker-compose.infra.yml. A
    service added there is inherited by all of them, and this plane carries
    eleven fail-closed secrets that only the dev lane provisions — an inheriting
    lane would fail at compose render, or (worse, with a ``:-`` default) start an
    API with an empty admin secret.
    """
    dev_lane = _load_dev_lane()
    for name in TENANT_PATH_SERVICES:
        assert name in dev_lane["services"], f"{name} missing from the dev-lane overlay"

    base = yaml.safe_load(
        (REPO_ROOT / "docker" / "docker-compose.infra.yml")
        .read_text()
        .replace("!!merge ", "")
    )
    for name in TENANT_PATH_SERVICES:
        assert name not in base["services"], (
            f"{name} is declared in docker-compose.infra.yml, which every other lane merges"
        )


@pytest.mark.unit
def test_every_tenant_path_service_has_a_catalog_manifest() -> None:
    names = {yaml.safe_load(p.read_text())["name"] for p in CATALOG.glob("*.yaml")}
    for name in TENANT_PATH_SERVICES:
        assert name in names, f"{name} has no docker/catalog/services manifest"


@pytest.mark.unit
def test_tenant_path_bundle_lists_exactly_the_three_services() -> None:
    bundles = yaml.safe_load(BUNDLES.read_text())
    assert "tenant-path" in bundles
    assert bundles["tenant-path"]["services"] == list(TENANT_PATH_SERVICES)
    # Not a member of `runtime`: see the docstring on the first test.
    assert "tenant-path" not in bundles["runtime"]["includes"]


@pytest.mark.unit
def test_onex_api_secrets_fail_closed_at_render() -> None:
    """Every credential-bearing variable uses ``${VAR:?}``, never ``${VAR:-}``.

    ``TENANT_BOOTSTRAP_ADMIN_SECRET`` is the sharpest case: unset is a 503 and
    wrong is a 401, so an unset value does not fail at boot — it fails at the
    first tenant bootstrap, hours later, as a service-unavailable that reads
    like an outage. ``:?`` moves that to compose render.
    """
    env = _load_dev_lane()["services"]["onex-api"]["environment"]
    must_fail_closed = (
        "TENANT_BOOTSTRAP_ADMIN_SECRET",
        "TENANT_TOPICS_ADMIN_SECRET",
        "TENANT_CLIENTS_ADMIN_SECRET",
        "TENANT_OFFBOARD_ADMIN_SECRET",
        "ALPHA_INVITE_ADMIN_SECRET",
        "STRIPE_API_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "VALKEY_PASSWORD",
        "KEYCLOAK_ADMIN_CLIENT_SECRET",
    )
    for name in must_fail_closed:
        assert name in env, f"{name} is not bound on the dev-lane onex-api"
        assert ":?" in str(env[name]), f"{name} must be ${{VAR:?}}, got {env[name]!r}"
        assert ":-" not in str(env[name]), f"{name} must not carry a silent default"


@pytest.mark.unit
def test_onex_api_connects_to_both_databases_as_a_non_superuser() -> None:
    """Neither DSN may name the superuser.

    Postgres exempts a table's owner and any BYPASSRLS role from row-level
    security unconditionally, FORCE included. A lane whose API connects as
    ``postgres`` cannot prove anything about RLS — every reading taken there is
    a false clean, which is what OMN-15363 established for the analytics
    consumers and what this asserts for the control plane too.
    """
    env = _load_dev_lane()["services"]["onex-api"]["environment"]
    cloud = str(env["OMNINODE_CLOUD_DB_URL"])
    analytics = str(env["OMNIDASH_ANALYTICS_DB_URL"])
    assert cloud.startswith("postgresql://role_omninode:"), cloud[:60]
    assert analytics.startswith("postgresql://role_omnidash:"), analytics[:60]
    for dsn in (cloud, analytics):
        assert "://postgres:" not in dsn, (
            f"superuser DSN on the proving lane: {dsn[:60]}"
        )


@pytest.mark.unit
def test_cloud_migration_applies_the_corpus_as_the_owning_login() -> None:
    env = _load_dev_lane()["services"]["cloud-migration"]["environment"]
    assert env["DB_USER"] == "role_omninode"
    assert env["DB_NAME"] == "omninode_cloud"
    assert ":?" in str(env["PGPASSWORD"])


@pytest.mark.unit
def test_allowed_origins_passes_the_real_validator() -> None:
    """``ONEX_API_ALLOWED_ORIGINS`` must be https or localhost.

    ``main.py``'s ``parse_allowed_origins`` refuses any entry that is not
    ``https://`` unless it is localhost/127.0.0.1, and there is no default —
    spelling the http service URL here cost the k3s lab lane fifteen minutes of
    CrashLoopBackOff on 2026-09-08. The lane has no browser client, so the value
    is a placeholder either way; a placeholder that passes the real validator is
    worth more than one that skips it.
    """
    value = str(
        _load_dev_lane()["services"]["onex-api"]["environment"][
            "ONEX_API_ALLOWED_ORIGINS"
        ]
    )
    for entry in value.split(","):
        entry = entry.strip()
        assert entry.startswith("https://") or re.match(
            r"^https?://(localhost|127\.0\.0\.1)", entry
        ), f"{entry!r} would be rejected at import by parse_allowed_origins"


@pytest.mark.unit
def test_the_lane_names_no_live_cloud_host() -> None:
    """No public omninode.ai hostname, and no cloud bridge tenant.

    Tenant identity is deployment-scoped (operator ruling 2026-09-08): the lane
    mints its own tenant ids and never carries a cloud one. Carrying the base's
    ``GATEWAY_BRIDGED_TENANT_SLUGS`` would put this lane's publishes on a live
    tenant's wire topics.
    """
    env = _load_dev_lane()["services"]["onex-api"]["environment"]
    assert env["GATEWAY_BRIDGED_TENANT_SLUGS"] == ""
    for name, value in env.items():
        assert "omninode.ai" not in str(value), (
            f"{name} names a live cloud host: {value!r}"
        )


@pytest.mark.unit
def test_broker_acl_provider_is_the_lane_s_own_broker() -> None:
    """``redpanda``, not ``msk``.

    ``redpanda`` is a real registered provider (``kafka_acl_manager.
    get_acl_provisioner``, supported values redpanda|msk) and the lane's broker
    really is Redpanda, so this is the honest value and not a disable. ``msk``
    would make every lane tenant-create attempt to reach AWS.
    """
    env = _load_dev_lane()["services"]["onex-api"]["environment"]
    assert env["BROKER_ACL_PROVIDER"] == "redpanda"


@pytest.mark.unit
def test_images_are_tag_referenced_and_fail_closed() -> None:
    """No ``:latest`` default, in either direction.

    A ``${VAR:-something:latest}`` default resolves to whatever that tag
    currently points at, which on a host running five lanes is not knowable from
    the manifest.
    """
    services = _load_dev_lane()["services"]
    for name in ("cloud-migration-files", "onex-api"):
        image = services[name]["image"]
        assert ":?" in image, f"{name} image must fail closed, got {image!r}"
        assert "latest" not in image, f"{name} image must not default to a mutable tag"
    assert services["cloud-migration"]["image"].startswith("postgres:")


@pytest.mark.unit
def test_cloud_migration_runner_has_no_lexicographic_fallback() -> None:
    """The order is consumed from the MANIFEST, never re-derived.

    The corpus has four dependency pairs a sort gets wrong from an empty
    database, and the sort dies outright at ``20260130_create_app_users.sql``
    with 12 of 43 applied. A runner that fell back to a sort when the manifest
    was absent would produce a database that looks migrated and is not.
    """
    body = CLOUD_RUNNER.read_text()
    assert "manifest_lib.sh" in body
    assert "manifest_verdict" in body
    assert "manifest_assert_complete" in body
    # No sorting of the corpus anywhere in the runner.
    assert not re.search(r"\bsort\b", body.split("set -euo pipefail", 1)[1]), (
        "the runner must not sort the corpus — MANIFEST order is the specification"
    )
    assert "FATAL: no ${MANIFEST}" in body


@pytest.mark.unit
def test_smoke_is_one_script_with_two_targets() -> None:
    """No copy-paste divergence between the compose and k8s proofs.

    Two scripts diverge the moment one lane grows a seam, and the divergence is
    invisible: both keep passing, and the seam only one of them checks is the
    one that breaks in front of a customer.
    """
    body = SMOKE.read_text()
    assert "--target compose|k8s" in body
    for adapter in ("api_python_env", "db_query", "writer_log"):
        assert body.count(f"{adapter}()") == 2, (
            f"{adapter} must be defined exactly twice — once per target"
        )
    # The seams are written once. Eight of them, outside both adapter branches.
    assert body.count("== SEAM ") == 8


@pytest.mark.unit
def test_smoke_never_puts_the_api_key_in_argv() -> None:
    """``docker exec -e NAME`` and ``kubectl exec -- env NAME=...``.

    The compose adapter forwards by NAME so the value never reaches argv, where
    ``ps`` would show it to every process on a shared lab host.
    """
    body = SMOKE.read_text()
    assert 'args+=(-e "$n")' in body
    assert (
        "ONEX_API_KEY="
        not in body.split("api_python_env()", 1)[1].split("db_query()", 1)[0]
    )


@pytest.mark.unit
def test_env_render_is_idempotent_and_prints_no_values() -> None:
    """A rerun keeps an existing value and reports only a fingerprint.

    Re-running must never change a live lane's admin secret out from under a
    running onex-api — that would be indistinguishable, at the next 401, from
    the secret having been wrong all along.
    """
    body = ENV_RENDER.read_text()
    assert "kept (" in body
    assert "sha256-12" in body
    assert "NOT a credential rotation" in body
