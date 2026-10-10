# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20207 — a satellite's runtime pair as a tenant of the .201 dev lane.

The lab-tenant overlay layers on docker/docker-compose.dogfood.yml and must:

* fence every dependency service of the dogfood file, so a bring-up of this
  project can never start a Postgres, broker, Valkey, migration or fixture;
* reset `depends_on` on the runtime pair (compose refuses to start a service
  whose dependency is profile-disabled);
* take every per-host value through a `:?` reference, never a default, and
  never connect as the dev lane's superuser or its identity;
* publish on the loopback address only, in its declared block;
* stay out of the governed and grant-interlock lane sets.

And deploy/lab/satellite-tenants.yaml must render, through
scripts/render_lab_tenant_env.py, exactly the prefixes the onex-api tenant ACLs
grant, so the runtime's topics and groups land inside its own tenant.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import yaml

from scripts.lane_census_plan import build_plan
from scripts.preflight_lane_deploy_attribution import (
    GOVERNED_LANES,
    GRANT_INTERLOCK_LANES,
)
from scripts.render_lab_tenant_env import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_SECRET_MISSING,
    EXIT_UNKNOWN_HOST,
    DeclarationError,
    load_declaration,
    main,
    render,
)

pytestmark = pytest.mark.ci

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "deploy" / "lane-census" / "lane-manifest.yaml"
DOGFOOD_PATH = ROOT / "docker" / "docker-compose.dogfood.yml"
OVERLAY_PATH = ROOT / "docker" / "docker-compose.lab-tenant.yml"
DECLARATION_PATH = ROOT / "deploy" / "lab" / "satellite-tenants.yaml"

LANE = "lab-tenant"
RUNTIME_SERVICES = {"omninode-runtime", "runtime-effects"}
EXPECTED_PUBLISHED_PORTS = {"44085", "44086"}
GOVERNED_LANE_PORT_LITERALS = ("28085", "28086", "18085", "18086")
# The onex-api tenant ACL prefixes (omninode_infra docker/onex-api
# kafka_acl_manager.py TENANT_TOPIC_PREFIX_TEMPLATE / TENANT_GROUP_PREFIX_TEMPLATE).
TENANT_PREFIX = "tenant-{slug}."


def _top_level_services(path: Path) -> set[str]:
    raw = path.read_text(encoding="utf-8")
    block = raw.split("\nservices:\n", 1)[1]
    block = re.split(r"\n(?:networks|volumes):\n", block, maxsplit=1)[0]
    return set(re.findall(r"^  ([a-z0-9][a-z0-9-]*):", block, re.MULTILINE))


def _manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_every_dogfood_service_but_the_runtime_pair_is_fenced() -> None:
    dogfood = _top_level_services(DOGFOOD_PATH)
    assert dogfood >= RUNTIME_SERVICES, "positive control: the scrape sees the pair"
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    fenced = set(
        re.findall(r"^  ([a-z0-9-]+): \*lab_tenant_fenced$", raw, re.MULTILINE)
    )
    already_disabled = {"keycloak", "infisical"}
    assert fenced == dogfood - RUNTIME_SERVICES - already_disabled, (
        "a dogfood service added later must be fenced here or run as part of the "
        f"pair; unfenced: {sorted(dogfood - RUNTIME_SERVICES - already_disabled - fenced)}"
    )


def test_the_runtime_pair_waits_on_nothing_and_never_builds() -> None:
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    assert raw.count("depends_on: !reset {}") == len(RUNTIME_SERVICES)
    assert "${LAB_TENANT_RUNTIME_IMAGE:?" in raw
    assert "${LAB_TENANT_EFFECTS_IMAGE:?" in raw


def test_no_connection_is_the_superuser_or_the_dev_lane_identity() -> None:
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    assert "POSTGRES_USER:-" not in raw and "${POSTGRES_USER" not in raw
    assert "omninode-pc" not in raw.split("\nname:", 1)[1], (
        "the dev lane's box id may appear in comments only"
    )
    for line in re.findall(r"^\s+[A-Z_]+_DB_URL: .*$", raw, re.MULTILINE):
        assert "postgres:" not in line.split("//", 1)[1].split("@", 1)[0], line
    assert 'KAFKA_SASL_USERNAME: "${LAB_TENANT_KAFKA_SASL_USERNAME:?' in raw


def test_every_published_port_is_loopback_and_in_the_block() -> None:
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    lines = re.findall(r'^\s+-\s+"([^"]*:\d+:\d+)"\s*$', raw, re.MULTILINE)
    assert lines, "positive control: the overlay publishes ports"
    published = set()
    for line in lines:
        host_ip, host_port, _ = line.split(":")
        assert host_ip == "127.0.0.1", line
        published.add(host_port)
        for literal in GOVERNED_LANE_PORT_LITERALS:
            assert literal not in host_port
    assert published == EXPECTED_PUBLISHED_PORTS


def test_lab_tenant_is_neither_governed_nor_grant_interlocked() -> None:
    assert LANE not in GOVERNED_LANES
    assert LANE not in GRANT_INTERLOCK_LANES


def test_manifest_declares_the_pair_on_the_two_satellites() -> None:
    lane = _manifest()["lanes"][LANE]
    assert sorted(lane["hosts"]) == ["lab-101", "lab-105"]
    assert lane["optional"] is True
    assert {s["name"] for s in lane["services"]} == {
        "omninode-lab-tenant-runtime",
        "omninode-lab-tenant-runtime-effects",
    }
    declared_hosts = {
        row["host"] for row in load_declaration(DECLARATION_PATH)["tenants"]
    }
    assert declared_hosts == set(lane["hosts"])


def test_census_reads_an_absent_lab_tenant_as_clean_and_a_stopped_one_as_drift() -> (
    None
):
    manifest = _manifest()
    envelope = {"host": "omnibook", "lane": None, "containers": [], "networks": []}
    plan = build_plan(envelope, manifest)
    assert LANE in plan["lanes_checked"]
    assert not [f for f in plan["findings"] if f["lane"] == LANE]
    running = {
        "Names": "omninode-lab-tenant-runtime-effects",
        "State": "running",
        "Status": "Up 5 minutes",
        "Image": "omnibase-infra-dogfood-runtime-effects",
        "Labels": {},
    }
    stopped = {
        **running,
        "Names": "omninode-lab-tenant-runtime",
        "State": "exited",
        "Status": "Exited (137) 1 minute ago",
    }
    plan = build_plan(
        {
            **envelope,
            "containers": [running, stopped],
            "networks": [manifest["lanes"][LANE]["network"]],
        },
        manifest,
    )
    assert [f for f in plan["findings"] if f["lane"] == LANE], plan["findings"]


@pytest.mark.parametrize("host", ["lab-101", "lab-105"])
def test_rendered_prefixes_are_the_tenant_acl_prefixes(host: str) -> None:
    doc = load_declaration(DECLARATION_PATH)
    env = dict(render(doc, host))
    slug = env["LAB_TENANT_SLUG"]
    assert f"{env['LAB_TENANT_TOPIC_NAMESPACE']}." == TENANT_PREFIX.format(slug=slug)
    assert "LAB_TENANT_KAFKA_SASL_USERNAME" not in env, (
        "the login is issued, not declared"
    )
    assert env["LAB_TENANT_BOX_ID"] != "omninode-pc"
    assert not any("PASSWORD" in key for key in env), "no secret is ever rendered"


def test_declaration_refuses_a_duplicate_or_reserved_value(tmp_path: Path) -> None:
    doc = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
    for mutate in (
        lambda d: d["tenants"][1].update(db_slot=d["tenants"][0]["db_slot"]),
        lambda d: d["tenants"][0].update(valkey_db_index=1),
        lambda d: d["tenants"][0].update(hostname="omninode-pc"),
        lambda d: d["tenants"][0].update(db_slot="lab_h105"),
    ):
        bad = copy.deepcopy(doc)
        mutate(bad)
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(bad), encoding="utf-8")
        with pytest.raises(DeclarationError):
            load_declaration(path)
    load_declaration(DECLARATION_PATH)  # positive control


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--host", "lab-105"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "LAB_TENANT_TOPIC_NAMESPACE=tenant-lab-h105" in out
    assert main(["--host", "lab-999"]) == EXIT_UNKNOWN_HOST
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: nope\n", encoding="utf-8")
    assert main(["--host", "lab-105", "--declaration", str(bad)]) == EXIT_INVALID
    secrets = tmp_path / "operator.env"
    secrets.write_text("LAB_TENANT_KAFKA_SASL_PASSWORD=x\n", encoding="utf-8")
    assert (
        main(["--host", "lab-105", "--operator-env-file", str(secrets)])
        == EXIT_SECRET_MISSING
    )
    names = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))["secret_env"]
    secrets.write_text("".join(f"{n}=x\n" for n in names), encoding="utf-8")
    # the username is the slug, not the issued principal id: refused
    assert (
        main(["--host", "lab-105", "--operator-env-file", str(secrets)])
        == EXIT_SECRET_MISSING
    )
    capsys.readouterr()
    secrets.write_text(
        "".join(
            f"{n}={'t-ecf6bdce01' if n.endswith('SASL_USERNAME') else 'sekret-x'}\n"
            for n in names
        ),
        encoding="utf-8",
    )
    assert main(["--host", "lab-105", "--operator-env-file", str(secrets)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "LAB_TENANT_KAFKA_SASL_USERNAME=t-ecf6bdce01\n" in out
    assert "sekret-x" not in out and "PASSWORD" not in out
