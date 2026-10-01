#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Render one lab satellite's tenant env file from its declaration (OMN-20207).

``deploy/lab/satellite-tenants.yaml`` declares, per satellite host, the tenant
resources its runtime uses on the .201 dev lane's shared servers.
``docker/docker-compose.lab-tenant.yml`` spells every one of them ``:?``. This
script is the one bridge between the two: it prints the NON-SECRET values as
``KEY=value`` lines for a compose ``--env-file``, and it checks that the host's
operator env file carries every secret the overlay needs without ever printing
one.

Usage::

    render_lab_tenant_env.py --host lab-105 [--declaration FILE]
        [--operator-env-file FILE] [--runtime-image TAG] [--effects-image TAG]

Exit codes: 0 rendered; 2 usage or an invalid declaration; 3 unknown host;
4 a required secret is absent from the operator env file.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

EXIT_OK = 0
EXIT_INVALID = 2
EXIT_UNKNOWN_HOST = 3
EXIT_SECRET_MISSING = 4

SCHEMA = "lab-satellite-tenants.v1"
DEFAULT_DECLARATION = (
    Path(__file__).resolve().parent.parent / "deploy" / "lab" / "satellite-tenants.yaml"
)

# The same token rules the consumers enforce, so a bad row fails here and not
# at the runtime's first boot: onex-api's tenant slug, provision_db_slot.sh's
# slot token, and topic_namespace.py's namespace token.
_SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{1,61}[a-z0-9]$")
_DB_SLOT_RE = re.compile(r"^[a-z][a-z0-9]{0,11}$")
_NAMESPACE_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
# The issued SCRAM login is the principal id (t-...), never the tenant slug.
BUS_LOGIN_VAR = "LAB_TENANT_KAFKA_SASL_USERNAME"
_BUS_LOGIN_RE = re.compile(r"^t-[a-z0-9]+$")
_BOX_ID_RE = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
# The dev lane and the pre-PR slots own these Valkey indexes.
_RESERVED_VALKEY_INDEXES = frozenset({0, 1, 2})
_DEV_LANE_BOX_ID = "omninode-pc"


class DeclarationError(ValueError):
    """The declaration is malformed or unsafe."""


def box_id_for(hostname: str) -> str:
    """The satellite's box id: its short hostname, lower-cased."""
    return hostname.split(".", 1)[0].lower()


def topic_namespace_for(tenant_slug: str) -> str:
    """``tenant-<slug>``: the onex-api tenant ACL prefix without its dot."""
    return f"tenant-{tenant_slug}"


def load_declaration(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise DeclarationError(f"{path}: schema must be {SCHEMA}")
    dep = doc.get("dependency_lane")
    if not isinstance(dep, dict):
        raise DeclarationError("dependency_lane must be a mapping")
    for key in (
        "host",
        "postgres_port",
        "kafka_port",
        "valkey_port",
        "kafka_security_protocol",
        "kafka_sasl_mechanism",
    ):
        if key not in dep:
            raise DeclarationError(f"dependency_lane.{key} is required")
    tenants = doc.get("tenants")
    if not isinstance(tenants, list) or not tenants:
        raise DeclarationError("tenants must be a non-empty list")
    seen: dict[str, set[Any]] = {
        k: set()
        for k in ("host", "tenant_slug", "db_slot", "valkey_db_index", "box_id")
    }
    for row in tenants:
        if not isinstance(row, dict):
            raise DeclarationError("every tenant row must be a mapping")
        for key in ("host", "hostname", "tenant_slug", "db_slot", "valkey_db_index"):
            if key not in row:
                raise DeclarationError(f"tenant row {row!r} lacks {key}")
        slug, slot, index = row["tenant_slug"], row["db_slot"], row["valkey_db_index"]
        box_id = box_id_for(str(row["hostname"]))
        if not _SLUG_RE.match(slug) or "--" in slug:
            raise DeclarationError(f"tenant_slug {slug!r} is not a valid onex-api slug")
        if not _DB_SLOT_RE.match(slot):
            raise DeclarationError(
                f"db_slot {slot!r} is not a provision_db_slot.sh token"
            )
        if not _NAMESPACE_RE.match(topic_namespace_for(slug)):
            raise DeclarationError(
                f"tenant-{slug} is not a valid topic namespace token"
            )
        if not isinstance(index, int) or index in _RESERVED_VALKEY_INDEXES or index < 0:
            raise DeclarationError(
                f"valkey_db_index {index!r} must be an int outside {sorted(_RESERVED_VALKEY_INDEXES)}"
            )
        if not _BOX_ID_RE.match(box_id) or box_id == _DEV_LANE_BOX_ID:
            raise DeclarationError(
                f"box id {box_id!r} must name the satellite, never .201"
            )
        values = {
            "host": row["host"],
            "tenant_slug": slug,
            "db_slot": slot,
            "valkey_db_index": index,
            "box_id": box_id,
        }
        for key, value in values.items():
            if value in seen[key]:
                raise DeclarationError(f"{key} {value!r} is declared twice")
            seen[key].add(value)
    secrets = doc.get("secret_env")
    if not isinstance(secrets, list) or not all(isinstance(s, str) for s in secrets):
        raise DeclarationError("secret_env must be a list of variable names")
    return doc


def render(doc: Mapping[str, Any], host: str) -> list[tuple[str, str]]:
    """The non-secret env pairs for ``host``; KeyError when it is not declared."""
    rows = {row["host"]: row for row in doc["tenants"]}
    row = rows[host]
    dep = doc["dependency_lane"]
    return [
        ("LAB_TENANT_HOST", row["host"]),
        ("LAB_TENANT_SLUG", row["tenant_slug"]),
        ("LAB_TENANT_BOX_ID", box_id_for(row["hostname"])),
        ("LAB_TENANT_TOPIC_NAMESPACE", topic_namespace_for(row["tenant_slug"])),
        ("LAB_TENANT_DB_SLOT", row["db_slot"]),
        ("LAB_TENANT_VALKEY_DB_INDEX", str(row["valkey_db_index"])),
        ("LAB_TENANT_DEPENDENCY_HOST", str(dep["host"])),
        ("LAB_TENANT_POSTGRES_PORT", str(dep["postgres_port"])),
        ("LAB_TENANT_KAFKA_PORT", str(dep["kafka_port"])),
        ("LAB_TENANT_KAFKA_SECURITY_PROTOCOL", str(dep["kafka_security_protocol"])),
        ("LAB_TENANT_KAFKA_SASL_MECHANISM", str(dep["kafka_sasl_mechanism"])),
        ("LAB_TENANT_VALKEY_PORT", str(dep["valkey_port"])),
        # The dogfood file interpolates this in its broker's command and its
        # contract-overlay extension, and compose interpolates a whole file
        # before profiles drop a service. The broker is fenced by the overlay
        # and never starts, so the value only has to satisfy the `:?` guard;
        # the satellite's own box id is the one value that cannot be mistaken
        # for a reachable dependency address.
        ("DOGFOOD_REDPANDA_ADVERTISED_HOST", box_id_for(row["hostname"])),
    ]


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.removeprefix("export ").partition("=")
        value = value.strip().strip("'\"")
        if value:
            values[name.strip()] = value
    return values


def env_file_keys(path: Path) -> set[str]:
    """Names assigned a non-empty value in an env file; values are never kept."""
    return set(_parse_env_file(path))


def bus_login(path: Path) -> str | None:
    """The issued bus login (the tenant's principal id) from the operator env file.

    The login is issued by onex-api at tenant create and is not derivable from
    the declaration, so it is read from the host's secrets file. It is an
    identifier, not a secret; the password beside it is never read here.
    """
    return _parse_env_file(path).get(BUS_LOGIN_VAR)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True)
    parser.add_argument("--declaration", type=Path, default=DEFAULT_DECLARATION)
    parser.add_argument("--operator-env-file", type=Path)
    parser.add_argument("--runtime-image")
    parser.add_argument("--effects-image")
    args = parser.parse_args(argv)
    try:
        doc = load_declaration(args.declaration)
    except (DeclarationError, OSError, yaml.YAMLError) as exc:
        print(f"REFUSED invalid declaration: {exc}", file=sys.stderr)
        return EXIT_INVALID
    try:
        pairs = render(doc, args.host)
    except KeyError:
        declared = sorted(row["host"] for row in doc["tenants"])
        print(
            f"REFUSED host {args.host!r} is not declared; declared: {declared}",
            file=sys.stderr,
        )
        return EXIT_UNKNOWN_HOST
    if args.operator_env_file is not None:
        missing = sorted(set(doc["secret_env"]) - env_file_keys(args.operator_env_file))
        if missing:
            print(f"REFUSED operator env file lacks {missing}", file=sys.stderr)
            return EXIT_SECRET_MISSING
        login = bus_login(args.operator_env_file)
        if login is None or not _BUS_LOGIN_RE.match(login):
            print(
                f"REFUSED {BUS_LOGIN_VAR} in the operator env file must be the "
                "issued principal id (t-...), not a tenant slug",
                file=sys.stderr,
            )
            return EXIT_SECRET_MISSING
        pairs.append((BUS_LOGIN_VAR, login))
    if args.runtime_image:
        pairs.append(("LAB_TENANT_RUNTIME_IMAGE", args.runtime_image))
    if args.effects_image:
        pairs.append(("LAB_TENANT_EFFECTS_IMAGE", args.effects_image))
    for key, value in pairs:
        print(f"{key}={value}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
