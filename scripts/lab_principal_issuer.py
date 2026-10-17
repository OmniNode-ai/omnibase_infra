#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Issue a developer machine its lab bus principal, with no operator step (OMN-20150).

THE DEFECT THIS CLOSES
    A developer machine reaches the lab dev lane's broker as its own SCRAM
    principal. Until now every one was created by the operator with hand-typed
    ``rpk security user create`` and ``rpk security acl create`` calls: a new
    developer waited on a person, an outside one could not start at all, and the
    grants existed only on the broker (OMN-19799).

WHO MAY ASK
    The service has no TCP listener on any network. It serves only a Unix socket,
    mode 0600, owned by the issuer user, in a named Docker volume under the Docker
    data root. Issuance is accepted only when the peer's kernel-attested uid
    (SO_PEERCRED) equals ``--proxy-uid``: root on the lab host, where tailscaled
    runs ``tailscale serve``. Serve strips client-supplied Tailscale-User-* headers
    and sets ``Tailscale-User-Login`` to the connecting device's owner. It sets no
    user identity for a TAGGED device (servers, CI runners), so those are refused.
    A container on the dev-lane network cannot connect at all. A non-root host
    process cannot open the socket and would be refused by uid even if it could.
    Root and the host docker group remain trusted: they can already read the
    broker superuser password.

WHAT IT GRANTS
    Exactly the grants in ``deploy/lab/developer-principal-grants.yaml``, which
    this module validates before serving (see :func:`load_declaration`). A
    principal is named ``<prefix><login>-<device>``; a login may hold at most
    ``max_devices_per_login`` of them. Asking again for the same device ROTATES
    its password: the password is never stored here, so re-issue is the only way
    to hand one back, and the device asking is the one that will use it.

HOW A PRINCIPAL IS HELD TO THE FILE (OMN-19975)
    ``audit-grants`` lists each principal's live ACLs read-only and compares them with
    the rows this file derives for that principal (:func:`expected_acl_rows`). A derived
    row the broker lacks is MISSING; an ALLOW row the file does not derive is EXTRA.
    Either one exits 1. It never creates or deletes an ACL: revoking an extra is a
    separate, consented act.

WHO HOLDS A PRINCIPAL (OMN-19975)
    With ``--overlay FILE`` (the lab's ``developer.principal`` document, see
    :func:`load_overlay`) the issuer reads the developer machines from that document
    instead of the ledger: ``apply-grants`` and ``audit-grants`` act on its principals,
    and ``serve`` issues only a (login, device) the document declares. The document
    carries no grant; the grants are the ones this file's grant declaration derives.

WHAT IT RECORDS
    One JSON line per issuance in the ledger: when, which login and device, which
    principal, created or rotated, and the digest of the grant file applied. Never
    the password. ``apply-grants`` records grants-applied rows without login or
    device, and never sets or rotates a password.

HOW IT TALKS TO THE BROKER
    Users through the Redpanda Admin API (the password travels in a request body,
    never in an argv), ACLs through ``rpk security acl create`` (no secret in its
    argv; ``rpk`` reads the superuser identity from ``RPK_USER``/``RPK_PASS``).

Usage::

    lab_principal_issuer.py check --grants FILE [--overlay FILE]
    lab_principal_issuer.py serve --grants FILE --ledger FILE --socket PATH --proxy-uid N [--overlay FILE]
    lab_principal_issuer.py apply-grants --grants FILE --ledger FILE [--overlay FILE] [--principal NAME ...]
    lab_principal_issuer.py audit-grants --grants FILE --ledger FILE [--overlay FILE] [--principal NAME ...]
    lab_principal_issuer.py health --socket PATH
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import secrets
import socket
import socketserver
import stat
import struct
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import yaml

LOGIN_HEADER = "Tailscale-User-Login"
MAX_BODY_BYTES = 4096
PASSWORD_BYTES = 32
MECHANISM = "SCRAM-SHA-256"
SECURITY_PROTOCOL = "SASL_PLAINTEXT"
PRINCIPAL_MAX_LEN = 63
MIN_PREFIXED_READ_GROUP_LEN = 16
ALLOWED_OPERATIONS = frozenset({"read", "write", "describe"})
ALLOWED_RESOURCES = frozenset({"topic", "group"})
ALLOWED_PATTERNS = frozenset({"literal", "prefixed"})


class IssuerError(Exception):
    """A refusal with the HTTP status it is answered with."""

    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class DeclarationError(Exception):
    """The grant file asks for something this issuer will not grant."""


@dataclasses.dataclass(frozen=True)
class Grant:
    resource: str
    name: str
    pattern: str
    operations: tuple[str, ...]
    templated: bool
    create_topic: bool

    def for_principal(self, principal: str) -> Grant:
        """Resolve this principal's literal resource name before applying the grant."""
        return dataclasses.replace(
            self,
            name=self.name.replace("{principal}", principal)
            if self.templated
            else self.name,
            templated=False,
        )


@dataclasses.dataclass(frozen=True)
class Declaration:
    lane: str
    principal_prefix: str
    max_devices_per_login: int
    allowed_login_domains: tuple[str, ...]
    grants: tuple[Grant, ...]
    digest: str


def load_declaration(path: Path) -> Declaration:
    """Read and validate the grant file; refuse anything broader than the rules allow."""
    raw = path.read_bytes()
    doc = yaml.safe_load(raw)
    if (
        not isinstance(doc, dict)
        or doc.get("schema") != "developer-principal-grants.v1"
    ):
        raise DeclarationError(f"{path}: schema must be developer-principal-grants.v1")
    lane = doc.get("lane")
    prefix = doc.get("principal_prefix")
    max_devices = doc.get("max_devices_per_login")
    domains = doc.get("allowed_login_domains") or []
    if not isinstance(lane, str) or not lane:
        raise DeclarationError(f"{path}: lane must be a non-empty string")
    if not isinstance(prefix, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,15}", prefix):
        raise DeclarationError(
            f"{path}: principal_prefix must match [a-z][a-z0-9-]{{0,15}}"
        )
    if not isinstance(max_devices, int) or not 1 <= max_devices <= 20:
        raise DeclarationError(
            f"{path}: max_devices_per_login must be an integer from 1 to 20"
        )
    if not isinstance(domains, list) or not all(
        isinstance(d, str) and d for d in domains
    ):
        raise DeclarationError(
            f"{path}: allowed_login_domains must be a list of domains"
        )
    digest = hashlib.sha256(raw)
    if "grants_from" in doc:
        if "grants" in doc:
            raise DeclarationError(
                f"{path}: grants_from and grants are exclusive; a derived grant is never typed here"
            )
        grants_raw, derived = _derived_grants(path, doc["grants_from"], lane)
        digest.update(derived)
    else:
        grants_raw = doc.get("grants")
    if not isinstance(grants_raw, list) or not grants_raw:
        raise DeclarationError(f"{path}: grants must be a non-empty list")
    grants = tuple(_grant(path, i, g) for i, g in enumerate(grants_raw))
    return Declaration(
        lane=lane,
        principal_prefix=prefix,
        max_devices_per_login=max_devices,
        allowed_login_domains=tuple(d.lower() for d in domains),
        grants=grants,
        digest=digest.hexdigest(),
    )


def _derived_grants(
    path: Path, grants_from: object, lane: str
) -> tuple[list[dict[str, object]], bytes]:
    """Read the grants ``node_broker_grant_derive_compute`` derived (OMN-19975).

    ``grants_from`` names a ``ModelBrokerGrantDerivation`` JSON document beside the
    policy file. Its rows are one per operation; they are folded into one grant per
    resource, which ``_grant`` then holds to the same rules as any other. A principal
    owns a templated topic it may write, so the issuer creates that topic for it.
    """
    if not isinstance(grants_from, str) or not grants_from:
        raise DeclarationError(f"{path}: grants_from must be a file name")
    source = path.parent / grants_from
    if source.name != grants_from:
        raise DeclarationError(f"{path}: grants_from must name a file beside it")
    raw = source.read_bytes()
    doc = json.loads(raw)
    if (
        not isinstance(doc, dict)
        or set(doc) != {"principal", "grants"}
        or not isinstance(doc["grants"], list)
    ):
        raise DeclarationError(
            f"{source}: holds exactly principal and grants, as the grant derivation emits"
        )
    folded: dict[tuple[str, str, str], set[str]] = {}
    for index, row in enumerate(doc["grants"]):
        if not isinstance(row, dict) or set(row) != {
            "broker",
            "resource_type",
            "resource",
            "operation",
            "pattern",
        }:
            raise DeclarationError(f"{source}: grants[{index}] is not a derived grant")
        if row["broker"] != lane:
            raise DeclarationError(
                f"{source}: grants[{index}] is for broker {row['broker']}, not lane {lane}"
            )
        key = (
            str(row["resource_type"]).lower(),
            str(row["resource"]),
            str(row["pattern"]).lower(),
        )
        folded.setdefault(key, set()).add(str(row["operation"]).lower())
    grants: list[dict[str, object]] = []
    for (resource, name, pattern), operations in sorted(folded.items()):
        grant: dict[str, object] = {
            "resource": resource,
            "name": name,
            "pattern": pattern,
            "operations": sorted(operations),
        }
        if "{principal}" in name:
            grant["create_topic"] = "write" in operations
        grants.append(grant)
    return grants, raw


def _grant(path: Path, index: int, raw: object) -> Grant:
    where = f"{path}: grants[{index}]"
    if not isinstance(raw, dict):
        raise DeclarationError(f"{where} must be a mapping")
    resource, name, pattern = raw.get("resource"), raw.get("name"), raw.get("pattern")
    operations = raw.get("operations")
    if resource not in ALLOWED_RESOURCES:
        raise DeclarationError(
            f"{where}: resource must be one of {sorted(ALLOWED_RESOURCES)}"
        )
    if not isinstance(name, str) or not name:
        raise DeclarationError(f"{where}: name must be a non-empty string")
    if pattern not in ALLOWED_PATTERNS:
        raise DeclarationError(
            f"{where}: pattern must be one of {sorted(ALLOWED_PATTERNS)}"
        )
    remainder = name.replace("{principal}", "")
    if "{" in remainder or "}" in remainder:
        raise DeclarationError(
            f"{where}: only the '{{principal}}' placeholder is allowed"
        )
    templated = "{principal}" in name
    if templated:
        if name.count("{principal}") != 1:
            raise DeclarationError(f"{where}: '{{principal}}' must appear exactly once")
        if resource != "topic" or pattern != "literal":
            raise DeclarationError(
                f"{where}: a templated grant must be a literal topic"
            )
    create_topic = raw.get("create_topic", False)
    if not isinstance(create_topic, bool):
        raise DeclarationError(f"{where}: create_topic must be a bool")
    if "create_topic" in raw and not templated:
        raise DeclarationError(
            f"{where}: create_topic is allowed only on templated topic grants"
        )
    if (
        not isinstance(operations, list)
        or not operations
        or not set(operations) <= ALLOWED_OPERATIONS
    ):
        raise DeclarationError(
            f"{where}: operations must be a subset of {sorted(ALLOWED_OPERATIONS)}"
        )
    ops = tuple(sorted(set(operations)))
    if name == "*" and ops != ("describe",):
        raise DeclarationError(f"{where}: '*' may carry describe only")
    if pattern == "prefixed" and name == "*":
        raise DeclarationError(f"{where}: '*' is a literal, not a prefix")
    if (
        resource == "group"
        and pattern == "prefixed"
        and "read" in ops
        and len(name) < MIN_PREFIXED_READ_GROUP_LEN
    ):
        raise DeclarationError(
            f"{where}: a prefixed group read grant must be at least "
            f"{MIN_PREFIXED_READ_GROUP_LEN} characters, so it cannot reach a lab runtime's group"
        )
    return Grant(
        resource=resource,
        name=name,
        pattern=pattern,
        operations=ops,
        templated=templated,
        create_topic=create_topic,
    )


def slug(value: str) -> str:
    """Lower-case, [a-z0-9-] only, no leading/trailing/double hyphens."""
    out = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return re.sub(r"-{2,}", "-", out)


def principal_name(declaration: Declaration, login: str, device: str) -> str:
    """``<prefix><login local part>-<device>``, capped at the broker-safe length."""
    who = slug(login.split("@", 1)[0])
    what = slug(device)
    if not who or not what:
        raise IssuerError(
            HTTPStatus.BAD_REQUEST,
            "login and device must each contain a letter or digit",
        )
    name = f"{declaration.principal_prefix}{who}-{what}"
    if len(name) > PRINCIPAL_MAX_LEN:
        digest = hashlib.sha256(name.encode()).hexdigest()[:8]
        name = f"{name[: PRINCIPAL_MAX_LEN - 9].rstrip('-')}-{digest}"
    return name


OVERLAY_SCHEMA_VERSION = "developer_principal.v1"
OVERLAY_GRANTS_SOURCE = "omnibase_infra:deploy/lab/developer-principal-grants.yaml"
_OVERLAY_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")


@dataclasses.dataclass(frozen=True)
class OverlayRow:
    principal: str
    login: str
    device: str


@dataclasses.dataclass(frozen=True)
class Overlay:
    rows: tuple[OverlayRow, ...]

    @property
    def principals(self) -> frozenset[str]:
        return frozenset(row.principal for row in self.rows)

    def declares(self, principal: str, login: str) -> bool:
        return any(
            row.principal == principal and row.login == login for row in self.rows
        )


def load_overlay(path: Path, declaration: Declaration) -> Overlay:
    """Read a ``developer.principal`` overlay document (OMN-19975).

    The document names who holds a principal and never a grant: the document must
    name this issuer's grant file as its ``grants_source``, and each row's
    principal must be the name the issuer derives from the row's login and device.
    Mirrors omnibase_core ``ModelDeveloperPrincipalOverlay``; the issuer image
    carries no omnibase_core, so the unit tests hold the two together.
    """
    doc = yaml.safe_load(path.read_bytes())
    if not isinstance(doc, dict) or set(doc) != {
        "schema_version",
        "grants_source",
        "principals",
    }:
        raise DeclarationError(
            f"{path}: an overlay holds exactly schema_version, grants_source and principals"
        )
    if doc["schema_version"] != OVERLAY_SCHEMA_VERSION:
        raise DeclarationError(
            f"{path}: schema_version must be {OVERLAY_SCHEMA_VERSION}"
        )
    if doc["grants_source"] != OVERLAY_GRANTS_SOURCE:
        raise DeclarationError(
            f"{path}: grants_source must be {OVERLAY_GRANTS_SOURCE}, "
            "the declaration this issuer applies"
        )
    raw_rows = doc["principals"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise DeclarationError(f"{path}: principals must be a non-empty list")
    rows: list[OverlayRow] = []
    for index, raw in enumerate(raw_rows):
        where = f"{path}: principals[{index}]"
        if not isinstance(raw, dict) or set(raw) != {"principal", "login", "device"}:
            raise DeclarationError(
                f"{where} holds exactly principal, login and device; a grant cannot be typed here"
            )
        principal, login, device = raw["principal"], raw["login"], raw["device"]
        if not all(isinstance(v, str) for v in (principal, login, device)):
            raise DeclarationError(f"{where}: principal, login and device are strings")
        if not login.strip():
            raise DeclarationError(f"{where}: login must not be blank")
        if not _OVERLAY_NAME.fullmatch(principal) or not _OVERLAY_NAME.fullmatch(
            device
        ):
            raise DeclarationError(
                f"{where}: principal and device use lower-case letters, digits and hyphens"
            )
        derived = principal_name(declaration, login, device)
        if principal != derived:
            raise DeclarationError(
                f"{where}: principal {principal} is not the name this issuer derives "
                f"from its login and device ({derived})"
            )
        rows.append(OverlayRow(principal, login.strip().lower(), device))
    if len({row.principal for row in rows}) != len(rows):
        raise DeclarationError(f"{path}: principal names must be unique")
    if len({(row.login, row.device) for row in rows}) != len(rows):
        raise DeclarationError(
            f"{path}: each login and device pair may hold one principal"
        )
    for login in {row.login for row in rows}:
        held = sum(1 for row in rows if row.login == login)
        if held > declaration.max_devices_per_login:
            raise DeclarationError(
                f"{path}: {login} holds {held} principals, more than "
                f"{declaration.max_devices_per_login}"
            )
    return Overlay(rows=tuple(rows))


@dataclasses.dataclass(frozen=True, order=True)
class AclRow:
    """One ACL row as the broker lists it, one per operation, lower-cased."""

    resource: str
    name: str
    pattern: str
    operation: str
    host: str = "*"


def expected_acl_rows(declaration: Declaration, principal: str) -> frozenset[AclRow]:
    """The exact rows the grant file derives for one principal."""
    return frozenset(
        AclRow(grant.resource, grant.name, grant.pattern, operation)
        for declared in declaration.grants
        for grant in [declared.for_principal(principal)]
        for operation in grant.operations
    )


ACL_LISTING_HEADER = (
    "PRINCIPAL",
    "HOST",
    "RESOURCE-TYPE",
    "RESOURCE-NAME",
    "RESOURCE-PATTERN-TYPE",
    "OPERATION",
    "PERMISSION",
    "ERROR",
)


def parse_acl_listing(text: str, principal: str) -> frozenset[AclRow]:
    """Read the ALLOW rows of one principal from ``rpk security acl list`` output."""
    if not text.strip():
        return frozenset()
    held: set[AclRow] = set()
    header_found = False
    for line in text.splitlines():
        columns = line.split()
        if tuple(columns) == ACL_LISTING_HEADER:
            header_found = True
            continue
        if not header_found or not columns:
            continue
        if len(columns) < 7:
            raise IssuerError(
                HTTPStatus.BAD_GATEWAY, f"unreadable rpk ACL row: {line!r}"
            )
        who, host, resource, name, pattern, operation, permission = columns[:7]
        if who == f"User:{principal}" and permission == "ALLOW":
            held.add(
                AclRow(resource.lower(), name, pattern.lower(), operation.lower(), host)
            )
    if not header_found:
        raise IssuerError(
            HTTPStatus.BAD_GATEWAY, "rpk ACL list output is missing its header"
        )
    return frozenset(held)


def audit_acl_rows(
    declaration: Declaration, principal: str, live: frozenset[AclRow]
) -> tuple[tuple[AclRow, ...], tuple[AclRow, ...]]:
    """Return (derived rows the broker lacks, live rows the file does not derive)."""
    expected = expected_acl_rows(declaration, principal)
    return tuple(sorted(expected - live)), tuple(sorted(live - expected))


@dataclasses.dataclass(frozen=True)
class BrokerConfig:
    admin_url: str
    superuser: str
    superuser_password: str


class Broker:
    """The broker operations an issuance needs. Replaced by a fake in tests."""

    def __init__(
        self,
        config: BrokerConfig,
        run: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self._config = config
        self._run = run or (
            lambda argv: subprocess.run(
                argv, capture_output=True, text=True, check=False
            )
        )

    def _admin(
        self, method: str, path: str, body: dict[str, str] | None = None
    ) -> object:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(  # noqa: S310 - the admin URL is fixed configuration
            f"{self._config.admin_url}{path}", data=data, method=method
        )
        token = base64.b64encode(
            f"{self._config.superuser}:{self._config.superuser_password}".encode()
        ).decode()
        request.add_header("Authorization", f"Basic {token}")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310 - fixed admin URL
            payload = response.read()
        return json.loads(payload) if payload else None

    def user_exists(self, name: str) -> bool:
        users = self._admin("GET", "/v1/security/users")
        return isinstance(users, list) and name in users

    def set_user(self, name: str, password: str, *, exists: bool) -> None:
        body = {"username": name, "password": password, "algorithm": MECHANISM}
        if exists:
            self._admin("PUT", f"/v1/security/users/{name}", body)
        else:
            self._admin("POST", "/v1/security/users", body)

    def ensure_topic(self, name: str) -> None:
        """Create a writer topic, accepting an existing one on re-issue."""
        result = self._run(
            ["rpk", "topic", "create", name, "--partitions", "1", "--replicas", "1"]
        )
        output = f"{result.stderr}\n{result.stdout}"
        if result.returncode != 0 and "TOPIC_ALREADY_EXISTS" not in output:
            raise IssuerError(
                HTTPStatus.BAD_GATEWAY,
                f"the broker refused topic {name}: {output.strip()[:300]}",
            )

    def grant(self, principal: str, grant: Grant) -> None:
        argv = [
            "rpk",
            "security",
            "acl",
            "create",
            "--allow-principal",
            f"User:{principal}",
        ]
        argv += ["--operation", ",".join(grant.operations)]
        argv += [
            f"--{grant.resource}",
            grant.name,
            "--resource-pattern-type",
            grant.pattern,
        ]
        result = self._run(argv)
        if result.returncode != 0:
            raise IssuerError(
                HTTPStatus.BAD_GATEWAY,
                f"the broker refused grant {grant.resource}:{grant.name}: {result.stderr.strip()[:300]}",
            )

    def list_acls(self, principal: str) -> str:
        """The broker's own listing of one principal's ACLs (read-only)."""
        result = self._run(
            ["rpk", "security", "acl", "list", "--allow-principal", f"User:{principal}"]
        )
        if result.returncode != 0:
            raise IssuerError(
                HTTPStatus.BAD_GATEWAY,
                f"the broker refused the ACL listing for {principal}: {result.stderr.strip()[:300]}",
            )
        return result.stdout


class Ledger:
    """Append-only record of issuances and grant applications. Never holds a password."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def principals_for(self, login: str) -> set[str]:
        if not self._path.exists():
            return set()
        held: set[str] = set()
        for line in self._path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("login") == login:
                held.add(str(row.get("principal")))
        return held

    def append(self, row: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def principals_in_lane(self, lane: str) -> set[str]:
        """Find every principal recorded for this declaration's lane."""
        if not self._path.exists():
            return set()
        held: set[str] = set()
        for line in self._path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(row, dict)
                and row.get("lane") == lane
                and isinstance(row.get("principal"), str)
            ):
                held.add(row["principal"])
        return held


def _apply_grants(broker: Broker, principal: str, grants: tuple[Grant, ...]) -> None:
    for declared in grants:
        grant = declared.for_principal(principal)
        if grant.create_topic:
            broker.ensure_topic(grant.name)
        broker.grant(principal, grant)


class Issuer:
    def __init__(
        self,
        declaration: Declaration,
        broker: Broker,
        ledger: Ledger,
        overlay: Overlay | None = None,
    ) -> None:
        self._declaration = declaration
        self._broker = broker
        self._ledger = ledger
        self._overlay = overlay
        self._lock = threading.Lock()

    def issue(self, login: str | None, body: object) -> dict[str, str]:
        if not login:
            raise IssuerError(
                HTTPStatus.FORBIDDEN,
                "no tailnet user identity on this request: tagged devices, and requests that did "
                "not arrive through the lab host's tailscale serve, are refused",
            )
        login = login.strip().lower()
        domains = self._declaration.allowed_login_domains
        if domains and login.rsplit("@", 1)[-1] not in domains:
            raise IssuerError(
                HTTPStatus.FORBIDDEN,
                f"tailnet login {login} is not in an allowed domain",
            )
        if not isinstance(body, dict):
            raise IssuerError(
                HTTPStatus.BAD_REQUEST, "the request body must be a JSON object"
            )
        lane, device = body.get("lane"), body.get("device")
        if lane != self._declaration.lane:
            raise IssuerError(
                HTTPStatus.BAD_REQUEST,
                f"this issuer serves lane {self._declaration.lane!r}, not {lane!r}",
            )
        if not isinstance(device, str) or not device.strip():
            raise IssuerError(
                HTTPStatus.BAD_REQUEST, "device must be a non-empty string"
            )
        principal = principal_name(self._declaration, login, device)
        if self._overlay is not None and not self._overlay.declares(principal, login):
            raise IssuerError(
                HTTPStatus.FORBIDDEN,
                f"{principal} is not declared for {login} in the developer.principal overlay",
            )

        with self._lock:
            held = self._ledger.principals_for(login)
            if (
                principal not in held
                and len(held) >= self._declaration.max_devices_per_login
            ):
                raise IssuerError(
                    HTTPStatus.CONFLICT,
                    f"{login} already holds {len(held)} machine principals, the most one login may hold; "
                    "retire one on the broker before adding another",
                )
            password = secrets.token_urlsafe(PASSWORD_BYTES)
            try:
                exists = self._broker.user_exists(principal)
                self._broker.set_user(principal, password, exists=exists)
                _apply_grants(self._broker, principal, self._declaration.grants)
            except urllib.error.URLError as exc:
                raise IssuerError(
                    HTTPStatus.BAD_GATEWAY,
                    f"the broker's admin API did not answer: {exc}",
                ) from exc
            self._ledger.append(
                {
                    "at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                    "login": login,
                    "device": device.strip(),
                    "principal": principal,
                    "lane": self._declaration.lane,
                    "action": "rotated" if exists else "created",
                    "grants_sha256": self._declaration.digest,
                }
            )
        return {
            "principal": principal,
            "password": password,
            "lane": self._declaration.lane,
            "mechanism": MECHANISM,
            "security_protocol": SECURITY_PROTOCOL,
            "action": "rotated" if exists else "created",
        }


class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """HTTP on an owner-only Unix socket, with no TCP listener."""

    daemon_threads = True

    def __init__(self, path: str | Path, handler: type[BaseHTTPRequestHandler]) -> None:
        self._socket_path = Path(path)
        super().__init__(str(path), handler)

    def server_bind(self) -> None:
        try:
            mode = self._socket_path.lstat().st_mode
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISSOCK(mode):
                raise SystemExit(
                    f"lab_principal_issuer: refusing non-socket path {self._socket_path}"
                )
            self._socket_path.unlink()
        previous_umask = os.umask(0o177)
        try:
            super().server_bind()
            os.chmod(self._socket_path, 0o600)  # noqa: PTH101 - explicit socket mode enforcement
        finally:
            os.umask(previous_umask)


def peer_uid(sock: socket.socket) -> int | None:
    """Read Linux's kernel-attested peer uid; fail closed elsewhere or on error."""
    if not hasattr(socket, "SO_PEERCRED"):
        return None
    try:
        _, uid, _ = struct.unpack(
            "3i",
            sock.getsockopt(
                socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
            ),
        )
    except OSError:
        return None
    return int(uid)


def make_handler(
    issuer: Issuer,
    proxy_uid: int,
    peer_uid_of: Callable[[socket.socket], int | None] = peer_uid,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "lab-principal-issuer/1"
        _peer_uid: int | None = None

        def handle_one_request(self) -> None:
            self._peer_uid = peer_uid_of(self.connection)
            super().handle_one_request()

        def address_string(self) -> str:
            return "unix"

        def _answer(self, status: HTTPStatus, payload: dict[str, str]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._answer(HTTPStatus.OK, {"status": "ok"})
            else:
                self._answer(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/principals":
                self._answer(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if self._peer_uid is None or self._peer_uid != proxy_uid:
                self._answer(
                    HTTPStatus.FORBIDDEN,
                    {
                        "error": "request did not arrive through the lab host's tailscale serve"
                    },
                )
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self._answer(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "request body too large"},
                )
                return
            try:
                body = json.loads(self.rfile.read(length) or b"null")
            except json.JSONDecodeError:
                self._answer(
                    HTTPStatus.BAD_REQUEST, {"error": "the request body is not JSON"}
                )
                return
            try:
                answer = issuer.issue(self.headers.get(LOGIN_HEADER), body)
            except IssuerError as exc:
                self._answer(exc.status, {"error": str(exc)})
                return
            self._answer(HTTPStatus.CREATED, answer)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - http.server's name
            # Request line and status only; bodies (which carry the password) are never logged.
            identity = f"peer_uid={self._peer_uid}"
            if self._peer_uid == proxy_uid and hasattr(self, "headers"):
                identity = self.headers.get(LOGIN_HEADER, "-")
            sys.stderr.write(
                f"{self.log_date_time_string()} {identity} {format % args}\n"
            )

    return Handler


def health(path: Path) -> int:
    """Check the Unix HTTP endpoint without requiring broker configuration."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(3)
            client.connect(str(path))
            client.sendall(b"GET /healthz HTTP/1.0\r\n\r\n")
            with client.makefile("rb") as response:
                status = response.readline(MAX_BODY_BYTES).split()
        return (
            0
            if len(status) >= 2
            and status[0] in {b"HTTP/1.0", b"HTTP/1.1"}
            and status[1] == b"200"
            else 1
        )
    except OSError:
        return 1


def _broker_from_environment() -> BrokerConfig:
    missing = [
        k for k in ("RPK_USER", "RPK_PASS", "RPK_ADMIN_HOSTS") if not os.environ.get(k)
    ]
    if missing:
        raise SystemExit(
            f"lab_principal_issuer: set {', '.join(missing)} (the lane's superuser identity)"
        )
    admin_host = os.environ["RPK_ADMIN_HOSTS"].split(",", 1)[0].strip()
    return BrokerConfig(
        admin_url=f"http://{admin_host}",
        superuser=os.environ["RPK_USER"],
        superuser_password=os.environ["RPK_PASS"],
    )


def _selected_principals(
    args: argparse.Namespace,
    declaration: Declaration,
    overlay: Overlay | None,
    ledger: Ledger,
) -> list[str]:
    """The overlay's principals when one is given, else those the ledger recorded."""
    if overlay is None:
        return sorted(
            set(args.principal or ledger.principals_in_lane(declaration.lane))
        )
    undeclared = sorted(set(args.principal or ()) - overlay.principals)
    if undeclared:
        raise IssuerError(
            HTTPStatus.BAD_REQUEST,
            f"not declared in the developer.principal overlay: {', '.join(undeclared)}",
        )
    return sorted(set(args.principal or overlay.principals))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="validate the grant file and exit")
    check.add_argument("--grants", type=Path, required=True)
    check.add_argument("--overlay", type=Path)
    serve = sub.add_parser("serve", help="serve issuance requests")
    serve.add_argument("--grants", type=Path, required=True)
    serve.add_argument("--ledger", type=Path, required=True)
    serve.add_argument("--socket", type=Path, required=True)
    serve.add_argument("--proxy-uid", type=int, required=True)
    serve.add_argument("--overlay", type=Path)
    apply = sub.add_parser(
        "apply-grants", help="re-apply grants without rotating passwords"
    )
    apply.add_argument("--grants", type=Path, required=True)
    apply.add_argument("--ledger", type=Path, required=True)
    apply.add_argument("--principal", action="extend", nargs="+")
    apply.add_argument("--overlay", type=Path)
    audit = sub.add_parser(
        "audit-grants",
        help="compare each principal's live ACLs with the rows the grant file derives",
    )
    audit.add_argument("--grants", type=Path, required=True)
    audit.add_argument("--ledger", type=Path, required=True)
    audit.add_argument("--principal", action="extend", nargs="+")
    audit.add_argument("--overlay", type=Path)
    health_parser = sub.add_parser(
        "health", help="check the Unix socket health endpoint"
    )
    health_parser.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "health":
        return health(args.socket)

    try:
        declaration = load_declaration(args.grants)
        overlay = (
            load_overlay(args.overlay, declaration)
            if args.overlay is not None
            else None
        )
    except (OSError, DeclarationError, yaml.YAMLError) as exc:
        print(f"lab_principal_issuer: {exc}", file=sys.stderr)
        return 2
    if args.command == "check":
        print(
            f"{args.grants}: {len(declaration.grants)} grants for lane {declaration.lane}, sha256 {declaration.digest}"
        )
        if overlay is not None:
            print(f"{args.overlay}: {len(overlay.rows)} developer principals")
        return 0

    broker = Broker(_broker_from_environment())
    ledger = Ledger(args.ledger)
    if args.command == "apply-grants":
        try:
            principals = _selected_principals(args, declaration, overlay, ledger)
            for principal in principals:
                if not broker.user_exists(principal):
                    raise IssuerError(
                        HTTPStatus.NOT_FOUND, f"the broker has no principal {principal}"
                    )
            for principal in principals:
                _apply_grants(broker, principal, declaration.grants)
                ledger.append(
                    {
                        "at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                        "principal": principal,
                        "lane": declaration.lane,
                        "action": "grants-applied",
                        "grants_sha256": declaration.digest,
                    }
                )
                print(f"applied {len(declaration.grants)} grants to {principal}")
        except (OSError, IssuerError, urllib.error.URLError) as exc:
            print(f"lab_principal_issuer: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command == "audit-grants":
        try:
            principals = _selected_principals(args, declaration, overlay, ledger)
            for principal in principals:
                if not broker.user_exists(principal):
                    raise IssuerError(
                        HTTPStatus.NOT_FOUND, f"the broker has no principal {principal}"
                    )
            drifted = False
            for principal in principals:
                live = parse_acl_listing(broker.list_acls(principal), principal)
                missing, extra = audit_acl_rows(declaration, principal, live)
                print(
                    f"principal={principal} expected={len(expected_acl_rows(declaration, principal))} "
                    f"live={len(live)} missing={len(missing)} extra={len(extra)}"
                )
                for label, rows in (("MISSING", missing), ("EXTRA", extra)):
                    for row in rows:
                        print(
                            f"{label} {principal} {row.resource} {row.pattern} {row.name} {row.operation}"
                            + ("" if row.host == "*" else f" host={row.host}")
                        )
                drifted = drifted or bool(missing or extra)
        except (OSError, IssuerError, urllib.error.URLError) as exc:
            print(f"lab_principal_issuer: {exc}", file=sys.stderr)
            return 2
        return int(drifted)

    issuer = Issuer(declaration, broker, ledger, overlay)
    with UnixHTTPServer(args.socket, make_handler(issuer, args.proxy_uid)) as server:
        print(
            f"lab_principal_issuer: lane {declaration.lane} on {args.socket}",
            file=sys.stderr,
        )
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
