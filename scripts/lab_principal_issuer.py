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

WHAT IT RECORDS
    One JSON line per issuance in the ledger: when, which login and device, which
    principal, created or rotated, and the digest of the grant file applied. Never
    the password.

HOW IT TALKS TO THE BROKER
    Users through the Redpanda Admin API (the password travels in a request body,
    never in an argv), ACLs through ``rpk security acl create`` (no secret in its
    argv; ``rpk`` reads the superuser identity from ``RPK_USER``/``RPK_PASS``).

Usage::

    lab_principal_issuer.py check --grants FILE
    lab_principal_issuer.py serve --grants FILE --ledger FILE --socket PATH --proxy-uid N
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
        digest=hashlib.sha256(raw).hexdigest(),
    )


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
    return Grant(resource=resource, name=name, pattern=pattern, operations=ops)


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


@dataclasses.dataclass(frozen=True)
class BrokerConfig:
    admin_url: str
    superuser: str
    superuser_password: str


class Broker:
    """The two broker operations an issuance needs. Replaced by a fake in tests."""

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


class Ledger:
    """Append-only record of issuances. Never holds a password."""

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


class Issuer:
    def __init__(
        self, declaration: Declaration, broker: Broker, ledger: Ledger
    ) -> None:
        self._declaration = declaration
        self._broker = broker
        self._ledger = ledger
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
                for grant in self._declaration.grants:
                    self._broker.grant(principal, grant)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="validate the grant file and exit")
    check.add_argument("--grants", type=Path, required=True)
    serve = sub.add_parser("serve", help="serve issuance requests")
    serve.add_argument("--grants", type=Path, required=True)
    serve.add_argument("--ledger", type=Path, required=True)
    serve.add_argument("--socket", type=Path, required=True)
    serve.add_argument("--proxy-uid", type=int, required=True)
    health_parser = sub.add_parser(
        "health", help="check the Unix socket health endpoint"
    )
    health_parser.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "health":
        return health(args.socket)

    try:
        declaration = load_declaration(args.grants)
    except (OSError, DeclarationError, yaml.YAMLError) as exc:
        print(f"lab_principal_issuer: {exc}", file=sys.stderr)
        return 2
    if args.command == "check":
        print(
            f"{args.grants}: {len(declaration.grants)} grants for lane {declaration.lane}, sha256 {declaration.digest}"
        )
        return 0

    issuer = Issuer(
        declaration, Broker(_broker_from_environment()), Ledger(args.ledger)
    )
    with UnixHTTPServer(args.socket, make_handler(issuer, args.proxy_uid)) as server:
        print(
            f"lab_principal_issuer: lane {declaration.lane} on {args.socket}",
            file=sys.stderr,
        )
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
