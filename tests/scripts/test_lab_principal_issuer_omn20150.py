# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20150: a developer machine's lab bus principal is issued with no operator step.

Covers the issuer's refusals, what it grants, what it records, and the two
declared files the onboarding script reads. The broker is a fake: no network,
no Docker.
"""

from __future__ import annotations

import json
import os
import re
import socket
import stat
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from http import HTTPStatus
from http.client import HTTPResponse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from scripts import lab_principal_issuer as lpi

REPO_ROOT = Path(__file__).resolve().parents[2]
GRANTS = REPO_ROOT / "deploy" / "lab" / "developer-principal-grants.yaml"
FACTS = REPO_ROOT / "deploy" / "lab" / "developer-onboarding.yaml"
LOGIN = "dev.person@omninode.ai"
UNIX_ONLY = pytest.mark.skipif(
    not hasattr(socket, "SO_PEERCRED"), reason="SO_PEERCRED is Linux-only"
)


class FakeBroker(lpi.Broker):
    def __init__(self) -> None:
        self.users: dict[str, str] = {}
        self.acl_argvs: list[list[str]] = []
        self.set_calls: list[tuple[str, bool]] = []

    def user_exists(self, name: str) -> bool:
        return name in self.users

    def set_user(self, name: str, password: str, *, exists: bool) -> None:
        self.set_calls.append((name, exists))
        self.users[name] = password

    def grant(self, principal: str, grant: lpi.Grant) -> None:
        self.acl_argvs.append(
            [principal, grant.resource, grant.name, grant.pattern, *grant.operations]
        )


@pytest.fixture
def declaration() -> lpi.Declaration:
    return lpi.load_declaration(GRANTS)


@pytest.fixture
def broker() -> FakeBroker:
    return FakeBroker()


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "issuances.jsonl"


@pytest.fixture
def issuer(
    declaration: lpi.Declaration, broker: FakeBroker, ledger_path: Path
) -> lpi.Issuer:
    return lpi.Issuer(declaration, broker, lpi.Ledger(ledger_path))


def _grant_file(tmp_path: Path, grants: list[dict[str, object]]) -> Path:
    doc = yaml.safe_load(GRANTS.read_text())
    doc["grants"] = grants
    path = tmp_path / "grants.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


# --- the declaration ------------------------------------------------------


def test_the_checked_in_grant_file_is_valid(declaration: lpi.Declaration) -> None:
    assert declaration.lane == "dev"
    names = {(g.resource, g.name) for g in declaration.grants}
    assert ("topic", "onex.cmd.omnimarket.delegate-skill.v1") in names
    assert all(
        set(g.operations) <= {"read", "write", "describe"} for g in declaration.grants
    )


def test_only_the_delegate_command_and_its_terminals_are_writable(
    declaration: lpi.Declaration,
) -> None:
    # A machine that runs a harness task publishes the terminal event itself, so it
    # holds write on the two delegate-skill terminals and on nothing else: literal
    # names, no prefix or wildcard grant.
    writable = [g for g in declaration.grants if "write" in g.operations]
    assert {g.name for g in writable} == {
        "onex.cmd.omnimarket.delegate-skill.v1",
        "onex.evt.omnimarket.delegate-skill-completed.v1",
        "onex.evt.omnimarket.delegate-skill-failed.v1",
    }
    assert all(g.resource == "topic" and g.pattern == "literal" for g in writable)


@pytest.mark.parametrize(
    ("grant", "refusal"),
    [
        (
            {
                "resource": "topic",
                "name": "*",
                "pattern": "literal",
                "operations": ["read"],
            },
            "describe only",
        ),
        (
            {
                "resource": "group",
                "name": "local.",
                "pattern": "prefixed",
                "operations": ["read"],
            },
            "at least",
        ),
        (
            {
                "resource": "topic",
                "name": "t",
                "pattern": "literal",
                "operations": ["alter"],
            },
            "subset",
        ),
        (
            {
                "resource": "cluster",
                "name": "k",
                "pattern": "literal",
                "operations": ["describe"],
            },
            "resource",
        ),
        (
            {
                "resource": "topic",
                "name": "*",
                "pattern": "prefixed",
                "operations": ["describe"],
            },
            "literal",
        ),
    ],
)
def test_a_grant_broader_than_the_rules_is_refused(
    tmp_path: Path, grant: dict[str, object], refusal: str
) -> None:
    with pytest.raises(lpi.DeclarationError, match=refusal):
        lpi.load_declaration(_grant_file(tmp_path, [grant]))


def test_check_command_validates_the_repo_file(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert lpi.main(["check", "--grants", str(GRANTS)]) == 0
    assert "grants for lane dev" in capsys.readouterr().out


# --- naming ---------------------------------------------------------------


def test_principal_is_named_from_login_and_device(declaration: lpi.Declaration) -> None:
    assert (
        lpi.principal_name(declaration, LOGIN, "Jakes-MacBook Pro")
        == "dev-dev-person-jakes-macbook-pro"
    )


def test_a_long_name_is_capped_and_stays_unique(declaration: lpi.Declaration) -> None:
    a = lpi.principal_name(declaration, LOGIN, "x" * 80 + "a")
    b = lpi.principal_name(declaration, LOGIN, "x" * 80 + "b")
    assert len(a) <= lpi.PRINCIPAL_MAX_LEN and len(b) <= lpi.PRINCIPAL_MAX_LEN
    assert a != b
    assert re.fullmatch(r"[a-z0-9-]+", a)


# --- issuance -------------------------------------------------------------


def test_a_request_without_a_tailnet_user_is_refused(
    issuer: lpi.Issuer, broker: FakeBroker
) -> None:
    with pytest.raises(lpi.IssuerError) as exc:
        issuer.issue(None, {"lane": "dev", "device": "mac"})
    assert exc.value.status is HTTPStatus.FORBIDDEN
    assert broker.users == {}


def test_another_lane_is_refused(issuer: lpi.Issuer) -> None:
    with pytest.raises(lpi.IssuerError) as exc:
        issuer.issue(LOGIN, {"lane": "stability-test", "device": "mac"})
    assert exc.value.status is HTTPStatus.BAD_REQUEST


def test_a_login_outside_the_allowed_domains_is_refused(
    tmp_path: Path, broker: FakeBroker, ledger_path: Path
) -> None:
    doc = yaml.safe_load(GRANTS.read_text())
    doc["allowed_login_domains"] = ["omninode.ai"]
    path = tmp_path / "grants.yaml"
    path.write_text(yaml.safe_dump(doc))
    issuer = lpi.Issuer(lpi.load_declaration(path), broker, lpi.Ledger(ledger_path))
    with pytest.raises(lpi.IssuerError) as exc:
        issuer.issue("someone@example.com", {"lane": "dev", "device": "mac"})
    assert exc.value.status is HTTPStatus.FORBIDDEN
    assert issuer.issue(LOGIN, {"lane": "dev", "device": "mac"})["action"] == "created"


def test_issuance_creates_the_user_applies_every_declared_grant_and_records_it(
    issuer: lpi.Issuer,
    broker: FakeBroker,
    declaration: lpi.Declaration,
    ledger_path: Path,
) -> None:
    answer = issuer.issue(LOGIN, {"lane": "dev", "device": "Jakes-MBP"})
    principal = answer["principal"]
    assert answer["action"] == "created"
    assert answer["mechanism"] == "SCRAM-SHA-256"
    assert broker.users[principal] == answer["password"]
    assert len(answer["password"]) >= 40
    assert [argv[0] for argv in broker.acl_argvs] == [principal] * len(
        declaration.grants
    )
    ledger = ledger_path.read_text()
    row = json.loads(ledger)
    assert (
        row["principal"] == principal
        and row["login"] == LOGIN
        and row["action"] == "created"
    )
    assert row["grants_sha256"] == declaration.digest
    assert answer["password"] not in ledger


def test_asking_again_for_the_same_device_rotates(
    issuer: lpi.Issuer, broker: FakeBroker
) -> None:
    first = issuer.issue(LOGIN, {"lane": "dev", "device": "mac"})
    second = issuer.issue(LOGIN, {"lane": "dev", "device": "mac"})
    assert second["principal"] == first["principal"]
    assert second["action"] == "rotated"
    assert second["password"] != first["password"]
    assert broker.set_calls == [(first["principal"], False), (first["principal"], True)]


def test_a_login_holds_at_most_the_declared_number_of_devices(
    issuer: lpi.Issuer, declaration: lpi.Declaration
) -> None:
    for i in range(declaration.max_devices_per_login):
        issuer.issue(LOGIN, {"lane": "dev", "device": f"mac-{i}"})
    with pytest.raises(lpi.IssuerError) as exc:
        issuer.issue(LOGIN, {"lane": "dev", "device": "one-more"})
    assert exc.value.status is HTTPStatus.CONFLICT
    # An existing device may still rotate at the cap.
    assert (
        issuer.issue(LOGIN, {"lane": "dev", "device": "mac-0"})["action"] == "rotated"
    )


# --- the broker adapter never puts a secret in an argv --------------------


def test_acl_argv_carries_no_secret(declaration: lpi.Declaration) -> None:
    seen: list[Sequence[str]] = []

    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    broker = lpi.Broker(lpi.BrokerConfig("http://admin:9644", "su", "su-pass"), run=run)
    for grant in declaration.grants:
        broker.grant("dev-a-b", grant)
    flat = " ".join(" ".join(argv) for argv in seen)
    assert "su-pass" not in flat
    assert "--allow-principal User:dev-a-b" in flat
    assert "--resource-pattern-type prefixed" in flat


def test_user_password_travels_in_the_admin_api_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[urllib.request.Request] = []

    class Response:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def urlopen(request: urllib.request.Request, timeout: float) -> Response:
        requests.append(request)
        return Response(b"[]" if request.get_method() == "GET" else b"")

    monkeypatch.setattr(lpi.urllib.request, "urlopen", urlopen)
    broker = lpi.Broker(lpi.BrokerConfig("http://admin:9644", "su", "su-pass"))
    assert broker.user_exists("dev-a-b") is False
    broker.set_user("dev-a-b", "new-secret", exists=False)
    create = requests[-1]
    assert create.get_method() == "POST"
    assert "new-secret" not in create.full_url
    assert json.loads(create.data)["password"] == "new-secret"  # type: ignore[arg-type]


# --- HTTP -----------------------------------------------------------------


@pytest.fixture
def socket_path() -> Iterator[str]:
    # macOS tmp_path can exceed AF_UNIX's 104-byte path limit.
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        yield str(Path(directory) / "issuer.sock")


@contextmanager
def _unix_server(issuer: lpi.Issuer, path: str, proxy_uid: int) -> Iterator[str]:
    with lpi.UnixHTTPServer(path, lpi.make_handler(issuer, proxy_uid)) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield path
        finally:
            httpd.shutdown()
            thread.join()


@pytest.fixture
def server(issuer: lpi.Issuer, socket_path: str) -> Iterator[str]:
    with _unix_server(issuer, socket_path, os.getuid()) as path:
        yield path


def _unix_request(
    path: str, method: str, body: bytes, headers: dict[str, str]
) -> tuple[int, dict[str, str], str]:
    headers = {**headers, "Content-Length": str(len(body))}
    request = f"{method} /v1/principals HTTP/1.0\r\n"
    request += "".join(f"{name}: {value}\r\n" for name, value in headers.items())
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(5)
        client.connect(path)
        client.sendall(request.encode() + b"\r\n" + body)
        with HTTPResponse(client) as response:
            response.begin()
            return (
                response.status,
                json.loads(response.read()),
                response.getheader("Cache-Control", ""),
            )


def _post(
    path: str, body: object, login: str | None
) -> tuple[int, dict[str, str], str]:
    headers = {"Content-Type": "application/json"}
    if login:
        headers[lpi.LOGIN_HEADER] = login
    return _unix_request(path, "POST", json.dumps(body).encode(), headers)


@contextmanager
def _proxy(path: str, login: str | None) -> Iterator[str]:
    """Simulate tailscale serve's header stripping and Unix socket forwarding."""

    class ProxyHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            for name in list(self.headers):
                if name.lower().startswith("tailscale-"):
                    del self.headers[name]
            if login is not None:
                self.headers[lpi.LOGIN_HEADER] = login
            status, payload, cache = _unix_request(
                path, "POST", body, dict(self.headers.items())
            )
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", cache)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - http.server's name
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_address[1]}"
        finally:
            httpd.shutdown()
            thread.join()


def _proxy_post(
    url: str, body: object, login: str | None
) -> tuple[int, dict[str, str], str]:
    request = urllib.request.Request(  # noqa: S310 - a loopback test server
        f"{url}/v1/principals", data=json.dumps(body).encode(), method="POST"
    )
    request.add_header("Content-Type", "application/json")
    if login:
        request.add_header(lpi.LOGIN_HEADER, login)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - a loopback test server
            return (
                response.status,
                json.loads(response.read()),
                response.headers.get("Cache-Control", ""),
            )
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read()), exc.headers.get("Cache-Control", "")


@UNIX_ONLY
def test_http_issues_to_a_tailnet_user_and_is_never_cached(server: str) -> None:
    with _proxy(server, LOGIN) as url:
        status, body, cache = _proxy_post(url, {"lane": "dev", "device": "mac"}, LOGIN)
    assert status == HTTPStatus.CREATED
    assert body["principal"].startswith("dev-")
    assert cache == "no-store"


@UNIX_ONLY
def test_http_refuses_a_request_with_no_tailnet_user(server: str) -> None:
    with _proxy(server, None) as url:
        status, body, _ = _proxy_post(url, {"lane": "dev", "device": "mac"}, None)
    assert status == HTTPStatus.FORBIDDEN
    assert "tagged devices" in body["error"]


@UNIX_ONLY
def test_a_forged_login_from_a_peer_that_is_not_the_proxy_is_refused(
    issuer: lpi.Issuer,
    socket_path: str,
    broker: FakeBroker,
    ledger_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    forged = "someone.else@omninode.ai"
    with _unix_server(issuer, socket_path, os.getuid() + 1) as path:
        status, body, _ = _post(path, {"lane": "dev", "device": "mac"}, forged)
    assert status == HTTPStatus.FORBIDDEN
    assert "tailscale serve" in body["error"]
    assert broker.users == {}
    assert not ledger_path.exists()
    log = capsys.readouterr().err
    assert f"peer_uid={os.getuid()}" in log
    assert forged not in log


@UNIX_ONLY
def test_a_proxied_request_is_issued(
    server: str, broker: FakeBroker, declaration: lpi.Declaration, ledger_path: Path
) -> None:
    forged = "someone.else@omninode.ai"
    with _proxy(server, LOGIN) as url:
        status, body, _ = _proxy_post(url, {"lane": "dev", "device": "mac"}, forged)
    assert status == HTTPStatus.CREATED
    assert body["principal"] == lpi.principal_name(declaration, LOGIN, "mac")
    assert lpi.principal_name(declaration, forged, "mac") not in broker.users
    assert set(broker.users) == {body["principal"]}
    assert json.loads(ledger_path.read_text())["login"] == LOGIN


@UNIX_ONLY
def test_a_proxied_request_from_a_tagged_device_is_refused(
    server: str, broker: FakeBroker, ledger_path: Path
) -> None:
    with _proxy(server, None) as url:
        status, body, _ = _proxy_post(
            url, {"lane": "dev", "device": "mac"}, "someone.else@omninode.ai"
        )
    assert status == HTTPStatus.FORBIDDEN
    assert "tagged devices" in body["error"]
    assert broker.users == {}
    assert not ledger_path.exists()


@UNIX_ONLY
def test_the_socket_is_owner_only(server: str) -> None:
    assert stat.S_IMODE(os.stat(server).st_mode) == 0o600  # noqa: PTH116 - check the socket's mode


@UNIX_ONLY
def test_peer_uid_reads_the_kernel_credential() -> None:
    a, b = socket.socketpair()
    with a, b:
        assert lpi.peer_uid(a) == os.getuid()


@UNIX_ONLY
def test_health_command(server: str) -> None:
    assert lpi.main(["health", "--socket", server]) == 0
    assert lpi.main(["health", "--socket", server + ".missing"]) == 1


@UNIX_ONLY
def test_serve_refuses_a_path_that_is_not_a_socket(
    issuer: lpi.Issuer, socket_path: str
) -> None:
    path = Path(socket_path)
    path.write_text("leave this file untouched")
    with pytest.raises(SystemExit, match="non-socket path"):
        lpi.UnixHTTPServer(path, lpi.make_handler(issuer, os.getuid()))
    assert path.read_text() == "leave this file untouched"


# --- the file the onboarding script reads ---------------------------------


def test_onboarding_facts_are_flat_and_complete() -> None:
    """lab-onboarding.sh parses this with sed before Python exists: one key per line."""
    keys: dict[str, str] = {}
    for line in FACTS.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([a-z_]+):[ \t]*(\S.*)", line)
        assert match, f"not a flat key: value line: {line!r}"
        keys[match.group(1)] = match.group(2)
    for required in (
        "schema",
        "tailnet_suffix",
        "lane",
        "principal_issuer_url",
        "lab_model_url",
        "lab_model_name",
    ):
        assert required in keys, required
    assert keys["lane"] == yaml.safe_load(GRANTS.read_text())["lane"]
    assert keys["principal_issuer_url"].startswith("https://")
    assert keys["lab_model_url"].endswith("/v1/chat/completions")
