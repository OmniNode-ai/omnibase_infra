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
LOGIN = "dev.person@omninode.ai"
UNIX_ONLY = pytest.mark.skipif(
    not hasattr(socket, "SO_PEERCRED"), reason="SO_PEERCRED is Linux-only"
)


class FakeBroker(lpi.Broker):
    def __init__(self) -> None:
        self.users: dict[str, str] = {}
        self.acl_argvs: list[list[str]] = []
        self.set_calls: list[tuple[str, bool]] = []
        self.topic_calls: list[str] = []

    def user_exists(self, name: str) -> bool:
        return name in self.users

    def set_user(self, name: str, password: str, *, exists: bool) -> None:
        self.set_calls.append((name, exists))
        self.users[name] = password

    def grant(self, principal: str, grant: lpi.Grant) -> None:
        assert not grant.templated
        if grant.create_topic:
            assert grant.name in self.topic_calls
        self.acl_argvs.append(
            [principal, grant.resource, grant.name, grant.pattern, *grant.operations]
        )

    def ensure_topic(self, name: str) -> None:
        self.topic_calls.append(name)


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


@pytest.fixture
def apply_broker(broker: FakeBroker, monkeypatch: pytest.MonkeyPatch) -> FakeBroker:
    monkeypatch.setenv("RPK_USER", "su")
    monkeypatch.setenv("RPK_PASS", "su-pass")
    monkeypatch.setenv("RPK_ADMIN_HOSTS", "admin:9644")

    def configured_broker(config: lpi.BrokerConfig) -> FakeBroker:
        assert config == lpi.BrokerConfig("http://admin:9644", "su", "su-pass")
        return broker

    monkeypatch.setattr(lpi, "Broker", configured_broker)
    return broker


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


def test_only_the_commands_and_the_delegate_terminals_are_writable(
    declaration: lpi.Declaration,
) -> None:
    # A machine that runs a harness task publishes the terminal event itself, so it
    # holds write on the two delegate-skill terminals. The ledger write path
    # (OMN-20275) adds write on its append command only: the ledger host's serve
    # process alone writes the append terminals. Literal names, no prefix or
    # wildcard grant.
    writable = [g for g in declaration.grants if "write" in g.operations]
    assert {g.name for g in writable} == {
        "onex.cmd.omnimarket.delegate-skill.v1",
        "onex.evt.omnimarket.delegate-skill-completed.v1",
        "onex.evt.omnimarket.delegate-skill-failed.v1",
        "onex.cmd.omnimarket.work-ledger-append-from-{principal}.v1",
        "onex.cmd.omnimarket.pr-handoff-requested.v1",
        "onex.cmd.omnimarket.github-repo-gateway-read-requested.v1",
    }
    assert all(g.resource == "topic" and g.pattern == "literal" for g in writable)


def test_scm_read_caller_reads_the_gateway_terminal_and_never_its_serve_group(
    declaration: lpi.Declaration,
) -> None:
    # OMN-21000: `onex scm job-log` (and every gateway read verb) publishes the
    # gateway read command and reads the one completed terminal under its own
    # per-call group. Without these three grants the dev broker refused the
    # terminal subscribe with TopicAuthorizationFailedError.
    grants = {g.name: g for g in declaration.grants}
    command = grants["onex.cmd.omnimarket.github-repo-gateway-read-requested.v1"]
    assert (command.resource, command.pattern) == ("topic", "literal")
    assert set(command.operations) == {"write", "describe"}
    terminal = grants["onex.evt.omnimarket.github-repo-gateway-read-completed.v1"]
    assert (terminal.resource, terminal.pattern) == ("topic", "literal")
    assert set(terminal.operations) == {"read", "describe"}

    prefix = "local.omnimarket.scm_command_client.consume.v1."
    caller = grants[prefix]
    assert (caller.resource, caller.pattern) == ("group", "prefixed")
    assert set(caller.operations) == {"read", "describe"}
    # The bus client consumes as <group>.__t.<topic>; the per-call group adds a
    # 12-hex suffix to the derived caller group.
    assert (
        f"{prefix}0123456789ab.__t.onex.evt.omnimarket."
        "github-repo-gateway-read-completed.v1"
    ).startswith(caller.name)
    for version in ("1.0.0", "1.1.0", "1.2.0"):
        serve = f"local.omnimarket.node_github_repo_gateway_effect.consume.{version}"
        assert not serve.startswith(caller.name)

    # The write node mutates GitHub; a developer principal holds none of it.
    assert not any("github-repo-write" in g.name for g in declaration.grants)


def test_handoff_caller_can_read_replies_without_writing_them(
    declaration: lpi.Declaration,
) -> None:
    grants = {g.name: g for g in declaration.grants}
    command = grants["onex.cmd.omnimarket.pr-handoff-requested.v1"]
    assert command.resource == "topic"
    assert command.pattern == "literal"
    assert set(command.operations) == {"write", "describe"}
    for event in ("accepted", "handed-off", "failed"):
        reply = grants[f"onex.evt.omnimarket.pr-handoff-{event}.v1"]
        assert reply.resource == "topic"
        assert reply.pattern == "literal"
        assert set(reply.operations) == {"read", "describe"}


def test_handoff_caller_group_cannot_join_runtime_groups(
    declaration: lpi.Declaration,
) -> None:
    prefix = "pr-handoff-request."
    caller = next(g for g in declaration.grants if g.name == prefix)
    assert caller.resource == "group"
    assert caller.pattern == "prefixed"
    assert set(caller.operations) == {"read", "describe"}
    assert f"{prefix}0123456789ab".startswith(caller.name)
    for node in ("node_pr_handoff_orchestrator", "node_pr_handoff_ledger_effect"):
        assert not f"local.omnimarket.{node}.consume.v1".startswith(caller.name)


def test_ledger_append_caller_group_cannot_join_the_serve_group(
    declaration: lpi.Declaration,
) -> None:
    # OMN-20275: a developer principal reads the append terminals under its own
    # caller group prefix. The ledger host's serve group must not start with any
    # prefix a developer may read under, or a developer could take its commands.
    serve_group = "local.omnimarket.node_work_ledger_append_effect.consume.v1"
    readable_prefixes = [
        g.name
        for g in declaration.grants
        if g.resource == "group" and g.pattern == "prefixed" and "read" in g.operations
    ]
    assert "local.omnimarket.work_ledger_append_client.consume.v1." in readable_prefixes
    assert not any(serve_group.startswith(prefix) for prefix in readable_prefixes)
    terminals = {
        g.name: set(g.operations)
        for g in declaration.grants
        if g.name.startswith("onex.evt.omnimarket.work-ledger-append-")
    }
    assert terminals == {
        "onex.evt.omnimarket.work-ledger-append-completed.v1": {"read", "describe"},
        "onex.evt.omnimarket.work-ledger-append-failed.v1": {"read", "describe"},
    }


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        ({"pattern": "prefixed"}, "literal topic"),
        ({"resource": "group"}, "literal topic"),
        ({"name": "t-{principal}-{principal}"}, "exactly once"),
        ({"name": "t-{login}"}, "only the"),
        ({"name": "t-{principal}-{login}"}, "only the"),
        ({"name": "t-{principal"}, "only the"),
        ({"name": "t-principal}"}, "only the"),
        ({"name": "t", "create_topic": True}, "only on templated"),
        ({"name": "t", "create_topic": False}, "only on templated"),
        ({"create_topic": "true"}, "must be a bool"),
        ({"create_topic": 1}, "must be a bool"),
    ],
)
def test_invalid_writer_topic_templates_are_refused(
    tmp_path: Path, overrides: dict[str, object], refusal: str
) -> None:
    grant = {
        "resource": "topic",
        "name": "t-{principal}",
        "pattern": "literal",
        "operations": ["write", "describe"],
        **overrides,
    }
    with pytest.raises(lpi.DeclarationError, match=refusal):
        lpi.load_declaration(_grant_file(tmp_path, [grant]))


def test_a_template_without_topic_creation_defaults_to_false(tmp_path: Path) -> None:
    grant = lpi.load_declaration(
        _grant_file(
            tmp_path,
            [
                {
                    "resource": "topic",
                    "name": "t-{principal}",
                    "pattern": "literal",
                    "operations": ["write"],
                }
            ],
        )
    ).grants[0]
    assert grant.templated and not grant.create_topic
    resolved = grant.for_principal("dev-a-b")
    assert resolved.name == "t-dev-a-b"
    assert not resolved.templated and not resolved.create_topic


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
    path = _grant_file(tmp_path, [grant])
    with pytest.raises(lpi.DeclarationError, match=refusal):
        lpi.load_declaration(path)


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
    assert broker.topic_calls == [
        f"onex.cmd.omnimarket.work-ledger-append-from-{principal}.v1"
    ]
    assert broker.acl_argvs == [
        [principal, g.resource, g.name, g.pattern, *g.operations]
        for declared in declaration.grants
        for g in [declared.for_principal(principal)]
    ]
    ledger = ledger_path.read_text()
    row = json.loads(ledger)
    assert (
        row["principal"] == principal
        and row["login"] == LOGIN
        and row["action"] == "created"
    )
    assert row["grants_sha256"] == declaration.digest
    assert answer["password"] not in ledger


def test_each_principal_can_write_only_its_own_ledger_topic(
    issuer: lpi.Issuer,
    broker: FakeBroker,
) -> None:
    a = issuer.issue(LOGIN, {"lane": "dev", "device": "mac-a"})["principal"]
    b = issuer.issue(LOGIN, {"lane": "dev", "device": "mac-b"})["principal"]
    for principal, other in ((a, b), (b, a)):
        applied = [argv for argv in broker.acl_argvs if argv[0] == principal]
        own_topic = f"onex.cmd.omnimarket.work-ledger-append-from-{principal}.v1"
        other_topic = f"onex.cmd.omnimarket.work-ledger-append-from-{other}.v1"
        assert not any(argv[2] == other_topic for argv in applied)
        writer_grants = [
            argv for argv in applied if argv[1] == "topic" and "write" in argv[4:]
        ]
        assert own_topic in {argv[2] for argv in writer_grants}
        for refused in (
            other_topic,
            "onex.cmd.omnimarket.work-ledger-append-requested.v1",
        ):
            assert not any(
                argv[2] in {"*", refused}
                or (argv[3] == "prefixed" and refused.startswith(argv[2]))
                for argv in writer_grants
            )


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
    for declared in declaration.grants:
        grant = declared.for_principal("dev-a-b")
        if grant.create_topic:
            broker.ensure_topic(grant.name)
        broker.grant("dev-a-b", grant)
    flat = " ".join(" ".join(argv) for argv in seen)
    assert "su-pass" not in flat
    assert "--allow-principal User:dev-a-b" in flat
    assert "--resource-pattern-type prefixed" in flat
    assert [
        "rpk",
        "topic",
        "create",
        "onex.cmd.omnimarket.work-ledger-append-from-dev-a-b.v1",
        "--partitions",
        "1",
        "--replicas",
        "1",
    ] in seen


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [("TOPIC_ALREADY_EXISTS", ""), ("", "TOPIC_ALREADY_EXISTS")],
)
def test_ensuring_an_existing_topic_is_success(stdout: str, stderr: str) -> None:
    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, stdout, stderr)

    broker = lpi.Broker(lpi.BrokerConfig("http://admin:9644", "su", "su-pass"), run=run)
    broker.ensure_topic("t")


@pytest.mark.parametrize("operation", ["topic", "grant"])
def test_other_rpk_failures_are_bad_gateway(
    declaration: lpi.Declaration,
    operation: str,
) -> None:
    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, "", "authorization failed")

    broker = lpi.Broker(lpi.BrokerConfig("http://admin:9644", "su", "su-pass"), run=run)
    with pytest.raises(lpi.IssuerError, match="authorization failed") as exc:
        if operation == "topic":
            broker.ensure_topic("t")
        else:
            broker.grant("dev-a-b", declaration.grants[0])
    assert exc.value.status is HTTPStatus.BAD_GATEWAY


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


# --- grant re-application --------------------------------------------------


def test_apply_grants_updates_every_issued_principal_in_the_lane(
    issuer: lpi.Issuer,
    apply_broker: FakeBroker,
    declaration: lpi.Declaration,
    ledger_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    principals = [
        issuer.issue(LOGIN, {"lane": "dev", "device": device})["principal"]
        for device in ("mac-a", "mac-b", "mac-a")
    ]
    ledger = lpi.Ledger(ledger_path)
    ledger.append({"principal": "other-lane", "lane": "other", "action": "created"})
    apply_broker.users["dev-c-mac"] = "existing-secret"
    ledger.append({"principal": "dev-c-mac", "lane": "dev", "action": "grants-applied"})
    passwords = dict(apply_broker.users)
    apply_broker.acl_argvs.clear()
    apply_broker.topic_calls.clear()
    apply_broker.set_calls.clear()
    assert (
        lpi.main(
            ["apply-grants", "--grants", str(GRANTS), "--ledger", str(ledger_path)]
        )
        == 0
    )
    assert apply_broker.set_calls == []
    assert apply_broker.users == passwords
    expected = sorted({*principals, "dev-c-mac"})
    assert capsys.readouterr().out.splitlines() == [
        f"applied {len(declaration.grants)} grants to {principal}"
        for principal in expected
    ]
    assert apply_broker.topic_calls == [
        f"onex.cmd.omnimarket.work-ledger-append-from-{principal}.v1"
        for principal in expected
    ]
    assert apply_broker.acl_argvs == [
        [principal, g.resource, g.name, g.pattern, *g.operations]
        for principal in expected
        for declared in declaration.grants
        for g in [declared.for_principal(principal)]
    ]
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()][
        -len(expected) :
    ]
    for row, principal in zip(rows, expected, strict=True):
        assert row == {
            "at": row["at"],
            "principal": principal,
            "lane": "dev",
            "action": "grants-applied",
            "grants_sha256": declaration.digest,
        }
        assert row["at"]


def test_apply_grants_accepts_multiple_names_and_deduplicates_them(
    apply_broker: FakeBroker,
    declaration: lpi.Declaration,
    ledger_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    apply_broker.users.update({"dev-a-mac": "a-secret", "dev-b-mac": "b-secret"})
    assert (
        lpi.main(
            [
                "apply-grants",
                "--grants",
                str(GRANTS),
                "--ledger",
                str(ledger_path),
                "--principal",
                "dev-a-mac",
                "dev-b-mac",
                "--principal",
                "dev-a-mac",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.splitlines() == [
        f"applied {len(declaration.grants)} grants to dev-a-mac",
        f"applied {len(declaration.grants)} grants to dev-b-mac",
    ]
    assert apply_broker.set_calls == []
    assert len(ledger_path.read_text().splitlines()) == 2


def test_apply_grants_refuses_a_principal_missing_from_the_broker(
    apply_broker: FakeBroker,
    ledger_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    apply_broker.users["dev-known-mac"] = "existing-secret"
    assert (
        lpi.main(
            [
                "apply-grants",
                "--grants",
                str(GRANTS),
                "--ledger",
                str(ledger_path),
                "--principal",
                "dev-known-mac",
                "--principal",
                "unknown",
            ]
        )
        == 2
    )
    output = capsys.readouterr()
    assert "the broker has no principal unknown" in output.err
    assert output.out == ""
    assert apply_broker.set_calls == []
    assert apply_broker.acl_argvs == []
    assert apply_broker.topic_calls == []
    assert not ledger_path.exists()


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
