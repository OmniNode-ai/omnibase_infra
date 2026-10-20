# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17427: apply and audit grants through the issuer container's own rpk, without a caller superuser identity or password changes."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from http import HTTPStatus
from pathlib import Path

import pytest
import yaml

from scripts import lab_principal_issuer as lpi

REPO_ROOT = Path(__file__).resolve().parents[2]
GRANTS = REPO_ROOT / "deploy" / "lab" / "developer-principal-grants.yaml"


class FakeRunner:
    def __init__(self, users: Sequence[str] = ()) -> None:
        self.users = users
        self.calls: list[list[str]] = []
        self.held: dict[str, set[lpi.AclRow]] = {}
        self.user_list_returncode = 0
        self.user_list_stderr = ""

    def __call__(
        self, argv: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        assert kwargs in ({}, {"capture_output": True, "text": True, "check": False})
        assert list(argv[:4]) == ["docker", "exec", "issuer-c", "rpk"]
        command = list(argv[3:])
        if command == ["rpk", "security", "user", "list"]:
            return subprocess.CompletedProcess(
                argv,
                self.user_list_returncode,
                "USERNAME\n" + "\n".join(self.users),
                self.user_list_stderr,
            )
        if command[:3] == ["rpk", "security", "acl"]:
            principal = command[command.index("--allow-principal") + 1].removeprefix(
                "User:"
            )
            if command[3] == "list":
                lines = [
                    "PRINCIPAL  HOST  RESOURCE-TYPE  RESOURCE-NAME  RESOURCE-PATTERN-TYPE  OPERATION  PERMISSION  ERROR"
                ]
                lines.extend(
                    f"User:{principal}  {row.host}  {row.resource.upper()}  {row.name}  "
                    f"{row.pattern.upper()}  {row.operation.upper()}  ALLOW"
                    for row in sorted(self.held.get(principal, set()))
                )
                return subprocess.CompletedProcess(argv, 0, "\n".join(lines), "")
            assert command[3] == "create"
            resource = "topic" if "--topic" in command else "group"
            name = command[command.index(f"--{resource}") + 1]
            pattern = command[command.index("--resource-pattern-type") + 1]
            operations = command[command.index("--operation") + 1].split(",")
            self.held.setdefault(principal, set()).update(
                lpi.AclRow(resource, name, pattern, operation)
                for operation in operations
            )
        else:
            assert command[:3] == ["rpk", "topic", "create"]
        return subprocess.CompletedProcess(argv, 0, "", "")


@pytest.fixture
def declaration() -> lpi.Declaration:
    return lpi.load_declaration(GRANTS)


@pytest.fixture
def principal(declaration: lpi.Declaration) -> str:
    name = lpi.principal_name(declaration, "dev.person@omninode.ai", "mac-a")
    assert name == "dev-dev-person-mac-a"
    return name


@pytest.fixture
def overlay(tmp_path: Path, principal: str) -> Path:
    path = tmp_path / "overlay.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "developer_principal.v1",
                "grants_source": "omnibase_infra:deploy/lab/developer-principal-grants.yaml",
                "principals": [
                    {
                        "principal": principal,
                        "login": "dev.person@omninode.ai",
                        "device": "mac-a",
                    }
                ],
            }
        )
    )
    return path


@pytest.fixture
def no_superuser_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("RPK_USER", "RPK_PASS", "RPK_ADMIN_HOSTS"):
        monkeypatch.delenv(key, raising=False)


def test_rpk_container_runs_every_rpk_call_inside_the_container(
    declaration: lpi.Declaration, principal: str
) -> None:
    fake = FakeRunner([principal])
    broker = lpi.ContainerBroker("issuer-c", run=fake)
    grant = declaration.grants[0].for_principal(principal)
    broker.grant(principal, grant)
    assert lpi.parse_acl_listing(broker.list_acls(principal), principal) == {
        lpi.AclRow(grant.resource, grant.name, grant.pattern, operation)
        for operation in grant.operations
    }
    broker.ensure_topic("writer.v1")
    assert broker.user_exists(principal)
    assert len(fake.calls) == 4
    assert all(call[:4] == ["docker", "exec", "issuer-c", "rpk"] for call in fake.calls)


def test_rpk_container_reads_users_from_rpk_and_not_the_admin_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_admin(*args: object, **kwargs: object) -> None:
        pytest.fail("a container broker must not call the admin API")

    monkeypatch.setattr(lpi.urllib.request, "urlopen", refuse_admin)
    fake = FakeRunner(["  listed-user  ", "another-user"])
    broker = lpi.ContainerBroker("issuer-c", run=fake)
    assert broker.user_exists("listed-user")
    assert not broker.user_exists("absent-user")
    assert not broker.user_exists("USERNAME")
    assert not broker.user_exists("listed")
    fake.user_list_returncode = 1
    fake.user_list_stderr = "  " + "x" * 350 + "\n"
    with pytest.raises(lpi.IssuerError) as exc:
        broker.user_exists("listed-user")
    assert exc.value.status == HTTPStatus.BAD_GATEWAY
    assert str(exc.value) == "the broker refused the user listing: " + "x" * 300


@pytest.mark.parametrize("exists", [False, True])
def test_rpk_container_never_sets_a_password(exists: bool) -> None:
    fake = FakeRunner()
    broker = lpi.ContainerBroker("issuer-c", run=fake)
    with pytest.raises(
        lpi.IssuerError, match="a container broker never sets or rotates a password"
    ) as exc:
        broker.set_user("a-user", "a-password", exists=exists)
    assert exc.value.status == HTTPStatus.BAD_REQUEST
    assert fake.calls == []


@pytest.mark.parametrize("container", ["", "  "])
def test_rpk_container_refuses_a_blank_container(container: str) -> None:
    with pytest.raises(lpi.IssuerError) as exc:
        lpi.ContainerBroker(container)
    assert exc.value.status == HTTPStatus.BAD_REQUEST


def test_apply_grants_with_rpk_container_needs_no_superuser_environment(
    monkeypatch: pytest.MonkeyPatch,
    no_superuser_environment: None,
    declaration: lpi.Declaration,
    principal: str,
    overlay: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeRunner([principal])
    monkeypatch.setattr(lpi.subprocess, "run", fake)
    ledger = tmp_path / "issuances.jsonl"
    assert (
        lpi.main(
            [
                "apply-grants",
                "--grants",
                str(GRANTS),
                "--ledger",
                str(ledger),
                "--overlay",
                str(overlay),
                "--rpk-container",
                "issuer-c",
            ]
        )
        == 0
    )
    assert all(call[:3] == ["docker", "exec", "issuer-c"] for call in fake.calls)
    assert fake.held[principal] == lpi.expected_acl_rows(declaration, principal)
    assert capsys.readouterr().out.splitlines() == [
        f"applied {len(declaration.grants)} grants to {principal}"
    ]
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["action"] == "grants-applied"
    assert rows[0]["principal"] == principal
    assert rows[0]["grants_sha256"] == declaration.digest


def test_audit_grants_with_rpk_container_reports_exact_missing_and_extra(
    monkeypatch: pytest.MonkeyPatch,
    no_superuser_environment: None,
    declaration: lpi.Declaration,
    principal: str,
    overlay: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = lpi.expected_acl_rows(declaration, principal)
    missing = min(expected)
    extra = lpi.AclRow("topic", "extra.v1", "literal", "read")
    fake = FakeRunner([principal])
    fake.held[principal] = (set(expected) - {missing}) | {extra}
    monkeypatch.setattr(lpi.subprocess, "run", fake)
    argv = [
        "audit-grants",
        "--grants",
        str(GRANTS),
        "--ledger",
        str(tmp_path / "issuances.jsonl"),
        "--overlay",
        str(overlay),
        "--rpk-container",
        "issuer-c",
    ]
    assert lpi.main(argv) == 1
    lines = capsys.readouterr().out.splitlines()
    assert "missing=1 extra=1" in lines[0]
    assert lines[1:] == [
        f"MISSING {principal} {missing.resource} {missing.pattern} {missing.name} {missing.operation}",
        f"EXTRA {principal} topic literal extra.v1 read",
    ]
    fake.held[principal] = set(expected)
    assert lpi.main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert "missing=0 extra=0" in lines[0]
    assert all(call[:3] == ["docker", "exec", "issuer-c"] for call in fake.calls)
    assert all(call[3:7] != ["rpk", "security", "acl", "create"] for call in fake.calls)
    assert not (tmp_path / "issuances.jsonl").exists()


@pytest.mark.parametrize("command", ["serve", "check"])
def test_serve_and_check_take_no_rpk_container(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [command, "--grants", str(GRANTS)]
    if command == "serve":
        argv += [
            "--ledger",
            str(tmp_path / "issuances.jsonl"),
            "--socket",
            str(tmp_path / "issuer.sock"),
            "--proxy-uid",
            "0",
        ]
    with pytest.raises(SystemExit) as exc:
        lpi.main([*argv, "--rpk-container", "x"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --rpk-container x" in capsys.readouterr().err


def test_without_rpk_container_apply_grants_still_needs_the_superuser_environment(
    no_superuser_environment: None, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="RPK_USER"):
        lpi.main(
            [
                "apply-grants",
                "--grants",
                str(GRANTS),
                "--ledger",
                str(tmp_path / "issuances.jsonl"),
            ]
        )


@pytest.mark.parametrize("command", ["apply-grants", "audit-grants"])
def test_rpk_container_cli_reports_a_blank_container(
    command: str,
    no_superuser_environment: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        lpi.main(
            [
                command,
                "--grants",
                str(GRANTS),
                "--ledger",
                str(tmp_path / "issuances.jsonl"),
                "--rpk-container",
                "  ",
            ]
        )
        == 2
    )
    assert capsys.readouterr().err == (
        "lab_principal_issuer: a container name must not be blank\n"
    )
