# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21622: a machine login is minted holding the whole machine profile, and audited.

Each machine login used to be minted with a set that lacked what its hook-emit drainer
publishes, and was then given those grants by hand, one host at a time (the operator
Mac's drainer, then h051, whose ledger append was refused on 2026-10-30). The machine
profile is now the issued profile: declared once in the bindings, granted in the act that
mints the principal. ``audit-grants --role machine`` reads the machines a
``developer.principal`` overlay declares and fails naming each one and each grant of the
profile the broker lacks. The broker is a fake: no network, no Docker.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts import lab_principal_issuer as lpi

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
LAB = REPO_ROOT / "deploy" / "lab"
GRANTS = LAB / "developer-principal-grants.yaml"
DERIVED = LAB / "developer-principal-derived-grants.json"
UNIT = (
    REPO_ROOT
    / "deploy"
    / "principal-grant-audit"
    / "onex-principal-grant-audit.service"
)
LEDGER_ROW_KINDS = (
    "ack",
    "claim",
    "correction",
    "friction",
    "hold",
    "msg",
    "operator-consent",
    "release",
    "ruling",
    "status",
    "terminal",
    "terminal-refused",
)
MACHINE_TOPICS = (
    "onex.evt.omnimarket.credential-registered.v1",
    *(f"onex.evt.omnimarket.work-ledger-{kind}.v1" for kind in LEDGER_ROW_KINDS),
)
MACHINE_ROW = {
    "principal": "dev-ops-lab-host",
    "login": "ops@example.invalid",
    "device": "lab-host",
    "role": "machine",
}
DEVELOPER_ROW = {
    "principal": "dev-ana-ana-laptop",
    "login": "ana@example.invalid",
    "device": "ana-laptop",
}
ACL_HEADER = "PRINCIPAL  HOST  RESOURCE-TYPE  RESOURCE-NAME  RESOURCE-PATTERN-TYPE  OPERATION  PERMISSION  ERROR"


class FakeBroker(lpi.Broker):
    def __init__(self, listings: dict[str, str] | None = None) -> None:
        self.users: dict[str, str] = dict.fromkeys(listings or {}, "x")
        self.listings = listings or {}
        self.asked: list[str] = []
        self.granted: list[tuple[str, str]] = []
        self.topics: list[str] = []

    def user_exists(self, name: str) -> bool:
        return name in self.users

    def set_user(self, name: str, password: str, *, exists: bool) -> None:
        self.users[name] = password

    def grant(self, principal: str, grant: lpi.Grant) -> None:
        assert not grant.templated
        self.granted.extend((grant.name, op) for op in grant.operations)

    def ensure_topic(self, name: str) -> None:
        self.topics.append(name)

    def list_acls(self, principal: str) -> str:
        self.asked.append(principal)
        return self.listings[principal]


def _overlay(tmp_path: Path, *rows: dict[str, str]) -> Path:
    path = tmp_path / "developer-principal-overlay.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "developer_principal.v1",
                "grants_source": lpi.OVERLAY_GRANTS_SOURCE,
                "principals": list(rows),
            }
        )
    )
    return path


def _listing(principal: str, drop: str | None = None, extra: str | None = None) -> str:
    lines = [ACL_HEADER]
    for g in json.loads(DERIVED.read_text())["grants"]:
        line = (
            f"User:{principal}  *  {g['resource_type']}  "
            f"{g['resource'].replace('{principal}', principal)}  {g['pattern']}  "
            f"{g['operation']}  ALLOW"
        )
        if line != drop:
            lines.append(line)
    if extra:
        lines.append(extra)
    return "\n".join(lines) + "\n"


@pytest.fixture
def declaration() -> lpi.Declaration:
    return lpi.load_declaration(GRANTS)


# --- the profile is the issued set ----------------------------------------------


def test_every_issued_principal_gets_the_machine_profile_at_issuance(
    declaration: lpi.Declaration, tmp_path: Path
) -> None:
    broker = FakeBroker()
    issuer = lpi.Issuer(declaration, broker, lpi.Ledger(tmp_path / "issuances.jsonl"))
    issuer.issue("ops@example.invalid", {"lane": "dev", "device": "lab-host"})
    for topic in MACHINE_TOPICS:
        assert (topic, "write") in broker.granted, topic
        assert (topic, "describe") in broker.granted, topic
    # the append client the h051 refusal of 2026-10-30 lacked is issued in the same act
    assert (
        "onex.evt.omnimarket.work-ledger-append-completed.v1",
        "read",
    ) in broker.granted
    assert (
        "onex.cmd.omnimarket.work-ledger-append-from-dev-ops-lab-host.v1",
        "write",
    ) in broker.granted


def test_a_machine_never_reads_a_ledger_row_event(declaration: lpi.Declaration) -> None:
    for grant in declaration.grants:
        if grant.name in MACHINE_TOPICS:
            assert grant.operations == ("describe", "write"), grant.name


# --- the overlay names which principals are machines ----------------------------


def test_the_role_vocabulary_is_the_core_enum() -> None:
    core = pytest.importorskip(
        "omnibase_core.enums.enum_developer_principal_role",
        reason="the omnibase_core release that ships EnumDeveloperPrincipalRole",
    )
    assert {role.value for role in core.EnumDeveloperPrincipalRole} == lpi.OVERLAY_ROLES


def test_a_machine_row_loads_with_its_role(
    declaration: lpi.Declaration, tmp_path: Path
) -> None:
    overlay = lpi.load_overlay(
        _overlay(tmp_path, MACHINE_ROW, DEVELOPER_ROW), declaration
    )
    assert {row.principal: row.role for row in overlay.rows} == {
        "dev-ops-lab-host": "machine",
        "dev-ana-ana-laptop": "developer",
    }


@pytest.mark.parametrize("role", ["admin", "Machine", ""])
def test_a_role_outside_the_vocabulary_is_refused(
    declaration: lpi.Declaration, tmp_path: Path, role: str
) -> None:
    with pytest.raises(lpi.DeclarationError, match="role must be one of"):
        lpi.load_overlay(_overlay(tmp_path, {**MACHINE_ROW, "role": role}), declaration)


def test_a_row_still_cannot_carry_a_grant(
    declaration: lpi.Declaration, tmp_path: Path
) -> None:
    with pytest.raises(lpi.DeclarationError, match="a grant cannot be typed here"):
        lpi.load_overlay(
            _overlay(tmp_path, {**MACHINE_ROW, "grants": "x"}), declaration
        )


# --- the drift check -----------------------------------------------------------


def _audit(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch, *extra: str
) -> int:
    monkeypatch.setattr(lpi, "ContainerBroker", lambda name: broker)
    return lpi.main(
        [
            "audit-grants",
            "--grants",
            str(GRANTS),
            "--ledger",
            str(tmp_path / "issuances.jsonl"),
            "--overlay",
            str(_overlay(tmp_path, MACHINE_ROW, DEVELOPER_ROW)),
            "--rpk-container",
            "issuer",
            *extra,
        ]
    )


def test_the_drift_check_passes_a_machine_holding_the_whole_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    principal = MACHINE_ROW["principal"]
    broker = FakeBroker({principal: _listing(principal)})
    assert _audit(tmp_path, broker, monkeypatch, "--role", "machine") == 0
    assert capsys.readouterr().out.splitlines()[0].endswith("missing=0 extra=0")
    assert broker.granted == []


def test_the_drift_check_fails_naming_the_machine_and_each_missing_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The planted violation: a machine whose drainer cannot publish a status row."""
    principal = MACHINE_ROW["principal"]
    planted = (
        f"User:{principal}  *  TOPIC  onex.evt.omnimarket.work-ledger-status.v1  "
        "LITERAL  WRITE  ALLOW"
    )
    broker = FakeBroker({principal: _listing(principal, drop=planted)})
    assert _audit(tmp_path, broker, monkeypatch, "--role", "machine") == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0].endswith("missing=1 extra=0")
    assert (
        f"MISSING {principal} topic literal onex.evt.omnimarket.work-ledger-status.v1 write"
        in out
    )


def test_a_grant_another_declaration_gives_a_machine_is_reported_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    principal = MACHINE_ROW["principal"]
    extra = f"User:{principal}  *  TOPIC  fleet.private.v1  LITERAL  WRITE  ALLOW"
    broker = FakeBroker({principal: _listing(principal, extra=extra)})
    assert _audit(tmp_path, broker, monkeypatch, "--role", "machine") == 0
    assert f"EXTRA {principal} topic literal fleet.private.v1 write" in (
        capsys.readouterr().out.splitlines()
    )
    # without --role an extra grant still fails, as before
    broker.listings[principal] = _listing(principal, extra=extra)
    monkeypatch.setattr(lpi, "ContainerBroker", lambda name: broker)
    code = lpi.main(
        [
            "audit-grants",
            "--grants",
            str(GRANTS),
            "--ledger",
            str(tmp_path / "issuances.jsonl"),
            "--overlay",
            str(_overlay(tmp_path, MACHINE_ROW, DEVELOPER_ROW)),
            "--principal",
            principal,
            "--rpk-container",
            "issuer",
        ]
    )
    assert code == 1


def test_the_drift_check_never_reads_a_developer_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    principal = MACHINE_ROW["principal"]
    broker = FakeBroker({principal: _listing(principal)})
    assert _audit(tmp_path, broker, monkeypatch, "--role", "machine") == 0
    assert broker.asked == [principal]


def test_a_role_no_row_holds_is_a_refusal_not_a_clean_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(lpi, "ContainerBroker", lambda name: FakeBroker())
    code = lpi.main(
        [
            "audit-grants",
            "--grants",
            str(GRANTS),
            "--ledger",
            str(tmp_path / "l.jsonl"),
            "--overlay",
            str(_overlay(tmp_path, DEVELOPER_ROW)),
            "--role",
            "machine",
            "--rpk-container",
            "issuer",
        ]
    )
    assert code == 2
    assert "no developer principals selected" in capsys.readouterr().err


def test_the_role_filter_needs_an_overlay(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["audit-grants", "--grants", str(GRANTS), "--ledger", str(tmp_path / "l")]
    assert lpi.main([*argv, "--role", "machine"]) == 2
    assert "--role needs --overlay" in capsys.readouterr().err


def test_the_audit_unit_runs_the_machine_audit_and_names_no_host() -> None:
    text = UNIT.read_text()
    assert "audit-grants" in text and "--role machine" in text
    assert "--overlay" in text and "--rpk-container" in text
    assert "apply-grants" not in text and "serve" not in text.split("ExecStart", 1)[1]
