# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22478: a read-only principal is issued without the work-ledger append grant.

Step 0's read-only principal must be refused by its credential when it attempts a
ledger append. For the ledger channel the credential is a broker principal, and the
refusal is the broker's: the principal holds no WRITE on the work-ledger append
writer topic, because the issuer never grants it. The grant declaration names the
withheld topic (``read_only_withholds``), and a row with role ``read_only`` is issued
the shared set less that topic. ``audit-grants`` fails when such a principal holds
the topic anyway. The broker is a fake: no network, no Docker.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from scripts import lab_principal_issuer as lpi

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
GRANTS = REPO_ROOT / "deploy" / "lab" / "developer-principal-grants.yaml"
DERIVED = REPO_ROOT / "deploy" / "lab" / "developer-principal-derived-grants.json"
APPEND_TOPIC = "onex.cmd.omnimarket.work-ledger-append-from-{principal}.v1"
RO = "dev-ro-lab-host"
GITHUB_WRITE_TOPIC = "onex.cmd.omnimarket.github-repo-write-requested.v1"
# Command topics whose consumers append ledger rows or change a pull request on the
# publisher's behalf: a write on them is a ledger or GitHub mutation by another route.
CONSUMER_WRITES = (
    "onex.cmd.omnimarket.pr-handoff-requested.v1",
    "onex.cmd.omnimarket.pr-polish-start.v1",
)
READ_ONLY_ROW = {
    "principal": "dev-ro-lab-host",
    "login": "ro@example.invalid",
    "device": "lab-host",
    "role": "read_only",
}
DEVELOPER_ROW = {
    "principal": "dev-ana-ana-laptop",
    "login": "ana@example.invalid",
    "device": "ana-laptop",
}
ACL_HEADER = "PRINCIPAL  HOST  RESOURCE-TYPE  RESOURCE-NAME  RESOURCE-PATTERN-TYPE  OPERATION  PERMISSION  ERROR"
RO_APPEND = "onex.cmd.omnimarket.work-ledger-append-from-dev-ro-lab-host.v1"
DEV_APPEND = "onex.cmd.omnimarket.work-ledger-append-from-dev-ana-ana-laptop.v1"


class FakeBroker(lpi.Broker):
    def __init__(self, listings: dict[str, str] | None = None) -> None:
        self.users: dict[str, str] = dict.fromkeys(listings or {}, "x")
        self.listings = listings or {}
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
        return self.listings[principal]


def _overlay(tmp_path: Path, *rows: dict[str, object]) -> Path:
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


def _issue(
    declaration: lpi.Declaration, tmp_path: Path, row: dict[str, object]
) -> FakeBroker:
    broker = FakeBroker()
    overlay = lpi.load_overlay(_overlay(tmp_path, row), declaration)
    lpi.Issuer(
        declaration, broker, lpi.Ledger(tmp_path / "issuances.jsonl"), overlay
    ).issue(str(row["login"]), {"lane": "dev", "device": str(row["device"])})
    return broker


def _listing(principal: str, declaration: lpi.Declaration, overlay: lpi.Overlay) -> str:
    lines = [ACL_HEADER]
    for row in sorted(lpi.expected_acl_rows(declaration, principal, overlay)):
        lines.append(
            f"User:{principal}  *  {row.resource}  {row.name}  {row.pattern}  "
            f"{row.operation}  ALLOW"
        )
    return "\n".join(lines) + "\n"


@pytest.fixture
def declaration() -> lpi.Declaration:
    return lpi.load_declaration(GRANTS)


# --- the declaration names what is withheld -----------------------------------


def test_the_declaration_withholds_the_ledger_append_writer_topic(
    declaration: lpi.Declaration,
) -> None:
    assert declaration.read_only_withheld[0] == APPEND_TOPIC
    assert set(declaration.read_only_withheld) == {APPEND_TOPIC, *CONSUMER_WRITES}
    shared = {g.name: g for g in declaration.grants if g.resource == "topic"}
    assert "write" in shared[APPEND_TOPIC].operations


@pytest.mark.parametrize(
    "withholds",
    [
        pytest.param(["onex.cmd.omnimarket.not-granted.v1"], id="stale-name"),
        pytest.param("a-string", id="not-a-list"),
        pytest.param([""], id="blank"),
        pytest.param([7], id="not-a-name"),
    ],
)
def test_a_withhold_that_is_stale_or_malformed_is_refused_at_load(
    tmp_path: Path, withholds: object
) -> None:
    doc = yaml.safe_load(GRANTS.read_text())
    doc["read_only_withholds"] = withholds
    for name in (
        doc["grants_from"],
        doc["machine_grants_from"],
        *doc["principal_grants_from"].values(),
    ):
        (tmp_path / name).write_bytes((GRANTS.parent / name).read_bytes())
    path = tmp_path / "grants.yaml"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(lpi.DeclarationError, match="must name shared topics"):
        lpi.load_declaration(path)


# --- the issued set -----------------------------------------------------------


def test_golden_a_developer_is_issued_the_ledger_append_write(
    declaration: lpi.Declaration, tmp_path: Path
) -> None:
    broker = _issue(declaration, tmp_path, DEVELOPER_ROW)
    assert (DEV_APPEND, "write") in broker.granted
    assert DEV_APPEND in broker.topics


def test_golden_a_read_only_principal_is_issued_no_ledger_append_write(
    declaration: lpi.Declaration, tmp_path: Path
) -> None:
    broker = _issue(declaration, tmp_path, READ_ONLY_ROW)
    assert not [g for g in broker.granted if g[0] == RO_APPEND]
    assert RO_APPEND not in broker.topics
    assert not [g for g in broker.granted if "work-ledger-append-from" in g[0]]


@pytest.mark.parametrize("topic", CONSUMER_WRITES)
def test_a_developer_writes_and_a_read_only_principal_is_refused_the_consumer_commands(
    declaration: lpi.Declaration, tmp_path: Path, topic: str
) -> None:
    assert (topic, "write") in _issue(declaration, tmp_path, DEVELOPER_ROW).granted
    sub = tmp_path / "ro"
    sub.mkdir()
    assert not [
        g for g in _issue(declaration, sub, READ_ONLY_ROW).granted if g[0] == topic
    ]


def test_a_read_only_principal_can_publish_no_github_write_command(
    declaration: lpi.Declaration, tmp_path: Path
) -> None:
    """GitHub channel: the write node's command topic is in no shared grant, so a write verb
    from a shared-set principal is refused by the broker, and read verbs are still served."""
    granted = _issue(declaration, tmp_path, READ_ONLY_ROW).granted
    assert not [g for g in granted if g[0] == GITHUB_WRITE_TOPIC]
    assert (
        "onex.cmd.omnimarket.github-repo-gateway-read-requested.v1",
        "write",
    ) in granted


def test_a_read_only_principal_still_reads_the_ledger(
    declaration: lpi.Declaration, tmp_path: Path
) -> None:
    broker = _issue(declaration, tmp_path, READ_ONLY_ROW)
    for topic in (
        "onex.evt.omnimarket.work-ledger-append-completed.v1",
        "onex.evt.omnimarket.work-ledger-query-completed.v1",
    ):
        assert (topic, "read") in broker.granted, topic


def test_a_read_only_principal_is_issued_the_shared_set_less_the_withheld_topic(
    declaration: lpi.Declaration, tmp_path: Path
) -> None:
    overlay = lpi.load_overlay(_overlay(tmp_path, READ_ONLY_ROW), declaration)
    held = declaration.grants_for("dev-ro-lab-host", overlay)
    assert [g for g in declaration.grants if g not in held] == [
        g for g in declaration.grants if g.name in declaration.read_only_withheld
    ]
    assert not [g for g in held if g in declaration.machine_grants]


def test_a_read_only_row_is_never_issued_the_machine_profile(
    declaration: lpi.Declaration, tmp_path: Path
) -> None:
    overlay = lpi.load_overlay(_overlay(tmp_path, READ_ONLY_ROW), declaration)
    machine_only = {g.name for g in declaration.machine_grants} - {
        g.name for g in declaration.grants
    }
    assert machine_only
    held = {g.name for g in declaration.grants_for("dev-ro-lab-host", overlay)}
    assert not held & machine_only


def test_a_read_only_row_cannot_bind_grant_roles(
    declaration: lpi.Declaration, tmp_path: Path
) -> None:
    with pytest.raises(lpi.DeclarationError, match="only a machine binds grant_roles"):
        lpi.load_overlay(
            _overlay(tmp_path, {**READ_ONLY_ROW, "grant_roles": ["x"]}), declaration
        )


def test_the_role_vocabulary_includes_read_only() -> None:
    assert "read_only" in lpi.OVERLAY_ROLES


# --- the failing control ------------------------------------------------------


def test_break_a_declaration_withholding_nothing_issues_the_append_write(
    declaration: lpi.Declaration, tmp_path: Path
) -> None:
    """The named break: empty ``read_only_withholds`` and the golden test above reads red."""
    broken = dataclasses.replace(declaration, read_only_withheld=())
    broker = _issue(broken, tmp_path, READ_ONLY_ROW)
    assert (RO_APPEND, "write") in broker.granted


# --- the drift check ----------------------------------------------------------


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
            str(_overlay(tmp_path, READ_ONLY_ROW, DEVELOPER_ROW)),
            "--rpk-container",
            "issuer",
            *extra,
        ]
    )


def test_the_drift_check_passes_a_read_only_principal_holding_no_append_write(
    declaration: lpi.Declaration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    overlay = lpi.load_overlay(_overlay(tmp_path, READ_ONLY_ROW), declaration)
    broker = FakeBroker(
        {"dev-ro-lab-host": _listing("dev-ro-lab-host", declaration, overlay)}
    )
    assert _audit(tmp_path, broker, monkeypatch, "--principal", RO) == 0
    assert capsys.readouterr().out.splitlines()[0].endswith("missing=0 extra=0")


def test_error_the_drift_check_fails_a_read_only_principal_that_holds_the_append_write(
    declaration: lpi.Declaration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The planted violation: the broker still lists WRITE on the append topic."""
    overlay = lpi.load_overlay(_overlay(tmp_path, READ_ONLY_ROW), declaration)
    listing = _listing("dev-ro-lab-host", declaration, overlay) + (
        f"User:dev-ro-lab-host  *  TOPIC  {RO_APPEND}  LITERAL  WRITE  ALLOW\n"
    )
    broker = FakeBroker({"dev-ro-lab-host": listing})
    assert _audit(tmp_path, broker, monkeypatch, "--principal", RO) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0].endswith("missing=0 extra=1")
    assert f"EXTRA dev-ro-lab-host topic literal {RO_APPEND} write" in out


def test_the_drift_check_names_a_missing_read_grant_of_a_read_only_principal(
    declaration: lpi.Declaration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    overlay = lpi.load_overlay(_overlay(tmp_path, READ_ONLY_ROW), declaration)
    dropped = "onex.evt.omnimarket.work-ledger-append-completed.v1"
    listing = "\n".join(
        line
        for line in _listing("dev-ro-lab-host", declaration, overlay).splitlines()
        if not (dropped in line and " read " in line)
    )
    broker = FakeBroker({"dev-ro-lab-host": listing + "\n"})
    assert _audit(tmp_path, broker, monkeypatch, "--principal", RO) == 1
    assert (
        f"MISSING dev-ro-lab-host topic literal {dropped} read"
        in capsys.readouterr().out.splitlines()
    )


def test_a_developer_extra_is_still_only_reported_under_a_role(
    declaration: lpi.Declaration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay = lpi.load_overlay(_overlay(tmp_path, DEVELOPER_ROW), declaration)
    listing = (
        _listing("dev-ana-ana-laptop", declaration, overlay)
        + "User:dev-ana-ana-laptop  *  TOPIC  fleet.private.v1  LITERAL  WRITE  ALLOW\n"
    )
    broker = FakeBroker({"dev-ana-ana-laptop": listing})
    assert _audit(tmp_path, broker, monkeypatch, "--role", "developer") == 0


def test_the_derived_file_still_carries_the_append_grant_for_developers() -> None:
    rows = json.loads(DERIVED.read_text())["grants"]
    assert [
        r for r in rows if r["resource"] == APPEND_TOPIC and r["operation"] == "WRITE"
    ]
