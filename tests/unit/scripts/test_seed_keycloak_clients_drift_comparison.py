# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Roster drift must be a difference in the realm, not in list order (OMN-18600).

``scripts/seed-keycloak-clients.py`` compared every declared field with
``existing.get(field) != spec[field]``. That comparison is positional for a
list and whole-map for a dict, and Keycloak is free to return ``redirectUris``
and ``webOrigins`` in its own order and to add attribute keys of its own. So
three clients -- ``omniweb``, ``omnidash-spa`` and ``onex-customer`` -- reported
``redirectUris``, ``webOrigins`` and ``attributes`` as drifted on every single
reconcile, including a reconcile run immediately after a clean one.

Measured read-only against the live ``.201`` dev-lane realm on 2026-09-17: for
all three clients the two collections are equal as sets and unequal as lists,
and the live attribute map is the roster's plus keys Keycloak maintains itself
(``realm_client`` on every client; ``client.secret.creation.time`` on every
client Keycloak has minted a secret for). Not one declared attribute value
differed.

Why this is more than untidy output: any drift at all sends the client down the
full-representation update path, and that path is the one whose secret-clearing
side effect destroyed ``onex-api``'s client secret under OMN-16504. Three
clients took the most dangerous path in this script on every reconcile, forever,
for no difference at all. Narrowing when that path is entered is a layer neither
the payload narrowing (``omnibase_infra#3555``) nor the post-loop secret guards
provide -- both of those act after the path has been taken.

The fail-closed direction is pinned here too, and it is the half that matters:
a comparison loose enough to stop reporting a genuine membership change would be
a worse defect than the spurious drift it removes, because the reconciler would
silently stop reconciling the fields it claims to own.

Related Tickets:
    - OMN-18600: spurious drift on three clients sends each down the update path
    - OMN-16504: onex-api ends a reconcile with no client secret; introspection 401
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED_SCRIPT = REPO_ROOT / "scripts" / "seed-keycloak-clients.py"

KC_URL = "http://keycloak.test.invalid"
REALM = "omninode"


def _load_seeder() -> types.ModuleType:
    """Import the reconciler from its path.

    It is deliberately a hyphenated plain file run as ``python
    scripts/seed-keycloak-clients.py``, so it has no importable module name.
    """
    spec = importlib.util.spec_from_file_location("_seed_keycloak_clients", SEED_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingKeycloak:
    """A Keycloak admin API that records every call that is not a read.

    Deliberately separate from the fake in the OMN-16504 guard test: that one
    models the secret-clearing side effect, this one models Keycloak's ordering
    freedom and its server-managed attribute keys, and the property under test
    here is *how many* writes happen rather than what a write contains.
    """

    def __init__(self, clients: list[dict[str, Any]]) -> None:
        self.clients: list[dict[str, Any]] = [dict(c) for c in clients]
        self.writes: list[tuple[str, str]] = []
        self.put_payloads: list[dict[str, Any]] = []

    def _find(self, internal_id: str) -> dict[str, Any]:
        for client in self.clients:
            if client["id"] == internal_id:
                return client
        raise AssertionError(f"fake has no client with internal id {internal_id}")

    def request(
        self,
        method: str,
        url: str,
        token: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        path = url[len(KC_URL) :]
        base = f"/admin/realms/{REALM}/clients"

        if method != "GET":
            self.writes.append((method, path))

        if method == "GET" and path.startswith(f"{base}?clientId="):
            wanted = path.split("clientId=", 1)[1]
            return 200, [c for c in self.clients if c["clientId"] == wanted]

        if method == "GET" and path.endswith("/client-secret"):
            client = self._find(path[len(base) + 1 : -len("/client-secret")])
            if client.get("publicClient"):
                return 404, {"error": "Client is not confidential"}
            return 200, {"type": "secret", "value": client.get("secret") or ""}

        if method == "GET" and path.startswith(f"{base}/"):
            return 200, dict(self._find(path[len(base) + 1 :]))

        if method == "PUT" and path.startswith(f"{base}/"):
            assert payload is not None
            self.put_payloads.append(dict(payload))
            client = self._find(path[len(base) + 1 :])
            for key, value in payload.items():
                if key in ("id", "clientId"):
                    continue
                client[key] = value
            return 204, None

        raise AssertionError(f"fake received an unmodelled call: {method} {path}")


# ---------------------------------------------------------------------------
# Fixtures mirroring the live .201 dev-lane realm
# ---------------------------------------------------------------------------

_ROSTER_REDIRECTS = [
    "https://dash.omninode.ai/*",
    "https://dev.dash.omninode.ai/*",
    "http://localhost:3000/*",
    "http://localhost:8080/*",
]
_ROSTER_ORIGINS = [
    "https://dash.omninode.ai",
    "https://dev.dash.omninode.ai",
    "http://localhost:3000",
    "http://localhost:8080",
]


def _spec(**overrides: Any) -> dict[str, Any]:
    """The omnidash-spa roster entry, trimmed to the fields under test."""
    spec: dict[str, Any] = {
        "clientId": "omnidash-spa",
        "publicClient": True,
        "fullScopeAllowed": False,
        "standardFlowEnabled": True,
        "directAccessGrantsEnabled": True,
        "serviceAccountsEnabled": False,
        "attributes": {"pkce.code.challenge.method": "S256"},
        "redirectUris": list(_ROSTER_REDIRECTS),
        "webOrigins": list(_ROSTER_ORIGINS),
    }
    spec.update(overrides)
    return spec


def _live(**overrides: Any) -> dict[str, Any]:
    """The same client as Keycloak returns it: reordered, plus its own keys.

    The permutation and the extra attribute key are both copied from the live
    realm rather than invented -- Keycloak returned the collections in a
    different order and had added ``realm_client`` to the attribute map.
    """
    live: dict[str, Any] = {
        "id": "internal-omnidash-spa",
        "clientId": "omnidash-spa",
        "publicClient": True,
        "fullScopeAllowed": False,
        "standardFlowEnabled": True,
        "directAccessGrantsEnabled": True,
        "serviceAccountsEnabled": False,
        "attributes": {
            "realm_client": "false",
            "pkce.code.challenge.method": "S256",
        },
        "redirectUris": list(reversed(_ROSTER_REDIRECTS)),
        "webOrigins": list(reversed(_ROSTER_ORIGINS)),
    }
    live.update(overrides)
    return live


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake: RecordingKeycloak,
    clients: list[dict[str, Any]],
) -> types.ModuleType:
    seeder = _load_seeder()
    monkeypatch.setattr(seeder, "_request", fake.request)
    monkeypatch.setattr(seeder, "_get_token", lambda *a, **k: "admin-token")

    config = tmp_path / "desired-clients.json"
    config.write_text(json.dumps({"realm": REALM, "clients": clients}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "seed-keycloak-clients.py",
            "--kc-url",
            KC_URL,
            "--realm",
            REALM,
            "--admin-password",
            "unused-by-the-fake",
            "--config",
            str(config),
        ],
    )
    seeder.main()
    return seeder


_RECONCILE_OPS = frozenset({"created", "updated", "unchanged"})


def _fields_changed(capsys: pytest.CaptureFixture[str], client_id: str) -> list[str]:
    """Read the reconcile's own JSON-lines verdict back for one client.

    The read-only preflight emits a record per client too, so the op is what
    separates the loop's verdict from the survey's observation. Matching on
    ``clientId`` alone silently reads the survey record, whose shape has no
    ``fields_changed`` at all -- which reads as "nothing drifted" and makes
    every assertion here pass for the wrong reason.
    """
    for line in capsys.readouterr().out.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("clientId") == client_id and record.get("op") in _RECONCILE_OPS:
            return list(record.get("fields_changed", []))
    raise AssertionError(f"no reconcile verdict was logged for {client_id}")


@pytest.mark.unit
class TestOrderIsNotDrift:
    """AC1: a list-valued roster field compares as an unordered collection."""

    @pytest.mark.parametrize("field", ["redirectUris", "webOrigins"])
    def test_a_permutation_of_the_declared_value_is_not_drift(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        field: str,
    ) -> None:
        """Keycloak returning the same URIs in another order is not a change."""
        live = _live()
        # Isolate the field under test: every other field matches positionally.
        other = "webOrigins" if field == "redirectUris" else "redirectUris"
        live[other] = list(_spec()[other])

        fake = RecordingKeycloak([live])
        _run_main(monkeypatch, tmp_path, fake, [_spec()])

        assert field not in _fields_changed(capsys, "omnidash-spa"), (
            f"{field} is a permutation of the declared value, not drift. "
            f"live={live[field]} desired={_spec()[field]}"
        )

    @pytest.mark.parametrize("field", ["redirectUris", "webOrigins"])
    def test_a_genuine_membership_change_is_still_drift(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        field: str,
    ) -> None:
        """AC6: the fail-closed half -- content differences still report."""
        live = _live()
        changed = list(_spec()[field])
        changed[0] = "https://attacker.example.invalid/*"
        live[field] = changed

        fake = RecordingKeycloak([live])
        _run_main(monkeypatch, tmp_path, fake, [_spec()])

        assert field in _fields_changed(capsys, "omnidash-spa"), (
            f"{field} differs in membership and must still be reported as drift"
        )
        assert fake.put_payloads, "a genuine drift must still be applied"
        assert fake.put_payloads[0][field] == _spec()[field], (
            "the update must carry the roster's declared value for the field"
        )


@pytest.mark.unit
class TestServerManagedAttributesAreNotDrift:
    """AC2/AC3: Keycloak's own attribute keys are named, and are not drift."""

    def test_a_server_added_attribute_key_is_not_drift(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The live map is the roster's plus realm_client. That is converged."""
        fake = RecordingKeycloak([_live()])
        _run_main(monkeypatch, tmp_path, fake, [_spec()])

        assert "attributes" not in _fields_changed(capsys, "omnidash-spa")

    def test_a_declared_attribute_value_that_differs_is_still_drift(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """AC6: a roster-declared key holding another value still reports."""
        live = _live(
            attributes={
                "realm_client": "false",
                "pkce.code.challenge.method": "plain",
            }
        )
        fake = RecordingKeycloak([live])
        _run_main(monkeypatch, tmp_path, fake, [_spec()])

        assert "attributes" in _fields_changed(capsys, "omnidash-spa"), (
            "pkce.code.challenge.method is declared by the roster and differs"
        )
        assert (
            fake.put_payloads[0]["attributes"]["pkce.code.challenge.method"] == "S256"
        ), "the declared value must be applied"

    def test_a_declared_attribute_key_absent_from_the_realm_is_still_drift(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """AC6: dropping a declared key must not read as converged."""
        live = _live(attributes={"realm_client": "false"})
        fake = RecordingKeycloak([live])
        _run_main(monkeypatch, tmp_path, fake, [_spec()])

        assert "attributes" in _fields_changed(capsys, "omnidash-spa")

    def test_an_undeclared_key_that_keycloak_does_not_manage_is_drift(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """AC6: only the NAMED keys are exempt, not every live-only key.

        An attribute nobody declared and Keycloak does not maintain is
        undeclared state on a surface the roster owns. Ignoring every live-only
        key would make the roster stop being the desired state for attributes.
        """
        live = _live(
            attributes={
                "realm_client": "false",
                "pkce.code.challenge.method": "S256",
                "hand.edited.by.someone": "yes",
            }
        )
        fake = RecordingKeycloak([live])
        _run_main(monkeypatch, tmp_path, fake, [_spec()])

        assert "attributes" in _fields_changed(capsys, "omnidash-spa")
        assert "hand.edited.by.someone" not in fake.put_payloads[0]["attributes"], (
            "an undeclared, unmanaged attribute must be reconciled away"
        )

    def test_the_server_managed_key_set_is_a_pinned_constant(self) -> None:
        """AC3: widening the exemption is a visible, one-line decision.

        Both keys were measured on the live ``.201`` dev-lane realm on
        2026-09-17: ``realm_client`` on all nine roster clients, and
        ``client.secret.creation.time`` on the three Keycloak has minted a
        secret for. This assertion exists so the next key Keycloak adds arrives
        as a failing test rather than as a silent return of the same defect.
        """
        seeder = _load_seeder()

        assert (
            frozenset({"realm_client", "client.secret.creation.time"})
            == seeder._SERVER_MANAGED_ATTRIBUTE_KEYS
        )

    def test_a_genuine_attribute_update_does_not_drop_keycloaks_own_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The exemption holds on the write path as well as the read path.

        The update payload replaces ``attributes`` wholesale with the roster's
        map, so without this the one reconcile that legitimately changes an
        attribute would also delete the timestamp Keycloak wrote when it minted
        the client's secret.
        """
        live = _live(
            attributes={
                "realm_client": "false",
                "client.secret.creation.time": "1757000000",
                "pkce.code.challenge.method": "plain",
            }
        )
        fake = RecordingKeycloak([live])
        _run_main(monkeypatch, tmp_path, fake, [_spec()])

        sent = fake.put_payloads[0]["attributes"]
        assert sent["pkce.code.challenge.method"] == "S256"
        assert sent["realm_client"] == "false"
        assert sent["client.secret.creation.time"] == "1757000000"


@pytest.mark.unit
class TestAConvergedRealmIsNotWritten:
    """AC4: a reconcile over a converged realm issues no client write at all."""

    def test_no_write_reaches_keycloak(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Three clients shaped exactly as the live realm returns them.

        This fails against the pre-OMN-18600 comparison, which updates all
        three on every run.
        """
        specs = [
            _spec(),
            _spec(clientId="omniweb", publicClient=False),
            _spec(clientId="onex-customer", directAccessGrantsEnabled=False),
        ]
        lives = [
            _live(),
            _live(
                id="internal-omniweb",
                clientId="omniweb",
                publicClient=False,
                secret="a-live-client-secret",  # pragma: allowlist secret
            ),
            _live(
                id="internal-onex-customer",
                clientId="onex-customer",
                directAccessGrantsEnabled=False,
            ),
        ]

        fake = RecordingKeycloak(lives)
        _run_main(monkeypatch, tmp_path, fake, specs)

        assert fake.writes == [], (
            "a converged realm must be read and left alone. Writes issued: "
            f"{fake.writes}"
        )


@pytest.mark.unit
class TestThePreflightAgreesWithTheReconcile:
    """The read-only survey and the loop must not disagree about drift.

    The OMN-18170 preflight surveys every roster entry before the first write
    and reports each one's drift fields. If it kept the positional comparison
    while the loop stopped using it, an operator reading the survey would be
    told three clients are drifted and then watch the reconcile report them
    unchanged.
    """

    def test_the_survey_reports_no_drift_on_a_converged_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seeder = _load_seeder()
        fake = RecordingKeycloak([_live()])
        monkeypatch.setattr(seeder, "_request", fake.request)

        record = seeder._preflight_client(KC_URL, REALM, "admin-token", _spec())

        assert record["drift_fields"] == []
        assert fake.writes == [], "the survey is read-only"

    def test_the_survey_still_reports_a_genuine_difference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC6, on the survey path: a real change must still be surfaced."""
        seeder = _load_seeder()
        live = _live()
        live["redirectUris"] = ["https://only-this-one.example.invalid/*"]
        fake = RecordingKeycloak([live])
        monkeypatch.setattr(seeder, "_request", fake.request)

        record = seeder._preflight_client(KC_URL, REALM, "admin-token", _spec())

        assert record["drift_fields"] == ["redirectUris"]
