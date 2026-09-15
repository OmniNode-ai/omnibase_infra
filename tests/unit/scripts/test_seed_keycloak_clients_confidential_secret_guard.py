# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The reconciler may not destroy a confidential client's secret (OMN-16504).

``scripts/seed-keycloak-clients.py`` used to answer any declared-field drift
with a PUT of the whole live client representation. Keycloak's client update
applies every non-null field it is handed, and a representation that marks the
client ``bearerOnly`` (or ``publicClient``) makes it clear the stored client
secret as a side effect. So a drift on one unrelated declared field was enough
to empty the secret of a client nobody asked to change.

That happened. omninode_infra ``b88ae5c1`` (2026-09-10) added
``"fullScopeAllowed": false`` to the ``onex-api`` entry of
``desired-clients.json``. Keycloak's default is ``true``, so the next reconcile
saw drift on exactly that field, carried ``bearerOnly: true`` forward, and the
realm emptied the secret of the realm's own introspection caller. Introspection
POSTs returned 401 from then on. Nothing went red: every subsequent reconcile
logged ``op=unchanged`` and the Job kept exiting 0.

Two properties are pinned here, and they are layered rather than redundant:

1. **The update path cannot destroy incidentally.** A drift on a declared field
   sends that field and nothing else, so an unrelated change can no longer
   reach a secret-clearing field.
2. **The run cannot end silently destroyed.** If a confidential live client is
   empty at the end of a reconcile, the Job exits non-zero and names it. Only
   ``publicClient`` exempts a client. Bearer-only clients are NOT exempt: this
   realm's ``onex-api`` is bearer-only and is the client the runtime presents
   as HTTP Basic auth on every introspection POST, so exempting bearer-only
   would make this guard skip the one client it exists to catch.

Every Keycloak interaction here is against the in-process fake below. The
secret-clearing behaviour is reproduced in the fake and is never exercised
against a live realm.

Related Tickets:
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


class FakeKeycloak:
    """A Keycloak admin API just complete enough to reproduce the defect.

    Models the one behaviour that matters: an update whose representation
    asserts ``bearerOnly`` (or ``publicClient``) leaves the client with no
    secret. Everything else is a plain merge over non-null fields, which is
    what the real server does.
    """

    def __init__(self, clients: list[dict[str, Any]]) -> None:
        self.clients: list[dict[str, Any]] = [dict(c) for c in clients]
        self.put_payloads: list[dict[str, Any]] = []
        self.secret_endpoint_404_for: set[str] = set()

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

        if method == "GET" and path.startswith(f"{base}?clientId="):
            wanted = path.split("clientId=", 1)[1]
            return 200, [c for c in self.clients if c["clientId"] == wanted]

        if method == "GET" and path.endswith("/client-secret"):
            client = self._find(path[len(base) + 1 : -len("/client-secret")])
            secret = client.get("secret") or ""
            if client["clientId"] in self.secret_endpoint_404_for:
                return 404, {"error": "Client secret endpoint unavailable"}
            # Keycloak answers 404 for a client that cannot hold a secret.
            if not secret and (client.get("publicClient") or client.get("bearerOnly")):
                return 404, {"error": "Client is not confidential"}
            return 200, {"type": "secret", "value": secret}

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
            # The destruction under test: Keycloak drops the secret of a
            # client an update declares bearer-only or public.
            asserts_secretless = payload.get("bearerOnly") is True or (
                payload.get("publicClient") is True
            )
            if asserts_secretless:
                client["secret"] = ""
            return 204, None

        raise AssertionError(f"fake received an unmodelled call: {method} {path}")


def _onex_api_live() -> dict[str, Any]:
    """The live onex-api client as it stood before b88ae5c1's reconcile.

    Confidential, bearer-only, holding a secret, and -- the point -- with
    ``fullScopeAllowed`` at Keycloak's default of ``true``.
    """
    return {
        "id": "internal-onex-api",
        "clientId": "onex-api",
        "publicClient": False,
        "bearerOnly": True,
        "fullScopeAllowed": True,
        "secret": "a-live-client-secret",  # pragma: allowlist secret
    }


def _onex_api_spec() -> dict[str, Any]:
    """The roster entry b88ae5c1 produced: one newly declared field."""
    return {
        "clientId": "onex-api",
        "publicClient": False,
        "bearerOnly": True,
        "fullScopeAllowed": False,
    }


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake: FakeKeycloak,
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


@pytest.mark.unit
class TestDriftedUpdateDoesNotCarryTheWholeRepresentation:
    """Property 1: a declared-field drift sends that field and nothing else."""

    def test_b88ae5c1_shape_sends_only_the_drifted_field(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake = FakeKeycloak([_onex_api_live()])
        _run_main(monkeypatch, tmp_path, fake, [_onex_api_spec()])

        assert len(fake.put_payloads) == 1, "expected exactly one update PUT"
        payload = fake.put_payloads[0]
        assert set(payload) == {"clientId", "fullScopeAllowed"}, (
            "the update must carry only the drifted field; carrying the live "
            "representation is what re-asserted bearerOnly and emptied the "
            f"secret. Got: {sorted(payload)}"
        )
        assert "id" not in payload
        assert "bearerOnly" not in payload
        assert "secret" not in payload

    def test_the_secret_survives_an_unrelated_declared_field_drift(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The whole incident, replayed end to end, must now be a no-op."""
        fake = FakeKeycloak([_onex_api_live()])
        _run_main(monkeypatch, tmp_path, fake, [_onex_api_spec()])

        client = fake._find("internal-onex-api")
        assert client["fullScopeAllowed"] is False, "the declared change must apply"
        assert client["secret"], (
            "onex-api ended the reconcile with no client secret -- this is the "
            "OMN-16504 destruction"
        )


@pytest.mark.unit
class TestEmptyConfidentialSecretFailsTheJob:
    """Property 2: an empty confidential secret ends the run non-zero."""

    def test_job_fails_red_and_names_the_client(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
    ) -> None:
        live = {
            "id": "internal-onex-service",
            "clientId": "onex-service",
            "publicClient": False,
            "bearerOnly": False,
            "serviceAccountsEnabled": True,
            "fullScopeAllowed": False,
            "secret": "",
        }
        spec = {
            "clientId": "onex-service",
            "publicClient": False,
            "serviceAccountsEnabled": True,
            "fullScopeAllowed": False,
        }
        fake = FakeKeycloak([live])

        with pytest.raises(SystemExit) as excinfo:
            _run_main(monkeypatch, tmp_path, fake, [spec])

        assert excinfo.value.code == 1, "an emptied confidential secret must fail red"
        stderr = capsys.readouterr().err
        record = json.loads(stderr.strip().splitlines()[-1])
        assert record["op"] == "error"
        assert record["check"] == "confidential_client_secret_present"
        assert record["clients"] == ["onex-service"], (
            "the failure must name the offending client; a redacted failure "
            "leaves an operator exactly where this check was added to help"
        )

    def test_a_confidential_client_that_was_already_empty_fails_red(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No drift, no PUT, nothing changed -- and still red.

        A broken secret-authenticating client must fail even when no declared
        field drifted and the reconcile loop would otherwise log unchanged.
        """
        live = {
            "id": "internal-onex-service",
            "clientId": "onex-service",
            "publicClient": False,
            "bearerOnly": False,
            "serviceAccountsEnabled": True,
            "fullScopeAllowed": False,
            "secret": "",
        }
        spec = {
            "clientId": "onex-service",
            "publicClient": False,
            "serviceAccountsEnabled": True,
            "fullScopeAllowed": False,
        }
        fake = FakeKeycloak([live])

        with pytest.raises(SystemExit) as excinfo:
            _run_main(monkeypatch, tmp_path, fake, [spec])

        assert excinfo.value.code == 1
        assert fake.put_payloads == [], "nothing drifted, so nothing should be written"

    def test_bearer_only_introspection_caller_with_no_secret_fails_red(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
    ) -> None:
        """The live state of both clusters, and the reason this ticket exists.

        ``onex-api`` is bearer-only, holds no secret, and is the client the
        runtime authenticates as on every introspection POST. Keycloak's
        textbook model says a bearer-only client never authenticates outbound
        and so needs no secret; this realm does not follow that model, and a
        guard that took Keycloak's word for it would skip the exact client
        whose emptied secret is the outage.

        Exempting bearer-only clients makes this test pass silently while
        introspection returns 401. That is the failure mode, not a fix for it.
        """
        live = _onex_api_live()
        live["fullScopeAllowed"] = False
        live["secret"] = ""
        fake = FakeKeycloak([live])

        with pytest.raises(SystemExit) as excinfo:
            _run_main(monkeypatch, tmp_path, fake, [_onex_api_spec()])

        assert excinfo.value.code == 1
        record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert record["clients"] == ["onex-api"]
        assert fake.put_payloads == [], "nothing drifted, so nothing should be written"


@pytest.mark.unit
class TestPublicClientsAreNotTrippedByTheGuard:
    """Positive control: a public client legitimately has no secret."""

    def test_public_client_with_no_secret_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        live = {
            "id": "internal-omnidash-spa",
            "clientId": "omnidash-spa",
            "publicClient": True,
            "fullScopeAllowed": True,
            "secret": "",
        }
        spec = {
            "clientId": "omnidash-spa",
            "publicClient": True,
            "fullScopeAllowed": False,
        }
        fake = FakeKeycloak([live])

        # No SystemExit: a public client holding no secret is correct, and a
        # guard that failed on it would be unlandable against the real roster,
        # which declares two of them.
        _run_main(monkeypatch, tmp_path, fake, [spec])

        assert fake._find("internal-omnidash-spa")["fullScopeAllowed"] is False

    def test_a_healthy_confidential_client_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The other half of the control: the guard is not simply always-red."""
        live = {
            "id": "internal-onex-service",
            "clientId": "onex-service",
            "publicClient": False,
            "fullScopeAllowed": True,
            "secret": "a-live-client-secret",  # pragma: allowlist secret
        }
        spec = {
            "clientId": "onex-service",
            "publicClient": False,
            "fullScopeAllowed": False,
        }
        fake = FakeKeycloak([live])

        _run_main(monkeypatch, tmp_path, fake, [spec])

        assert fake._find("internal-onex-service")["secret"]

    def test_secret_endpoint_404_falls_back_to_live_representation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        live = {
            "id": "internal-onex-service",
            "clientId": "onex-service",
            "publicClient": False,
            "bearerOnly": False,
            "serviceAccountsEnabled": True,
            "fullScopeAllowed": False,
            "secret": "a-live-client-secret",  # pragma: allowlist secret
        }
        spec = {
            "clientId": "onex-service",
            "publicClient": False,
            "serviceAccountsEnabled": True,
            "fullScopeAllowed": False,
        }
        fake = FakeKeycloak([live])
        fake.secret_endpoint_404_for.add("onex-service")

        _run_main(monkeypatch, tmp_path, fake, [spec])
