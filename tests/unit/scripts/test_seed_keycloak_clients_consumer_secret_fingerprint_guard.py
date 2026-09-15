# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The confidential-secret guard must catch a Keycloak-minted secret no consumer holds (OMN-18391).

``omnibase_infra#3550`` (OMN-16504) added a guard that fails the seed Job red
when a confidential client ends a reconcile with an empty secret. That guard
only checks non-emptiness. When the desired-clients roster flipped the
``onex-api`` client on onex-dev to confidential with no ``secretEnv``
declared, Keycloak minted its own secret -- non-empty, and no consumer held
it. The presence guard reported green (``op=updated
fields_changed=[bearerOnly]``) while every consumer's introspection call
returned 401.

This guard compares the client's live Keycloak secret against the value the
consuming k8s Secret holds, by fingerprint only (sha256-12 plus length). It
never reads, logs, or compares the raw secret value in any assertion or log
line.

Three cases, kept mechanically distinct so a single change cannot silently
merge one class of failure into another:

1. Empty live secret -- handled by the existing OMN-16504 presence guard
   (``confidential_client_secret_present``), not this one.
2. Non-empty live secret whose fingerprint does not match the consumer's
   copy -- a NEW check (``client_secret_fingerprint_mismatch``), which is the
   OMN-18391 defect itself: Keycloak-minted-but-unconsumed.
3. Matching fingerprints -- passes, no error from either guard.

Related Tickets:
    - OMN-18391: guard passes on a Keycloak-minted secret no consumer holds
    - OMN-16504: onex-api ends a reconcile with no client secret; introspection 401
"""

from __future__ import annotations

import hashlib
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

CONSUMER_ENV_VAR = "ONEX_API_CONSUMER_CLIENT_SECRET"  # pragma: allowlist secret


def _load_seeder() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("_seed_keycloak_clients", SEED_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeKeycloak:
    """Just enough of the admin API to serve GET client / GET client-secret."""

    def __init__(self, clients: list[dict[str, Any]]) -> None:
        self.clients: list[dict[str, Any]] = [dict(c) for c in clients]

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
            live_value = client.get("secret")
            resolved_value = live_value if live_value else ""
            if not resolved_value:
                return 404, {"error": "Client is not confidential"}
            return 200, {"type": "secret", "value": resolved_value}

        if method == "GET" and path.startswith(f"{base}/"):
            return 200, dict(self._find(path[len(base) + 1 :]))

        raise AssertionError(f"fake received an unmodelled call: {method} {path}")


def _onex_api_client_id() -> str:
    return "onex-api"


def _onex_api_spec(*, consumer_env: str | None = CONSUMER_ENV_VAR) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "clientId": _onex_api_client_id(),
        "publicClient": False,
        "bearerOnly": False,
        "serviceAccountsEnabled": True,
    }
    if consumer_env is not None:
        spec["consumerSecretEnv"] = consumer_env
    return spec


def _run_guard(
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


def _fingerprint(value: str) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{digest}/{len(value)}"


@pytest.mark.unit
class TestConsumerSecretFingerprintGuard:
    def test_keycloak_minted_mismatch_fails_red_naming_fingerprints_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: Any,
    ) -> None:
        """The OMN-18391 defect itself: onex-api flipped confidential, Keycloak
        minted a value, and no consumer holds it -- the two values differ."""
        live_value = "kc-minted-573229082147-value-not-held-anywhere-else"  # pragma: allowlist secret
        consumer_value = (
            "old-stale-consumer-held-9b676d4cc11b-value"  # pragma: allowlist secret
        )
        monkeypatch.setenv(CONSUMER_ENV_VAR, consumer_value)

        live = {
            "id": "internal-onex-api",
            "clientId": _onex_api_client_id(),
            "publicClient": False,
            "bearerOnly": False,
            "serviceAccountsEnabled": True,
            "secret": live_value,
        }
        fake = FakeKeycloak([live])

        with pytest.raises(SystemExit) as excinfo:
            _run_guard(monkeypatch, tmp_path, fake, [_onex_api_spec()])

        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        last_line = captured.err.strip().splitlines()[-1]
        record = json.loads(last_line)
        assert record["op"] == "error"
        assert record["check"] == "client_secret_fingerprint_mismatch"
        assert record["clients"] == ["onex-api"]

        rendered = json.dumps(record)
        assert live_value not in rendered, "the live value must never be printed"
        assert consumer_value not in rendered, (
            "the consumer value must never be printed"
        )

        assert _fingerprint(live_value) in rendered, (
            "the live fingerprint must be named"
        )
        assert _fingerprint(consumer_value) in rendered, (
            "the consumer fingerprint must be named"
        )

    def test_matching_fingerprints_pass(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The restored state: live and consumer hold the identical value."""
        shared_value = (
            "the-one-true-restored-value-both-sides-hold"  # pragma: allowlist secret
        )
        monkeypatch.setenv(CONSUMER_ENV_VAR, shared_value)

        live = {
            "id": "internal-onex-api",
            "clientId": _onex_api_client_id(),
            "publicClient": False,
            "bearerOnly": False,
            "serviceAccountsEnabled": True,
            "secret": shared_value,
        }
        fake = FakeKeycloak([live])

        # No SystemExit: matching fingerprints must not fail the Job.
        _run_guard(monkeypatch, tmp_path, fake, [_onex_api_spec()])

    def test_empty_live_value_is_not_conflated_with_a_mismatch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
    ) -> None:
        """An empty value must still fail via the OMN-16504 presence guard,
        never as a fingerprint mismatch -- the two failure classes must stay
        distinguishable in the Job's log."""
        consumer_value = "some-consumer-held-value"  # pragma: allowlist secret
        monkeypatch.setenv(CONSUMER_ENV_VAR, consumer_value)

        live = {
            "id": "internal-onex-api",
            "clientId": _onex_api_client_id(),
            "publicClient": False,
            "bearerOnly": False,
            "serviceAccountsEnabled": True,
            "secret": "",
        }
        fake = FakeKeycloak([live])

        with pytest.raises(SystemExit) as excinfo:
            _run_guard(monkeypatch, tmp_path, fake, [_onex_api_spec()])

        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        last_line = captured.err.strip().splitlines()[-1]
        record = json.loads(last_line)
        assert record["check"] == "confidential_client_secret_present", (
            "an empty value must fail the presence guard, not the fingerprint "
            "mismatch guard -- conflating the two hides which failure occurred"
        )

    def test_no_consumer_env_declared_skips_the_check(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A client with no consumerSecretEnv declared is not checked by this
        guard at all -- unrelated clients must not start failing."""
        live = {
            "id": "internal-onex-api",
            "clientId": _onex_api_client_id(),
            "publicClient": False,
            "bearerOnly": False,
            "serviceAccountsEnabled": True,
            "secret": "whatever-value",  # pragma: allowlist secret
        }
        fake = FakeKeycloak([live])

        _run_guard(monkeypatch, tmp_path, fake, [_onex_api_spec(consumer_env=None)])

    def test_declared_but_unset_consumer_env_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A declared consumerSecretEnv that is unset must fail the Job rather
        than silently skip the check -- same fail-closed contract as
        _resolve_client_secret."""
        monkeypatch.delenv(CONSUMER_ENV_VAR, raising=False)

        live = {
            "id": "internal-onex-api",
            "clientId": _onex_api_client_id(),
            "publicClient": False,
            "bearerOnly": False,
            "serviceAccountsEnabled": True,
            "secret": "whatever-value",  # pragma: allowlist secret
        }
        fake = FakeKeycloak([live])

        with pytest.raises(SystemExit) as excinfo:
            _run_guard(monkeypatch, tmp_path, fake, [_onex_api_spec()])

        assert excinfo.value.code == 1
