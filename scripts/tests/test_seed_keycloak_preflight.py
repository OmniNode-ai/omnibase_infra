#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the read-only client preflight in seed-keycloak-clients.py.

OMN-18170. Before this, the reconcile's ordering made two real defects on the
`.201` dev lane unreportable:

* ``_resolve_client_secret`` ``_die()``s on the FIRST roster entry whose
  ``secretEnv`` is unset, so a run named one absent variable out of however
  many were absent, and an operator learned about them one re-run at a time;
* both secret guards (``_assert_confidential_clients_have_secrets`` and
  ``_assert_client_secrets_match_consumers``) run AFTER the client loop, so a
  run that died mid-loop never reached either. The live ``onex-api`` client on
  that lane already holds an EMPTY secret with declared drift on
  ``bearerOnly``; the loop would have sent that drift -- which is what makes
  Keycloak clear a secret -- and then exited on a later entry, before the
  guard that reports it.

The fix is a read-only preflight over every roster entry, run BEFORE any
client mutation: it names every refusal at once and refuses before the first
write, so the partial-application hazard does not exist rather than being
merely reported after the fact.

Uses the same module-loading / ``_request``-mock harness as the sibling test
files so every file exercises the same script instance.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import urllib.parse
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_SCRIPT_PATH = Path(__file__).parent.parent / "seed-keycloak-clients.py"
_mod: types.ModuleType  # assigned by _ensure_mod() on first use


def _ensure_mod() -> types.ModuleType:
    global _mod  # noqa: PLW0603
    try:
        return _mod
    except NameError:
        pass
    spec = importlib.util.spec_from_file_location("seed_keycloak_clients", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["seed_keycloak_clients"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _mod = mod
    return _mod


@pytest.fixture(autouse=True, scope="session")
def _load_module() -> None:
    _ensure_mod()


_REALM = "omninode"
_KC_URL = "http://localhost:28080"
_TOKEN = "test-access-token"


def _fake_realm(
    live: dict[str, dict[str, Any]],
    secrets: dict[str, str] | None = None,
    writes: list[tuple[str, str]] | None = None,
) -> Any:
    """Build a ``_request`` stub over a dict of live clients.

    ``live`` maps clientId -> client representation (an ``id`` is filled in if
    absent). ``secrets`` maps clientId -> its live secret value; a clientId
    absent from that dict reads back as having no secret. Every non-GET is
    appended to ``writes`` so a test can assert that nothing was mutated.
    """
    secrets = secrets or {}
    for client_id, rep in live.items():
        rep.setdefault("id", f"uuid-{client_id}")
        rep.setdefault("clientId", client_id)

    by_uuid = {rep["id"]: (client_id, rep) for client_id, rep in live.items()}

    def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
        if method != "GET":
            if writes is not None:
                writes.append((method, url))
            return (204, None)

        base = f"{_KC_URL}/admin/realms/{_REALM}/clients"
        if url.startswith(f"{base}?clientId="):
            wanted = urllib.parse.unquote(url.split("clientId=", 1)[1])
            rep = live.get(wanted)
            return (200, [rep] if rep else [])

        if url.startswith(f"{base}/"):
            tail = url[len(base) + 1 :]
            if tail.endswith("/client-secret"):
                uuid = tail[: -len("/client-secret")]
                client_id, _rep = by_uuid.get(uuid, (None, None))
                value = secrets.get(client_id or "")
                # A client shape with no dedicated secret endpoint 404s, which
                # is why the reader falls back to the representation.
                return (200, {"value": value}) if value else (404, None)
            client_id, rep = by_uuid.get(tail, (None, None))
            if rep is None:
                return (404, None)
            full = dict(rep)
            if secrets.get(client_id or ""):
                full["secret"] = secrets[client_id or ""]
            return (200, full)

        raise AssertionError(f"unexpected GET: {url}")

    return fake_request


def _records_by_client(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["clientId"]: r for r in records}


def _finding_codes(record: dict[str, Any]) -> set[str]:
    return {f["code"] for f in record["findings"]}


# ---------------------------------------------------------------------------
# (b) every roster entry is evaluated; every refusal is named; one exit
# ---------------------------------------------------------------------------


class TestPreflightEvaluatesEveryRosterEntry:
    def test_every_absent_roster_secret_is_named_not_just_the_first(self) -> None:
        """The defect: the run died on entry 0 and never looked at entry 1.

        An operator provisioning credentials needs the whole list in one run,
        not one name per re-run.
        """
        clients = [
            {"clientId": "alpha", "secretEnv": "ALPHA_SECRET"},
            {"clientId": "beta", "secretEnv": "BETA_SECRET"},
            {"clientId": "gamma", "secretEnv": "GAMMA_SECRET"},
        ]
        live = {
            "alpha": {"publicClient": False},
            "beta": {"publicClient": False},
            "gamma": {"publicClient": False},
        }
        secrets = {"alpha": "a", "beta": "b", "gamma": "g"}

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(
                _ensure_mod(), "_request", side_effect=_fake_realm(live, secrets)
            ):
                records = _ensure_mod()._preflight_clients(
                    _KC_URL, _REALM, _TOKEN, clients
                )

        by_client = _records_by_client(records)
        assert set(by_client) == {"alpha", "beta", "gamma"}
        for client_id in ("alpha", "beta", "gamma"):
            assert "roster_secret_absent" in _finding_codes(by_client[client_id])

    def test_each_refusal_names_the_env_var_the_operator_must_seed(self) -> None:
        clients = [
            {"clientId": "alpha", "secretEnv": "ALPHA_SECRET"},
            {"clientId": "beta", "secretEnv": "BETA_SECRET"},
        ]
        live = {"alpha": {"publicClient": False}, "beta": {"publicClient": False}}

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(
                _ensure_mod(),
                "_request",
                side_effect=_fake_realm(live, {"alpha": "a", "beta": "b"}),
            ):
                records = _ensure_mod()._preflight_clients(
                    _KC_URL, _REALM, _TOKEN, clients
                )

        by_client = _records_by_client(records)
        assert by_client["alpha"]["secret_env"] == "ALPHA_SECRET"
        assert by_client["beta"]["secret_env"] == "BETA_SECRET"
        keys = {f["key"] for r in records for f in r["findings"] if "key" in f}
        assert keys == {"ALPHA_SECRET", "BETA_SECRET"}

    def test_refusal_exits_non_zero_once_naming_every_client_and_key(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        clients = [
            {"clientId": "alpha", "secretEnv": "ALPHA_SECRET"},
            {"clientId": "beta", "secretEnv": "BETA_SECRET"},
        ]
        live = {"alpha": {"publicClient": False}, "beta": {"publicClient": False}}

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(
                _ensure_mod(),
                "_request",
                side_effect=_fake_realm(live, {"alpha": "a", "beta": "b"}),
            ):
                records = _ensure_mod()._preflight_clients(
                    _KC_URL, _REALM, _TOKEN, clients
                )
                with pytest.raises(SystemExit) as exc_info:
                    _ensure_mod()._die_preflight(records)

        assert exc_info.value.code != 0
        record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert record["op"] == "error"
        assert record["check"] == "client_preflight"
        assert sorted(record["clients"]) == ["alpha", "beta"]
        assert sorted(record["keys"]) == ["ALPHA_SECRET", "BETA_SECRET"]


# ---------------------------------------------------------------------------
# (d) roster-pushed-and-absent is a DIFFERENT fact from Keycloak-minted
# ---------------------------------------------------------------------------


class TestPreflightDistinguishesSecretSources:
    def test_roster_pushed_absent_says_seed_required(self) -> None:
        """What the operator has to do about it is the point of the message.

        `omniweb-signup-admin` declares `secretEnv`, so the roster PUSHES its
        secret and a human has to seed the value. That is not the same fact as
        a client Keycloak mints for itself, which needs nothing.
        """
        clients = [{"clientId": "signup", "secretEnv": "SIGNUP_SECRET"}]
        live = {"signup": {"publicClient": False}}

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(
                _ensure_mod(),
                "_request",
                side_effect=_fake_realm(live, {"signup": "s"}),
            ):
                records = _ensure_mod()._preflight_clients(
                    _KC_URL, _REALM, _TOKEN, clients
                )

        record = _records_by_client(records)["signup"]
        assert record["secret_source"] == "roster_pushed"
        finding = next(
            f for f in record["findings"] if f["code"] == "roster_secret_absent"
        )
        assert finding["detail"] == "absent, roster-pushed, seed required"

    def test_keycloak_minted_client_is_classified_and_is_not_a_refusal(self) -> None:
        """`onex-api`'s shape: confidential, no roster secret. Nothing to seed."""
        clients = [{"clientId": "minted", "bearerOnly": False}]
        live = {"minted": {"publicClient": False, "bearerOnly": False}}

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(
                _ensure_mod(),
                "_request",
                side_effect=_fake_realm(live, {"minted": "m"}),
            ):
                records = _ensure_mod()._preflight_clients(
                    _KC_URL, _REALM, _TOKEN, clients
                )

        record = _records_by_client(records)["minted"]
        assert record["secret_source"] == "keycloak_minted"
        assert record["secret_env"] is None
        assert record["findings"] == []

    def test_public_client_is_classified_and_needs_no_secret(self) -> None:
        """The positive control: a client that SHOULD hold no secret."""
        clients = [{"clientId": "spa", "publicClient": True}]
        live = {"spa": {"publicClient": True}}

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(_ensure_mod(), "_request", side_effect=_fake_realm(live)):
                records = _ensure_mod()._preflight_clients(
                    _KC_URL, _REALM, _TOKEN, clients
                )

        record = _records_by_client(records)["spa"]
        assert record["secret_source"] == "public"
        assert record["findings"] == []

    def test_consumer_owned_absent_env_is_its_own_refusal_code(self) -> None:
        clients = [{"clientId": "api", "consumerSecretEnv": "API_CONSUMER_SECRET"}]
        live = {"api": {"publicClient": False}}

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(
                _ensure_mod(), "_request", side_effect=_fake_realm(live, {"api": "x"})
            ):
                records = _ensure_mod()._preflight_clients(
                    _KC_URL, _REALM, _TOKEN, clients
                )

        record = _records_by_client(records)["api"]
        assert record["secret_source"] == "consumer_owned"
        assert "consumer_secret_absent" in _finding_codes(record)
        assert "roster_secret_absent" not in _finding_codes(record)


# ---------------------------------------------------------------------------
# (c) an existing client with an empty live secret is a NAMED refusal
# ---------------------------------------------------------------------------


class TestPreflightRefusesEmptyLiveSecret:
    def test_existing_confidential_client_with_empty_secret_is_named(self) -> None:
        """The live `.201` condition, reported before anything is written.

        The presence guard already catches this -- but only AFTER the client
        loop, which on that lane never completes. Reporting it first is what
        makes it visible at all.
        """
        clients = [{"clientId": "api", "bearerOnly": False}]
        live = {"api": {"publicClient": False, "bearerOnly": True}}

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(_ensure_mod(), "_request", side_effect=_fake_realm(live)):
                records = _ensure_mod()._preflight_clients(
                    _KC_URL, _REALM, _TOKEN, clients
                )

        record = _records_by_client(records)["api"]
        assert record["present"] is True
        assert record["live_secret_present"] is False
        assert "live_secret_empty" in _finding_codes(record)

    def test_an_absent_client_is_not_a_refusal_it_will_be_created(self) -> None:
        """A first-ever reconcile on a fresh realm must not be refused.

        This is the boundary the post-loop presence guard gets to draw the
        other way: after the loop, a still-absent client IS an offender.
        Before the loop it is just work to do.
        """
        clients = [{"clientId": "fresh", "secretEnv": "FRESH_SECRET"}]

        with patch.dict("os.environ", {"FRESH_SECRET": "v"}, clear=True):
            with patch.object(_ensure_mod(), "_request", side_effect=_fake_realm({})):
                records = _ensure_mod()._preflight_clients(
                    _KC_URL, _REALM, _TOKEN, clients
                )

        record = _records_by_client(records)["fresh"]
        assert record["present"] is False
        assert record["findings"] == []

    def test_an_empty_secret_value_is_never_written_as_a_client_secret(self) -> None:
        """The other half of (c): refuse, never overwrite with emptiness.

        An empty env var must not reach Keycloak as the client's new secret;
        `_resolve_client_secret` treats empty exactly as absent.
        """
        spec = {"clientId": "alpha", "secretEnv": "ALPHA_SECRET"}
        with patch.dict("os.environ", {"ALPHA_SECRET": ""}, clear=True):
            with pytest.raises(SystemExit):
                _ensure_mod()._resolve_client_secret(spec)

        payload = _ensure_mod()._build_update_payload(
            {"id": "uuid-alpha", "clientId": "alpha", "bearerOnly": True},
            {"clientId": "alpha", "bearerOnly": False},
            ["bearerOnly"],
            None,
        )
        assert "secret" not in payload


# ---------------------------------------------------------------------------
# (a) the read-only guards run BEFORE any mutation, and still print
# ---------------------------------------------------------------------------


class TestPreflightRunsBeforeAnyMutation:
    def test_preflight_issues_no_write_request(self) -> None:
        clients = [
            {"clientId": "alpha", "secretEnv": "ALPHA_SECRET", "bearerOnly": False},
            {"clientId": "api", "consumerSecretEnv": "API_CONSUMER_SECRET"},
        ]
        live = {
            "alpha": {"publicClient": False, "bearerOnly": True},
            "api": {"publicClient": False},
        }
        writes: list[tuple[str, str]] = []

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(
                _ensure_mod(),
                "_request",
                side_effect=_fake_realm(live, {"alpha": "a", "api": "x"}, writes),
            ):
                _ensure_mod()._preflight_clients(_KC_URL, _REALM, _TOKEN, clients)

        assert writes == []

    def test_drift_is_reported_alongside_a_refusal_not_instead_of_it(self) -> None:
        """(a): findings print even when a later step will refuse.

        `onex-api`'s drift is the fact that explains WHY its secret is empty.
        A run that refuses on a different client's absent credential must
        still have said so.
        """
        clients = [
            {"clientId": "signup", "secretEnv": "SIGNUP_SECRET"},
            {
                "clientId": "api",
                "bearerOnly": False,
                "fullScopeAllowed": False,
                "serviceAccountsEnabled": True,
            },
        ]
        live = {
            "signup": {"publicClient": False},
            "api": {
                "publicClient": False,
                "bearerOnly": True,
                "fullScopeAllowed": True,
                "serviceAccountsEnabled": False,
            },
        }

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(
                _ensure_mod(),
                "_request",
                side_effect=_fake_realm(live, {"signup": "s", "api": "a"}),
            ):
                records = _ensure_mod()._preflight_clients(
                    _KC_URL, _REALM, _TOKEN, clients
                )

        by_client = _records_by_client(records)
        assert "roster_secret_absent" in _finding_codes(by_client["signup"])
        assert sorted(by_client["api"]["drift_fields"]) == [
            "bearerOnly",
            "fullScopeAllowed",
            "serviceAccountsEnabled",
        ]

    def test_consumer_fingerprint_mismatch_is_reported_by_the_preflight(self) -> None:
        clients = [{"clientId": "api", "consumerSecretEnv": "API_CONSUMER_SECRET"}]
        live = {"api": {"publicClient": False}}

        with patch.dict(
            "os.environ", {"API_CONSUMER_SECRET": "what-the-consumer-holds"}, clear=True
        ):
            with patch.object(
                _ensure_mod(),
                "_request",
                side_effect=_fake_realm(live, {"api": "what-keycloak-holds"}),
            ):
                records = _ensure_mod()._preflight_clients(
                    _KC_URL, _REALM, _TOKEN, clients
                )

        record = _records_by_client(records)["api"]
        finding = next(
            f
            for f in record["findings"]
            if f["code"] == "consumer_secret_fingerprint_mismatch"
        )
        # Fingerprints only -- neither value may appear anywhere in the record.
        blob = json.dumps(records)
        assert "what-the-consumer-holds" not in blob
        assert "what-keycloak-holds" not in blob
        assert finding["live"] != finding["consumer"]


# ---------------------------------------------------------------------------
# The positive control: a clean roster preflights clean
# ---------------------------------------------------------------------------


class TestPreflightPassesACleanRoster:
    def test_clean_roster_yields_no_findings(self) -> None:
        clients = [
            {"clientId": "alpha", "secretEnv": "ALPHA_SECRET"},
            {"clientId": "spa", "publicClient": True},
            {"clientId": "api", "consumerSecretEnv": "API_CONSUMER_SECRET"},
        ]
        live = {
            "alpha": {"publicClient": False},
            "spa": {"publicClient": True},
            "api": {"publicClient": False},
        }
        env = {"ALPHA_SECRET": "a", "API_CONSUMER_SECRET": "shared"}

        with patch.dict("os.environ", env, clear=True):
            with patch.object(
                _ensure_mod(),
                "_request",
                side_effect=_fake_realm(live, {"alpha": "a", "api": "shared"}),
            ):
                records = _ensure_mod()._preflight_clients(
                    _KC_URL, _REALM, _TOKEN, clients
                )

        assert all(r["findings"] == [] for r in records), records
        assert _ensure_mod()._preflight_refusals(records) == []
