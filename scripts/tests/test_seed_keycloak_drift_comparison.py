#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Drift comparison for seed-keycloak-clients.py (OMN-18600).

`omniweb`, `omnidash-spa` and `onex-customer` reported `redirectUris`,
`webOrigins` and `attributes` as drifted on EVERY reconcile of the `.201` dev
lane, including runs immediately following a clean one. Measured against that
live realm on 2026-09-17: the two collections are equal as sets and unequal as
lists, because Keycloak returns them in its own order, and the attribute maps
differ only by `realm_client`, which Keycloak adds itself, with zero differing
values.

The comparison was `existing.get(field) != spec[field]` -- order-sensitive for
lists, whole-map for dicts -- so all three compared unequal forever.

This is not cosmetic. Any drift at all sends the client down the
full-representation update path, and that is the path whose secret-clearing
side effect destroyed `onex-api`'s secret under OMN-16504. Three clients took
the most dangerous path in this script on every run for no reason.

The fix must not loosen the comparison into blindness, which is what the
fail-closed tests at the bottom of this file exist to prevent: a live-only
attribute key that is NOT declared server-managed is still drift, and every
field that differs in membership rather than order is still drift and is still
applied.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import urllib.parse
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_SCRIPT_PATH = Path(__file__).parent.parent / "seed-keycloak-clients.py"
_mod: types.ModuleType


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

# A non-secret stand-in for whatever the live client's secret is. The drift
# tests never compare it; the reader just has to return something non-empty so
# the presence check does not refuse.
_STUB_LIVE_SECRET = "stub-value-not-a-credential"

# The live `.201` shapes, transcribed from the 2026-09-17 read-only probe.
_OMNIWEB_SPEC: dict[str, Any] = {
    "clientId": "omniweb",
    "redirectUris": [
        "https://app.omninode.ai/*",
        "https://dev.app.omninode.ai/*",
        "http://localhost:3000/*",
        "http://localhost:8080/*",
    ],
    "webOrigins": [
        "https://app.omninode.ai",
        "https://omninode.ai",
        "http://localhost:3000",
    ],
    "attributes": {"post.logout.redirect.uris": "+"},
}


def _live_omniweb_permuted() -> dict[str, Any]:
    """What Keycloak actually returns: same members, its own order, plus its
    own server-managed attribute key."""
    return {
        "id": "uuid-omniweb",
        "clientId": "omniweb",
        "redirectUris": [
            "https://dev.app.omninode.ai/*",
            "http://localhost:8080/*",
            "http://localhost:3000/*",
            "https://app.omninode.ai/*",
        ],
        "webOrigins": [
            "http://localhost:3000",
            "https://app.omninode.ai",
            "https://omninode.ai",
        ],
        "attributes": {
            "post.logout.redirect.uris": "+",
            "realm_client": "false",
        },
    }


def _drift(spec: dict[str, Any], live: dict[str, Any]) -> list[str]:
    return _ensure_mod()._drifted_fields(spec, live)


# ---------------------------------------------------------------------------
# AC1 / AC2 — the spurious drift is gone
# ---------------------------------------------------------------------------


class TestPermutedCollectionsAreNotDrift:
    def test_shuffled_redirect_uris_are_not_drift(self) -> None:
        assert "redirectUris" not in _drift(_OMNIWEB_SPEC, _live_omniweb_permuted())

    def test_shuffled_web_origins_are_not_drift(self) -> None:
        assert "webOrigins" not in _drift(_OMNIWEB_SPEC, _live_omniweb_permuted())

    def test_server_managed_attribute_key_is_not_drift(self) -> None:
        assert "attributes" not in _drift(_OMNIWEB_SPEC, _live_omniweb_permuted())

    def test_the_live_shape_reports_no_drift_at_all(self) -> None:
        """The whole finding, in one assertion: this realm is converged."""
        assert _drift(_OMNIWEB_SPEC, _live_omniweb_permuted()) == []


# ---------------------------------------------------------------------------
# AC3 — the server-managed key set is a pinned constant
# ---------------------------------------------------------------------------


class TestServerManagedAttributeKeysArePinned:
    def test_the_constant_holds_exactly_the_keys_we_have_observed(self) -> None:
        """Widening this set is a decision, so it fails a test rather than
        passing silently. `realm_client` is the only key Keycloak was observed
        adding on the `.201` realm; the next one is a one-line change with a
        red test to prompt it.
        """
        assert (
            frozenset({"realm_client"}) == _ensure_mod()._SERVER_MANAGED_ATTRIBUTE_KEYS
        )

    def test_the_unordered_field_set_holds_exactly_the_collections_we_mean(
        self,
    ) -> None:
        assert (
            frozenset({"redirectUris", "webOrigins"})
            == _ensure_mod()._UNORDERED_COLLECTION_FIELDS
        )


# ---------------------------------------------------------------------------
# AC6 (labelled criterion) — real drift is still caught, and still applied
# ---------------------------------------------------------------------------


class TestGenuineDriftIsStillReported:
    def test_a_changed_redirect_uri_is_still_drift(self) -> None:
        live = _live_omniweb_permuted()
        live["redirectUris"] = [*live["redirectUris"][:-1], "https://evil.example/*"]
        assert "redirectUris" in _drift(_OMNIWEB_SPEC, live)

    def test_a_missing_redirect_uri_is_still_drift(self) -> None:
        live = _live_omniweb_permuted()
        live["redirectUris"] = live["redirectUris"][:-1]
        assert "redirectUris" in _drift(_OMNIWEB_SPEC, live)

    def test_a_changed_web_origin_is_still_drift(self) -> None:
        live = _live_omniweb_permuted()
        live["webOrigins"] = [*live["webOrigins"][:-1], "https://evil.example"]
        assert "webOrigins" in _drift(_OMNIWEB_SPEC, live)

    def test_a_declared_attribute_with_a_different_value_is_still_drift(self) -> None:
        live = _live_omniweb_permuted()
        live["attributes"]["post.logout.redirect.uris"] = "https://evil.example"
        assert "attributes" in _drift(_OMNIWEB_SPEC, live)

    def test_a_missing_declared_attribute_is_still_drift(self) -> None:
        live = _live_omniweb_permuted()
        del live["attributes"]["post.logout.redirect.uris"]
        assert "attributes" in _drift(_OMNIWEB_SPEC, live)

    def test_an_undeclared_live_attribute_key_is_still_drift(self) -> None:
        """Fail-closed: only the NAMED server-managed keys are forgiven.

        Ignoring every live-only key would be the blind version of this fix --
        a key someone added out of band would become invisible. It is excluded
        by name or it is drift.
        """
        live = _live_omniweb_permuted()
        live["attributes"]["someone_added_this"] = "out-of-band"
        assert "attributes" in _drift(_OMNIWEB_SPEC, live)

    def test_a_non_collection_field_still_compares_exactly(self) -> None:
        spec = {"clientId": "c", "publicClient": False, "fullScopeAllowed": False}
        live = {"id": "u", "clientId": "c", "publicClient": True}
        assert "publicClient" in _drift(spec, live)


# ---------------------------------------------------------------------------
# AC4 — a converged realm issues zero client updates
# ---------------------------------------------------------------------------


def _fake_realm(live_clients: dict[str, dict[str, Any]]) -> tuple[Any, list[str]]:
    """A `_request` stub that records every mutating call it receives."""
    writes: list[str] = []
    by_uuid = {rep["id"]: rep for rep in live_clients.values()}

    def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
        base = f"{_KC_URL}/admin/realms/{_REALM}/clients"
        if method != "GET":
            writes.append(f"{method} {url}")
            return (204, None)
        if url.startswith(f"{base}?clientId="):
            wanted = urllib.parse.unquote(url.split("clientId=", 1)[1])
            rep = live_clients.get(wanted)
            return (200, [rep] if rep else [])
        if url.startswith(f"{base}/") and url.endswith("/client-secret"):
            return (200, {"value": _STUB_LIVE_SECRET})
        if url.startswith(f"{base}/"):
            tail = url[len(base) + 1 :]
            rep = by_uuid.get(tail)
            return (200, {**rep, "secret": _STUB_LIVE_SECRET}) if rep else (404, None)
        raise AssertionError(f"unexpected GET: {url}")

    return fake_request, writes


class TestConvergedRealmIssuesNoUpdate:
    def test_reconciling_a_converged_client_writes_nothing(self) -> None:
        """The defect's operational cost, stated as a test.

        Before this, `omniweb` was PUT on every reconcile forever. The PUT path
        is the one that cleared `onex-api`'s secret under OMN-16504, so how
        often it is entered is a safety property, not a performance one.
        """
        live = {"omniweb": _live_omniweb_permuted()}
        fake_request, writes = _fake_realm(live)

        with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
            _ensure_mod()._reconcile_client(_KC_URL, _REALM, _TOKEN, _OMNIWEB_SPEC)

        assert writes == []

    def test_a_genuinely_drifted_client_is_still_updated(self) -> None:
        """The positive control for the test above."""
        drifted = _live_omniweb_permuted()
        drifted["redirectUris"] = ["https://only-this-one.example/*"]
        fake_request, writes = _fake_realm({"omniweb": drifted})

        with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
            _ensure_mod()._reconcile_client(_KC_URL, _REALM, _TOKEN, _OMNIWEB_SPEC)

        assert any(w.startswith("PUT ") for w in writes), writes


# ---------------------------------------------------------------------------
# The update payload must not erase what Keycloak owns
# ---------------------------------------------------------------------------


class TestUpdatePayloadPreservesServerManagedAttributes:
    def test_applying_attribute_drift_keeps_the_server_managed_key(self) -> None:
        """Applying the roster's attributes wholesale would drop `realm_client`.

        Keycloak re-adds it, so the old code's every-run drift was partly
        self-inflicted: PUT dropped the key, Keycloak restored it, and the next
        comparison saw a difference again. Merging keeps the loop closed even
        when a declared attribute genuinely changes.
        """
        live = _live_omniweb_permuted()
        live["attributes"]["post.logout.redirect.uris"] = "stale"

        payload = _ensure_mod()._build_update_payload(
            live, _OMNIWEB_SPEC, ["attributes"], None
        )

        assert payload["attributes"]["post.logout.redirect.uris"] == "+"
        assert payload["attributes"]["realm_client"] == "false"


# ---------------------------------------------------------------------------
# The preflight and the reconcile must agree about what has drifted
# ---------------------------------------------------------------------------


class TestPreflightAgreesWithReconcile:
    def test_preflight_reports_the_same_drift_fields(self) -> None:
        """One comparison, two callers. A survey that disagrees with the loop
        it precedes is worse than no survey.
        """
        live = {"omniweb": _live_omniweb_permuted()}
        fake_request, _writes = _fake_realm(live)

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
                records = _ensure_mod()._preflight_clients(
                    _KC_URL, _REALM, _TOKEN, [_OMNIWEB_SPEC]
                )

        assert records[0]["drift_fields"] == []
