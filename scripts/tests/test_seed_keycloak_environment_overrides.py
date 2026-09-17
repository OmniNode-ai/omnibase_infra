# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One roster, two realms: the declared per-environment override (OMN-18582).

Operator ruling 2026-09-17T17:33:52Z, verbatim "yeah we'll take your
recommendation": the ``onex-api`` entry is **per-environment** — confidential
with a consumer on dev-system, bearer-only on production. Resolve that ruling
by timestamp; the ledger rolled the same day and every line number moved.

**Why per-environment rather than one shape everywhere.** Both shapes were
measured, read-only, on 2026-09-17:

* dev-system runs a real consumer. ``deployment-omninode-runtime-effects.yaml``
  mounts ``KEYCLOAK_CLIENT_SECRET`` with ``optional: false`` and introspects
  with it, so ``onex-api`` there must be confidential and must hold a secret.
  Bearer-only on that realm is the OMN-16504 defect: Keycloak clears a
  bearer-only client's secret on every admin-API update.
* production runs no consumer at all. The ``onex-api`` Deployment in
  ``onex-prod`` carries ``KEYCLOAK_ISSUER_URL``, ``KEYCLOAK_AUDIENCE`` and the
  two tenant claim names and NO client secret of any kind, and
  ``onex-prod/onex-runtime-credentials`` carries eleven keys of which none is
  ``KEYCLOAK_CLIENT_SECRET``. There, bearer-only is simply true, and declaring
  the client confidential would have Keycloak mint a credential nothing uses —
  the anti-pattern OMN-16504 already paid for once.

**Why an override rather than a second file.** The roster is byte-identical
across three tracked copies and a parity gate enforces that. Forking it would
give the two realms two sources of truth, which is the failure this whole
component exists to remove. One file, one declared exception, and the
environment selected by the Job that runs.

**Why it fails closed.** A roster carrying overrides that is reconciled with
no environment named cannot be resolved to a single realm shape, so the run
refuses rather than silently picking the base. That matters most in the
direction nobody tests: a production Job that forgot the flag would otherwise
reconcile the *dev* shape into production, flip ``onex-api`` confidential, and
have Keycloak mint the credential the ruling exists to refuse.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = Path(__file__).parent.parent / "seed-keycloak-clients.py"
_ROSTER_PATH = (
    Path(__file__).resolve().parents[2] / "docker" / "keycloak" / "desired-clients.json"
)


def _mod() -> types.ModuleType:
    if "seed_keycloak_clients" in sys.modules:
        return sys.modules["seed_keycloak_clients"]
    spec = importlib.util.spec_from_file_location("seed_keycloak_clients", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["seed_keycloak_clients"] = module
    spec.loader.exec_module(module)
    return module


def _roster_clients() -> list[dict[str, Any]]:
    payload = json.loads(_ROSTER_PATH.read_text(encoding="utf-8"))
    clients = payload["clients"]
    assert isinstance(clients, list)
    return clients


def _onex_api() -> dict[str, Any]:
    entry = next(c for c in _roster_clients() if c["clientId"] == "onex-api")
    return entry


@pytest.mark.unit
class TestTheRosterDeclaresTheException:
    """The exception lives in the roster, in the open, with its reason."""

    def test_the_base_entry_is_the_dev_system_shape(self) -> None:
        """Base is what MOST realms get; the override is the deviation.

        Keeping dev-system as the base and production as the override is
        deliberate: the base shape is the one with a live consumer, so a new
        realm added without an override inherits the SAFE shape rather than
        one that silently holds no credential.
        """
        spec = _onex_api()
        assert spec.get("bearerOnly") is False
        assert spec.get("publicClient") is False
        assert spec.get("serviceAccountsEnabled") is True
        assert spec.get("consumerSecretEnv") == "ONEX_API_CONSUMER_CLIENT_SECRET"

    def test_the_prod_override_is_declared_with_a_reason_and_a_ticket(self) -> None:
        overrides = _onex_api().get("environmentOverrides")
        assert isinstance(overrides, dict), (
            "the per-environment exception must be declared in the roster, "
            "not implied by a code branch"
        )
        assert "prod" in overrides
        prod = overrides["prod"]
        assert prod["bearerOnly"] is True
        assert prod["serviceAccountsEnabled"] is False
        # Explicit null, not omission: the consumer variable must be REMOVED on
        # production, and an absent key would read as "inherit the base".
        assert "consumerSecretEnv" in prod
        assert prod["consumerSecretEnv"] is None
        assert prod.get("ticket") == "OMN-18582"
        assert isinstance(prod.get("reason"), str) and prod["reason"].strip()

    def test_onex_api_is_the_only_client_carrying_an_override(self) -> None:
        """One declared exception. A second needs its own deliberate change."""
        carriers = [
            c["clientId"] for c in _roster_clients() if "environmentOverrides" in c
        ]
        assert carriers == ["onex-api"]


@pytest.mark.unit
class TestApplyingAnOverride:
    """The pure resolution step, which is where the two realms diverge."""

    def test_prod_resolves_to_the_bearer_only_shape(self) -> None:
        resolved = _mod().apply_environment_overrides(_roster_clients(), "prod")
        spec = next(c for c in resolved if c["clientId"] == "onex-api")
        assert spec["bearerOnly"] is True
        assert spec["serviceAccountsEnabled"] is False
        # Removed, not set to None: a None would flow into the payload and be
        # sent to Keycloak as a field.
        assert "consumerSecretEnv" not in spec
        # The override block itself must not survive into the payload builders.
        assert "environmentOverrides" not in spec

    def test_dev_resolves_to_the_base_shape(self) -> None:
        resolved = _mod().apply_environment_overrides(_roster_clients(), "dev")
        spec = next(c for c in resolved if c["clientId"] == "onex-api")
        assert spec["bearerOnly"] is False
        assert spec["serviceAccountsEnabled"] is True
        assert spec["consumerSecretEnv"] == "ONEX_API_CONSUMER_CLIENT_SECRET"
        assert "environmentOverrides" not in spec

    def test_untouched_clients_are_unchanged_in_both_environments(self) -> None:
        base = {c["clientId"]: c for c in _roster_clients()}
        for environment in ("dev", "prod"):
            resolved = {
                c["clientId"]: c
                for c in _mod().apply_environment_overrides(
                    _roster_clients(), environment
                )
            }
            for client_id, spec in resolved.items():
                if client_id == "onex-api":
                    continue
                assert spec == base[client_id], (
                    f"{client_id} changed under environment {environment!r}; "
                    "only a client declaring an override may differ"
                )

    def test_the_input_roster_is_not_mutated(self) -> None:
        clients = _roster_clients()
        snapshot = json.dumps(clients, sort_keys=True)
        _mod().apply_environment_overrides(clients, "prod")
        assert json.dumps(clients, sort_keys=True) == snapshot

    def test_an_environment_with_no_override_gets_the_base(self) -> None:
        resolved = _mod().apply_environment_overrides(_roster_clients(), "lab")
        spec = next(c for c in resolved if c["clientId"] == "onex-api")
        assert spec["bearerOnly"] is False
        assert "environmentOverrides" not in spec


@pytest.mark.unit
class TestItFailsClosedWithoutAnEnvironment:
    """The direction that matters: an unnamed environment must not default."""

    def test_an_unresolvable_roster_refuses_and_names_the_client(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            _mod().apply_environment_overrides(_roster_clients(), None)
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "onex-api" in (captured.err + captured.out)

    def test_a_roster_with_no_overrides_needs_no_environment(self) -> None:
        """Backwards compatibility, stated as a test.

        Every roster written before this field existed resolves unchanged with
        no environment named, so this change cannot break a caller that has
        nothing to resolve.
        """
        plain = [c for c in _roster_clients() if c["clientId"] != "onex-api"]
        assert _mod().apply_environment_overrides(plain, None) == plain


@pytest.mark.unit
class TestTheSecretGuardLearnsTheSameException:
    """The guard that would otherwise refuse the production shape.

    ``_live_client_requires_secret`` classifies from the LIVE representation
    and deliberately does NOT exempt bearer-only, because dev-system's
    ``onex-api`` is bearer-shaped AND introspects. Under the ruling the
    production shape is a genuine pure resource server, so the exemption is
    now taken from the RESOLVED SPEC — which, unlike a partial roster entry,
    states the fact explicitly.
    """

    LIVE_CONFIDENTIAL = {"publicClient": False}

    def test_a_spec_that_declares_bearer_only_and_no_secret_source_is_exempt(
        self,
    ) -> None:
        spec = {"clientId": "onex-api", "publicClient": False, "bearerOnly": True}
        assert (
            _mod()._live_client_requires_secret(self.LIVE_CONFIDENTIAL, spec) is False
        )

    def test_a_spec_that_declares_bearer_only_but_names_a_secret_is_still_guarded(
        self,
    ) -> None:
        """The exemption is about having no credential, not about the flag."""
        for field in ("secretEnv", "consumerSecretEnv"):
            spec = {
                "clientId": "x",
                "publicClient": False,
                "bearerOnly": True,
                field: "SOME_VAR",
            }
            assert (
                _mod()._live_client_requires_secret(self.LIVE_CONFIDENTIAL, spec)
                is True
            ), f"{field} must keep the client guarded"

    def test_a_live_bearer_only_client_the_spec_does_not_declare_is_still_guarded(
        self,
    ) -> None:
        """The OMN-16504 client itself, before this ruling: live bearer-only,
        roster silent. That must stay guarded — the whole incident was a
        bearer-only live shape whose secret got cleared."""
        live = {"publicClient": False, "bearerOnly": True}
        assert (
            _mod()._live_client_requires_secret(live, {"clientId": "onex-api"}) is True
        )

    def test_a_public_client_is_exempt_as_before(self) -> None:
        assert (
            _mod()._live_client_requires_secret(
                {"publicClient": True}, {"clientId": "p"}
            )
            is False
        )

    def test_the_dev_shape_of_onex_api_remains_guarded(self) -> None:
        """The realm this guard was written for must not be weakened."""
        resolved = _mod().apply_environment_overrides(_roster_clients(), "dev")
        spec = next(c for c in resolved if c["clientId"] == "onex-api")
        assert _mod()._live_client_requires_secret(self.LIVE_CONFIDENTIAL, spec) is True

    def test_the_prod_shape_of_onex_api_is_exempt(self) -> None:
        resolved = _mod().apply_environment_overrides(_roster_clients(), "prod")
        spec = next(c for c in resolved if c["clientId"] == "onex-api")
        assert (
            _mod()._live_client_requires_secret(self.LIVE_CONFIDENTIAL, spec) is False
        )


@pytest.mark.unit
class TestTheEnvironmentIsReachableFromTheCommandLine:
    """A Job supplies it; the overlays are what make the two realms differ."""

    def test_the_parser_declares_an_environment_option(self) -> None:
        parser = _mod()._build_parser() if hasattr(_mod(), "_build_parser") else None
        options: set[str] = set()
        if parser is not None:
            for action in parser._actions:
                options.update(action.option_strings)
        else:
            import argparse
            from unittest.mock import patch

            with patch.object(sys, "argv", ["seed"]):
                captured: list[argparse.ArgumentParser] = []
                original = argparse.ArgumentParser.parse_args

                def _capture(
                    self: argparse.ArgumentParser, *args: Any, **kwargs: Any
                ) -> Any:
                    captured.append(self)
                    return original(self, *args, **kwargs)

                with patch.object(argparse.ArgumentParser, "parse_args", _capture):
                    _mod()._parse_args()
                for parser_instance in captured:
                    for action in parser_instance._actions:
                        options.update(action.option_strings)
        assert "--environment" in options
