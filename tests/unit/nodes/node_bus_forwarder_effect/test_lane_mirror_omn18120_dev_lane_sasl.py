# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18120: the gateway's DEV-lane legs must speak SCRAM, like every other client.

OMN-18012 flipped the dev lane's broker to SASL and, in its own words, gave
"all 15 clients the credential" (#3276, d7c558531, 2026-09-07). The gateway
forwarder was not one of the fifteen. It lives in a different compose project
(``docker/docker-compose.gateway.yml``, project ``omninode-gateway``) and
carries its own per-leg bus configuration in a mounted YAML rather than in the
``x-dev-lane-broker-auth-env`` anchor every dev-lane service shares, so the
enumeration that credentialed the lane did not reach it.

Measured on ``.201`` 2026-09-10, read-only, with a positive control on the same
command so the failure is not a broken probe:

    rpk --config <empty> -X brokers=omninode-pc...:19092 cluster metadata
      -> broker closed the connection immediately after a request was issued,
         which happens when SASL is required but not provided: is SASL missing?

    rpk --config <empty> -X brokers=100.109.203.94:39092 cluster metadata
      -> CLUSTER / BROKERS ...                       # stability, still PLAINTEXT

That asymmetry is the whole defect. The mirror's SOURCE leg dials stability,
which is plaintext, and works. Its DEV leg dials a listener that now refuses a
plaintext client, so every republish fails at connect and not one record has
reached the dev lane since. Measured the same morning: all four dev hook topics'
high watermarks were static across a 91s window (``tool-executed`` 599687,
``session-started`` 1251873, ``prompt-submitted`` 8660, ``session-ended`` 9759)
while the forwarder logged 110 ``Lane mirror delivered`` lines in three minutes.
Positive control on the same reads: ``onex.dlq.omnibase-infra.quarantine.v1``
advanced 11475729 -> 11475736 in that window, so the broker was taking writes.

Why this is a lane-credential bug and not the OMN-17919 alias bug it looks
like: the alias was fixed. ``getent hosts redpanda`` inside the forwarder now
answers 172.19.0.4, the dev broker, and the dev leg is addressed at the dev
lane's external listener anyway. Both of those are correct. The leg still
cannot produce, because it declares ``PLAINTEXT``.

Why the credential cannot be a literal in the shipped YAML: that file is mounted
into the container verbatim (``docker-compose.gateway.yml``,
``./gateway/beta-gateway-canary.yaml:/app/config/gateway-forwarder.yaml:ro``)
and is tracked in this repo. The leg names a REF, resolved at the effect
boundary against an operator-supplied map mounted the same way the cloud leg's
endpoint already is -- and NOT from an environment variable, which would be
readable from ``docker inspect`` for the life of the container and which this
repo's ``check-env-reads`` gate refuses outside the overlay surfaces. The mount
is declared ``:?`` and not ``:-`` for the same reason
``x-dev-lane-broker-auth-env`` is: a leg that silently falls back to no
credential is a leg whose refusal proves nothing.

NOT A ROTATION (Operating Rule 22). This binds the principal OMN-18012 phase A
provisioned. Nothing is rotated, re-issued or revoked, and no value appears here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GATEWAY_CANARY = _REPO_ROOT / "docker" / "gateway" / "beta-gateway-canary.yaml"
_GATEWAY_COMPOSE = _REPO_ROOT / "docker" / "docker-compose.gateway.yml"

# The mechanism the dev lane's broker principal was provisioned with
# (OMN-18012 phase A, `redpanda-scram-user`). Named, never spelled twice.
_DEV_LANE_SASL_MECHANISM = "SCRAM-SHA-256"
_SASL_PROTOCOLS = frozenset({"SASL_PLAINTEXT", "SASL_SSL"})

# Every bus leg in the shipped gateway config that dials the DEV lane's broker.
_DEV_LANE_LEGS = ("local_bus", "lane_mirror_buses.dev")

# The credential-reference key the effect boundary resolves against the
# operator-supplied lane-credential map. A NAME, never a value.
_CREDENTIAL_REF_KEY = "sasl_credential_ref"

# Keys that would mean a literal credential had been committed to this repo.
_FORBIDDEN_LITERAL_KEYS = ("sasl_plain_username", "sasl_plain_password")


def _gateway_config() -> dict[str, Any]:
    return cast(
        "dict[str, Any]", yaml.safe_load(_GATEWAY_CANARY.read_text(encoding="utf-8"))
    )


def _leg(config: dict[str, Any], dotted: str) -> dict[str, Any]:
    node: Any = config
    for part in dotted.split("."):
        assert isinstance(node, dict) and part in node, (
            f"the shipped gateway config has no leg at {dotted!r}; this test "
            "constrains the legs that dial the dev lane and cannot silently "
            "pass because one was renamed"
        )
        node = node[part]
    return cast("dict[str, Any]", node)


# ---------------------------------------------------------------------------
# The premise, asserted so the fix below is not cargo-culted
# ---------------------------------------------------------------------------


def test_the_source_leg_stays_plaintext() -> None:
    """The stability leg must NOT be swept up in this change.

    The two lanes genuinely differ: stability's external listener still accepts
    a plaintext client, measured above. Giving the source leg a dev-lane
    credential would break the one leg that works, and would also hand a dev
    principal to a governed lane. This test is what stops a later "make them
    consistent" edit.
    """
    source = _leg(_gateway_config(), "lane_mirror_source_bus")
    assert source["security_protocol"] == "PLAINTEXT", (
        "the lane-mirror SOURCE leg dials the stability lane, which accepts "
        "plaintext; it must not be given the dev lane's SCRAM credential"
    )
    assert _CREDENTIAL_REF_KEY not in source


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dotted", _DEV_LANE_LEGS)
def test_dev_lane_legs_declare_sasl(dotted: str) -> None:
    """RED at parent: both dev-lane legs ship ``security_protocol: PLAINTEXT``.

    The dev broker's listener refuses a plaintext client (measured above), so a
    leg declaring PLAINTEXT cannot connect, cannot produce, and -- because the
    forwarder logs a delivery per attempted republish -- fails silently.
    """
    leg = _leg(_gateway_config(), dotted)
    protocol = leg.get("security_protocol")
    assert protocol in _SASL_PROTOCOLS, (
        f"gateway leg {dotted!r} declares security_protocol={protocol!r}, but "
        "the dev lane's broker has enable_sasl=true and closes a plaintext "
        "connection at handshake (OMN-18012). Every republish on this leg "
        "fails at connect."
    )
    assert leg.get("sasl_mechanism") == _DEV_LANE_SASL_MECHANISM, (
        f"gateway leg {dotted!r} must authenticate with the mechanism the dev "
        f"lane's principal was provisioned with ({_DEV_LANE_SASL_MECHANISM})"
    )


@pytest.mark.parametrize("dotted", _DEV_LANE_LEGS)
def test_dev_lane_legs_reference_a_credential_and_never_inline_one(
    dotted: str,
) -> None:
    """The credential is a REFERENCE. A literal in this repo would be the leak.

    This file is tracked and is mounted into the container verbatim, so a
    username or password spelled here is committed to the repository. The leg
    names an env PREFIX; the effect boundary resolves the two names from the
    operator env file and fails closed when either is unset.
    """
    leg = _leg(_gateway_config(), dotted)
    prefix = leg.get(_CREDENTIAL_REF_KEY)
    assert isinstance(prefix, str) and prefix, (
        f"gateway leg {dotted!r} must declare {_CREDENTIAL_REF_KEY!r} so the "
        "effect boundary can resolve its SASL credential from the operator "
        "env file"
    )
    for forbidden in _FORBIDDEN_LITERAL_KEYS:
        assert forbidden not in leg, (
            f"gateway leg {dotted!r} declares {forbidden!r} as a literal; the "
            "shipped config is tracked in this repo and mounted verbatim, so "
            "the value must be resolved at the effect boundary instead"
        )


def test_compose_passes_the_dev_lane_credential_fail_closed() -> None:
    """The credential map must be mounted AND named on the command line.

    Either half missing leaves the reference unresolvable. ``:-`` would let the
    forwarder start without a credential and produce the same silent inertness
    this ticket exists to remove, so the assertion is on the fail-closed form.
    """
    compose = _GATEWAY_COMPOSE.read_text(encoding="utf-8")
    config = _gateway_config()
    prefix = _leg(config, "lane_mirror_buses.dev")[_CREDENTIAL_REF_KEY]
    assert "${GATEWAY_LANE_CREDENTIAL_MAP_FILE:?" in compose, (
        f"{_GATEWAY_COMPOSE.name} must mount the lane-credential map with the "
        "fail-closed ${VAR:?...} form; a forwarder that starts without it is a "
        "forwarder whose legs connect to nothing while still logging deliveries"
    )
    assert "--lane-credential-map" in compose, (
        f"{_GATEWAY_COMPOSE.name} must pass --lane-credential-map, otherwise "
        f"the {prefix!r} reference resolves against nothing"
    )


# ---------------------------------------------------------------------------
# The boundary that resolves the reference
# ---------------------------------------------------------------------------

_REF = "lane.dev.kafka.scram"


def _leg_with_ref(**overrides: Any) -> dict[str, object]:
    leg: dict[str, object] = {
        "bootstrap_servers": "dev.example:19092",
        "environment": "t",
        "security_protocol": "SASL_PLAINTEXT",
        "sasl_mechanism": _DEV_LANE_SASL_MECHANISM,
        _CREDENTIAL_REF_KEY: _REF,
    }
    leg.update(overrides)
    return {"lane_mirror_buses": {"dev": leg}}


def _credential_map(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "lane-credentials.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _valid_map(tmp_path: Path) -> Path:
    # Obviously fake. The point of the whole change is that a real value never
    # appears in this repository.
    return _credential_map(
        tmp_path,
        f"{_REF}:\n  username: unit-principal\n  password: unit-not-a-real-secret\n",
    )


def _materialize() -> Any:
    from omnibase_infra.runtime.gateway_forwarder import (
        _materialize_lane_broker_credentials,
    )

    return _materialize_lane_broker_credentials


def test_boundary_resolves_the_reference_and_strips_it(tmp_path: Path) -> None:
    """The resolved leg carries the values and NOT the ref key.

    The bus-config model is ``extra="forbid"``, so leaving the reference behind
    would fail validation. Asserting the strip is what keeps that from being
    rediscovered at deploy time.
    """
    raw = _leg_with_ref()
    _materialize()(raw, _valid_map(tmp_path))

    leg = cast("dict[str, Any]", raw["lane_mirror_buses"])["dev"]
    assert leg["sasl_plain_username"] == "unit-principal"
    assert leg["sasl_plain_password"] == "unit-not-a-real-secret"
    assert _CREDENTIAL_REF_KEY not in leg


def test_boundary_fails_closed_with_no_map_at_all() -> None:
    """A leg naming a ref with no map mounted refuses to start."""
    with pytest.raises(ValueError, match=_REF):
        _materialize()(_leg_with_ref(), None)


def test_boundary_fails_closed_when_the_map_file_is_absent(tmp_path: Path) -> None:
    """A declared-but-missing map is a refusal, not an empty map."""
    with pytest.raises(ValueError, match="no lane-credential map"):
        _materialize()(_leg_with_ref(), tmp_path / "not-there.yaml")


def test_boundary_fails_closed_on_a_missing_entry(tmp_path: Path) -> None:
    """A map that resolves nothing for this ref refuses, naming the ref."""
    path = _credential_map(tmp_path, "some.other.ref:\n  username: u\n  password: p\n")
    with pytest.raises(ValueError, match="no resolvable entry"):
        _materialize()(_leg_with_ref(), path)


@pytest.mark.parametrize("field", ["username", "password"])
def test_boundary_fails_closed_on_an_empty_field(tmp_path: Path, field: str) -> None:
    """An empty credential field is the silent-inertness case, so it must refuse."""
    values = {"username": "unit-principal", "password": "unit-not-a-real-secret"}
    values[field] = "   "
    path = _credential_map(
        tmp_path,
        f'{_REF}:\n  username: "{values["username"]}"\n  password: "{values["password"]}"\n',
    )
    with pytest.raises(ValueError, match=field):
        _materialize()(_leg_with_ref(), path)


def test_boundary_refuses_a_reference_and_a_literal_together(tmp_path: Path) -> None:
    """Two sources for one credential is ambiguous, and the literal is the leak."""
    with pytest.raises(ValueError, match="sasl_plain_password"):
        _materialize()(
            _leg_with_ref(sasl_plain_password="inlined"), _valid_map(tmp_path)
        )


def test_boundary_refuses_a_credential_it_would_never_send(tmp_path: Path) -> None:
    """Naming a ref on a PLAINTEXT leg is a config error, not a no-op."""
    with pytest.raises(ValueError, match="never sent"):
        _materialize()(
            _leg_with_ref(security_protocol="PLAINTEXT"), _valid_map(tmp_path)
        )


def test_boundary_round_trips_its_own_resolved_output(tmp_path: Path) -> None:
    """A leg carrying only resolved literals must load again unchanged.

    The loader's output is round-tripped through ``model_dump()`` in several
    places, and that output carries the RESOLVED credential with no reference
    left. Refusing a bare literal would make the loader unable to re-read what
    it just produced -- the same trap ``_materialize_contract_https_ingest``
    records for its own null round trip. The guarantee that the SHIPPED file
    carries no literal is a static assertion over the tracked YAML above.
    """
    raw = _leg_with_ref()
    _materialize()(raw, _valid_map(tmp_path))
    resolved = cast("dict[str, Any]", raw["lane_mirror_buses"])["dev"]

    again: dict[str, object] = {"lane_mirror_buses": {"dev": dict(resolved)}}
    _materialize()(again, None)
    assert cast("dict[str, Any]", again["lane_mirror_buses"])["dev"] == resolved


def test_boundary_leaves_a_leg_without_a_reference_alone() -> None:
    """The stability source leg must round-trip untouched.

    The negative control for the whole change: a leg that names no ref is not
    given a credential and not refused, even with no map supplied at all, so
    this boundary cannot break the plaintext leg that works.
    """
    raw: dict[str, object] = {
        "lane_mirror_source_bus": {
            "bootstrap_servers": "stability.example:39092",
            "environment": "t",
            "security_protocol": "PLAINTEXT",
        }
    }
    _materialize()(raw, None)
    assert raw["lane_mirror_source_bus"] == {
        "bootstrap_servers": "stability.example:39092",
        "environment": "t",
        "security_protocol": "PLAINTEXT",
    }


# ---------------------------------------------------------------------------
# The HEALTHCHECK is a second consumer of the same loader (OMN-18120 follow-up)
# ---------------------------------------------------------------------------
# `test_compose_passes_the_dev_lane_credential_fail_closed` above asserts
# `"--lane-credential-map" in compose` -- a substring match over the WHOLE
# file. That is why this defect shipped green: the flag IS in the file, on the
# forwarder's `command:`, so the substring is satisfied while the healthcheck's
# own `test:` array never received it.
#
# `onex-gateway-canary-probe` calls the SAME
# `load_gateway_forwarder_runtime_config` the forwarder does. Once a dev-lane
# leg names `sasl_credential_ref`, that loader fails closed without the map --
# correctly, by design. So the probe inherited a hard refusal it was never
# given the argument to satisfy, and every 15s healthcheck tick died in
# argument parsing before dialing a single broker:
#
#   ValueError: gateway bus leg 'local_bus' names sasl_credential_ref=... but
#   no lane-credential map was supplied; the gateway refuses to start rather
#   than open a client that cannot authenticate
#
# Measured on `.201` 2026-09-10T18:17Z, on the deployed fix: the forwarder
# itself was healthy and delivering (`Authenticated as onex-dev-lane via
# SCRAM-SHA-256`, dev `tool-executed` 599687 -> 599711, `work_events`
# n_tup_ins 71208 -> 71237 -- real INSERTs, not the no-op upsert loop), while
# the container's own health status read `unhealthy` with FailingStreak 5 and
# the traceback above as its last probe output. A healthcheck that cannot
# resolve its own configuration reports the one state it must never invent: it
# says the path is dead while the path is carrying traffic. The inverse of the
# 2026-08-04 outage this probe was built for, and just as misleading.
#
# The assertions below are per-INVOCATION, not per-file, because per-file is
# the exact granularity that let this through.

_PROBE_ENTRYPOINT = "onex-gateway-canary-probe"
_LANE_CREDENTIAL_FLAG = "--lane-credential-map"
_BROKER_REF_FLAG = "--broker-ref-map"


def _gateway_forwarder_service() -> dict[str, Any]:
    """The `gateway-forwarder` service block from the shipped compose file."""
    compose = cast(
        "dict[str, Any]", yaml.safe_load(_GATEWAY_COMPOSE.read_text(encoding="utf-8"))
    )
    service = compose["services"]["gateway-forwarder"]
    return cast("dict[str, Any]", service)


def test_healthcheck_receives_the_same_lane_credential_map_as_the_process() -> None:
    """The canary probe shares the forwarder's fail-closed loader, so it needs the map.

    Asserted against the healthcheck's own argv rather than the file's text.
    A whole-file substring is satisfied by the forwarder's `command:` alone,
    which is precisely how a probe with no credential map shipped as green.
    """
    service = _gateway_forwarder_service()
    healthcheck_test = service["healthcheck"]["test"]
    assert _PROBE_ENTRYPOINT in healthcheck_test, (
        "this test is pinned to the canary-probe healthcheck; the healthcheck "
        f"no longer invokes {_PROBE_ENTRYPOINT!r}, so re-derive it rather than "
        "letting the assertion below pass vacuously"
    )
    assert _LANE_CREDENTIAL_FLAG in healthcheck_test, (
        f"the healthcheck invokes {_PROBE_ENTRYPOINT}, which calls the same "
        "load_gateway_forwarder_runtime_config the forwarder process does. "
        f"Once a dev-lane leg names {_CREDENTIAL_REF_KEY!r} that loader fails "
        f"closed without {_LANE_CREDENTIAL_FLAG}, so the probe dies in "
        "configuration load and reports the traffic path dead while it is "
        "carrying traffic"
    )


def test_healthcheck_and_process_resolve_the_same_credential_map_path() -> None:
    """Two argv lists naming two different maps is a probe testing a fiction.

    The probe's whole value is dialing the legs with the same transport and
    the same credentials as real traffic. A map path that disagrees with the
    forwarder's makes it a check on a configuration nothing runs.
    """
    service = _gateway_forwarder_service()
    command = service["command"]
    healthcheck_test = service["healthcheck"]["test"]

    for flag in (_LANE_CREDENTIAL_FLAG, _BROKER_REF_FLAG):
        assert flag in command, f"the forwarder process must be given {flag}"
        process_path = command[command.index(flag) + 1]
        probe_path = healthcheck_test[healthcheck_test.index(flag) + 1]
        assert process_path == probe_path, (
            f"the forwarder resolves {flag} from {process_path!r} but its "
            f"healthcheck reads {probe_path!r}; the probe would be verifying a "
            "configuration the running process does not use"
        )


def test_canary_probe_accepts_the_lane_credential_map_argument() -> None:
    """The compose flag is inert unless the probe's own parser takes it.

    Both halves are load-bearing and they live in different files, so each is
    asserted where it can actually fail.
    """
    from omnibase_infra.runtime.gateway_canary_probe import _build_parser

    parser = _build_parser()
    flags = {option for action in parser._actions for option in action.option_strings}
    assert _LANE_CREDENTIAL_FLAG in flags, (
        f"{_PROBE_ENTRYPOINT} must accept {_LANE_CREDENTIAL_FLAG}; compose "
        "passing a flag argparse rejects turns the healthcheck into an "
        "immediate usage error on every tick"
    )
