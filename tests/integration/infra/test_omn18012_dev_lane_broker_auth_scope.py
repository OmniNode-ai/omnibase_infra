# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18012 -- the dev-lane broker principal must reach the DEV LANE ONLY.

``docker/docker-compose.infra.yml`` is the BASE that every lane merges:
stability-test, prod and judge each layer their overlay on top of it. A service
written into the base is inherited by all of them, and one carrying a ``:?``
required variable would fail their next deploy at compose render on a variable
that has nothing to do with them.

The first draft of this change put ``redpanda-scram-user`` in the base file. It
parsed, it looked dev-scoped, and it would have made the dev lane's synthetic
SCRAM credential a hard precondition of deploying prod. These assertions are
the RED guard for that, in both directions: the service must be in the dev-lane
overlay, and it must be in nothing else.

No docker required -- these read the committed compose files.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

DOCKER_DIR = Path(__file__).resolve().parents[3] / "docker"
DEV_LANE_OVERLAY = DOCKER_DIR / "docker-compose.dev-lane.yml"
BASE = DOCKER_DIR / "docker-compose.infra.yml"
OTHER_LANE_OVERLAYS = (
    "docker-compose.stability-test.yml",
    "docker-compose.prod.yml",
    "docker-compose.judge.yml",
    "docker-compose.lakshman.yml",
)

SERVICE = "redpanda-scram-user"
FLIP_SERVICE = "redpanda-sasl-enable"

# The four names ModelKafkaEventBusConfig.apply_environment_overrides() reads.
# Every Kafka client in every lane resolves transport auth through that one
# function, so this tuple IS the client-side contract of phase B.
AUTH_ENV_KEYS = (
    "KAFKA_SECURITY_PROTOCOL",
    "KAFKA_SASL_MECHANISM",
    "KAFKA_SASL_USERNAME",
    "KAFKA_SASL_PASSWORD",
)

# Every service on the dev lane that speaks Kafka, taken from the LIVE lane on
# 2026-09-07 (`docker inspect` over every container in compose project
# omnibase-infra, counting KAFKA/BROKER environment variables) rather than from
# a reading of the compose files -- the compose files are the thing under test.
# The three consumers at the end are the ones a prefix-blind inventory misses:
# they carry only OMNIBASE_INFRA_<SERVICE>_KAFKA_BOOTSTRAP_SERVERS in the base
# and resolve auth through the unprefixed names.
DEV_LANE_KAFKA_CLIENTS = frozenset(
    {
        "omninode-runtime",
        "runtime-effects",
        "runtime-worker",
        "projection-api",
        "omninode-contract-resolver",
        "projection-tenant-registry-writer",
        "projection-delegation-writer",
        "projection-registration-writer",
        "projection-savings-writer",
        "projection-tenant-credentials-writer",
        "projection-live-events-writer",
        "infra-routing-decisions-consumer",
        "agent-actions-consumer",
        "skill-lifecycle-consumer",
        "context-audit-consumer",
    }
)


class _TolerantLoader(yaml.SafeLoader):
    """compose uses `!override`, which SafeLoader refuses."""


_TolerantLoader.add_multi_constructor(  # type: ignore[no-untyped-call]
    "",
    lambda loader, suffix, node: (
        loader.construct_mapping(node)
        if isinstance(node, yaml.MappingNode)
        else (
            loader.construct_sequence(node)
            if isinstance(node, yaml.SequenceNode)
            else loader.construct_scalar(node)
        )
    ),
)


def _services(path: Path) -> dict[str, object]:
    with open(path, encoding="utf-8") as handle:
        # S506: _TolerantLoader subclasses SafeLoader; the only widening is a
        # multi-constructor for compose's `!override` tag, which resolves to a
        # plain mapping/sequence/scalar. No arbitrary object can be built.
        parsed = yaml.load(handle, Loader=_TolerantLoader)  # noqa: S506
        return dict(parsed.get("services") or {})


def test_the_scram_principal_service_is_declared_on_the_dev_lane_overlay() -> None:
    services = _services(DEV_LANE_OVERLAY)
    assert SERVICE in services, (
        f"{SERVICE} is missing from {DEV_LANE_OVERLAY.name}; the dev lane has "
        "no principal and phase B would flip SASL on a broker with no user"
    )


def test_the_scram_principal_never_leaks_into_the_base_or_another_lane() -> None:
    """The whole blast-radius assertion, stated as one test."""
    assert SERVICE not in _services(BASE), (
        f"{SERVICE} is in the BASE compose file. stability-test, prod and "
        "judge all merge that file, so their next deploy would fail at "
        "compose render on DEV_KAFKA_SASL_USERNAME. Move it to "
        f"{DEV_LANE_OVERLAY.name}."
    )
    for name in OTHER_LANE_OVERLAYS:
        path = DOCKER_DIR / name
        if not path.exists():
            continue
        assert SERVICE not in _services(path), f"{SERVICE} leaked into {name}"


def test_no_lane_but_dev_gains_a_sasl_listener_flag() -> None:
    """Positive control on the negative: prove the grep can actually find one.

    An assertion that a string is absent from four files is worthless unless
    the same search finds it where it IS present. The dev-lane overlay is that
    control.
    """
    needle = "DEV_KAFKA_SASL_USERNAME"
    dev_text = DEV_LANE_OVERLAY.read_text(encoding="utf-8")
    assert needle in dev_text, (
        "positive control failed: the needle is absent from the file that is "
        "supposed to contain it, so the absence assertions below prove nothing"
    )
    assert needle not in BASE.read_text(encoding="utf-8")
    for name in OTHER_LANE_OVERLAYS:
        path = DOCKER_DIR / name
        if path.exists():
            assert needle not in path.read_text(encoding="utf-8"), (
                f"{needle} leaked into {name}"
            )


def test_the_principal_credential_is_a_reference_never_a_literal() -> None:
    services = _services(DEV_LANE_OVERLAY)
    principal = dict(services[SERVICE])  # type: ignore[call-overload]
    env = dict(principal["environment"])
    for key in ("DEV_KAFKA_SASL_USERNAME", "DEV_KAFKA_SASL_PASSWORD"):
        value = str(env[key])
        assert value.startswith("${") and ":?" in value, (
            f"{key} must be a fail-closed compose reference, not a literal or "
            f"a `:-` fallback; got {value!r}"
        )


# ---------------------------------------------------------------------------
# PHASE B -- the flip, and the client credentials that must accompany it
# ---------------------------------------------------------------------------


def test_the_sasl_flip_is_a_dev_lane_one_shot_and_reaches_no_other_lane() -> None:
    """The broker service is BASE-declared; only a separate one-shot is safe.

    A ``command:`` override on ``redpanda`` would have flipped stability-test,
    prod, judge and lakshman along with the dev lane -- the base file is what
    all five merge. The flip therefore has to be its own service, declared
    here.
    """
    assert FLIP_SERVICE in _services(DEV_LANE_OVERLAY), (
        f"{FLIP_SERVICE} is missing: the lane has a principal but its listener "
        "still accepts PLAINTEXT, so nothing is proven"
    )
    assert FLIP_SERVICE not in _services(BASE), (
        f"{FLIP_SERVICE} is in the BASE compose file, which stability-test, "
        "prod, judge and lakshman all merge -- this would flip four other "
        "lanes' brokers to SASL with no client anywhere configured for it"
    )
    for name in OTHER_LANE_OVERLAYS:
        path = DOCKER_DIR / name
        if path.exists():
            assert FLIP_SERVICE not in _services(path), (
                f"{FLIP_SERVICE} leaked into {name}"
            )


def test_the_flip_waits_on_the_principal_not_merely_on_the_broker() -> None:
    """Ordering, stated as an assertion because it is the lockout condition.

    ``enable_sasl`` on a cluster whose user creation silently failed is a lane
    locked out of its own broker with no unauthenticated path back in. The
    dependency must be ``service_completed_successfully`` on the principal
    one-shot -- ``service_healthy`` on the broker would satisfy compose while
    proving nothing about the user.
    """
    flip = dict(_services(DEV_LANE_OVERLAY)[FLIP_SERVICE])  # type: ignore[call-overload]
    depends = dict(flip["depends_on"])
    assert SERVICE in depends, (
        f"{FLIP_SERVICE} does not wait for {SERVICE}; SASL can come on before "
        "the principal exists"
    )
    condition = dict(depends[SERVICE])["condition"]
    assert condition == "service_completed_successfully", (
        f"{FLIP_SERVICE} waits on {SERVICE} with condition {condition!r}. Only "
        "service_completed_successfully proves the user was actually created; "
        "a failed one-shot must block the flip, not merely precede it."
    )


def test_the_superuser_grant_precedes_the_sasl_flip_in_the_script() -> None:
    """kafka_enable_authorization follows enable_sasl on this cluster.

    So the instant SASL comes on, authorization comes on with it, and a
    principal holding neither superuser status nor ACLs authenticates and can
    then do nothing. Granting first closes that window; the reverse order opens
    it, and the failure reads as an auth error in every client log.
    """
    flip = dict(_services(DEV_LANE_OVERLAY)[FLIP_SERVICE])  # type: ignore[call-overload]
    script = "\n".join(str(part) for part in flip["command"])
    grant_at = script.index("cluster config set superusers")
    flip_at = script.index("cluster config set enable_sasl")
    assert grant_at < flip_at, (
        "enable_sasl is set before the superuser grant. Between those two "
        "lines every client on the lane can authenticate and is authorized "
        "for nothing."
    )


def test_the_flip_asserts_the_refusal_with_a_positive_control() -> None:
    """A refusal proves nothing unless the same broker accepts the good client.

    A broker that is simply DOWN refuses the anonymous client too. Without the
    authenticated half in the same script, this one-shot would go green on a
    dead lane.
    """
    flip = dict(_services(DEV_LANE_OVERLAY)[FLIP_SERVICE])  # type: ignore[call-overload]
    script = "\n".join(str(part) for part in flip["command"])
    assert "unauthenticated client refused" in script, (
        "the flip does not assert that an unauthenticated client is refused, "
        "which is the entire point of the change"
    )
    assert "positive control" in script, (
        "the flip asserts a refusal with no positive control; a down broker "
        "would satisfy it"
    )
    assert "|| true" not in script, (
        "a swallowed exit code here is the invisible failure this ticket exists "
        "to remove"
    )


def test_every_dev_lane_kafka_client_carries_the_four_auth_variables() -> None:
    """The client half, per service.

    One unbound client is not a partial success: it is a service that keeps
    speaking PLAINTEXT to a listener that now refuses it, which is escape 1's
    exact shape reintroduced by the change meant to prevent it.
    """
    services = _services(DEV_LANE_OVERLAY)
    missing: dict[str, list[str]] = {}
    for name in sorted(DEV_LANE_KAFKA_CLIENTS):
        body = dict(services.get(name) or {})  # type: ignore[call-overload]
        env = dict(body.get("environment") or {})
        absent = [key for key in AUTH_ENV_KEYS if key not in env]
        if absent:
            missing[name] = absent
    assert not missing, (
        f"dev-lane Kafka clients missing broker credentials: {missing}. Each "
        "will fail authentication against the lane's own broker."
    )


def test_the_credentials_never_reach_the_base_or_another_lane() -> None:
    """Blast radius of the CLIENT half, with its own positive control.

    Binding KAFKA_SECURITY_PROTOCOL in the base's ``x-common-env`` would point
    stability-test, prod, judge and lakshman clients at a SASL handshake their
    brokers do not offer.
    """
    dev_text = DEV_LANE_OVERLAY.read_text(encoding="utf-8")
    for key in AUTH_ENV_KEYS:
        assert key in dev_text, (
            f"positive control failed: {key} is absent from the file that must "
            "contain it, so the absence assertions below prove nothing"
        )
    for key in ("KAFKA_SASL_USERNAME", "KAFKA_SASL_PASSWORD"):
        assert key not in BASE.read_text(encoding="utf-8"), (
            f"{key} is in the BASE compose file; four other lanes merge it"
        )
        for name in OTHER_LANE_OVERLAYS:
            path = DOCKER_DIR / name
            if path.exists():
                assert key not in path.read_text(encoding="utf-8"), (
                    f"{key} leaked into {name}"
                )


def test_the_client_credentials_are_references_never_literals() -> None:
    services = _services(DEV_LANE_OVERLAY)
    runtime = dict(services["omninode-runtime"])  # type: ignore[call-overload]
    env = dict(runtime["environment"])
    for key in ("KAFKA_SASL_USERNAME", "KAFKA_SASL_PASSWORD"):
        value = str(env[key])
        assert value.startswith("${") and ":?" in value, (
            f"{key} must be a fail-closed compose reference, not a literal or "
            f"a `:-` fallback; got {value!r}"
        )
