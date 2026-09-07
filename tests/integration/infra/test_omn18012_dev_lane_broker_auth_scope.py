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


class _TolerantLoader(yaml.SafeLoader):
    """compose uses `!override`, which SafeLoader refuses."""


_TolerantLoader.add_multi_constructor(
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
    env = dict(services[SERVICE]["environment"])  # type: ignore[index,call-overload]
    for key in ("DEV_KAFKA_SASL_USERNAME", "DEV_KAFKA_SASL_PASSWORD"):
        value = str(env[key])
        assert value.startswith("${") and ":?" in value, (
            f"{key} must be a fail-closed compose reference, not a literal or "
            f"a `:-` fallback; got {value!r}"
        )
