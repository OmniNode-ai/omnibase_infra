# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26121: a stability roll flips its broker to SASL before it recreates a client.

Recorded case: agent jobs d1d52aac-9cf9-4307-b3dc-c0f8499dc0a0 and
53c37071-39e5-4b3c-a1b2-5dc3c5cc028b (2026-11-22T22:01Z and 22:11Z) failed in the
runtime phase with "stability lane writer tenant-projection-writer is not stable
after the recreate: status=running health=starting restarts=1". The recreated
writer was rendered with KAFKA_SECURITY_PROTOCOL SASL_PLAINTEXT and logged
``aiokafka: Unable connect to "redpanda:9092": [Error 34] IllegalSaslStateError``,
because the stability broker held enable_sasl=false, no SCRAM user and no
superusers: the overlay's ``redpanda-scram-user`` and ``redpanda-sasl-enable``
one-shots were never started, the roll's ``up -d --no-deps`` bypassing the
``depends_on`` chain that orders them and ``CONTROL_PLANE_ONESHOT_SERVICES`` naming
only ``redpanda-partition-cap``.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest
from deploy_agent import executor as execution
from deploy_agent.events import EnumRuntimeLane, Scope

pytestmark = [pytest.mark.unit]

LANE = EnumRuntimeLane.STABILITY_TEST
SHA = "c" * 40
PROJECT = execution.lane_config_for(LANE).compose_project
BROKER_IMAGE = "redpandadata/redpanda:v24.2.7"
SCRAM = "redpanda-scram-user"
ENABLE = "redpanda-sasl-enable"
WRITER = "tenant-projection-writer"
DECLARED = f"redpanda\n{SCRAM}\n{ENABLE}\n{WRITER}\n"


def _effect() -> object:
    return execution.load_dev_boot_guard(
        "nodes.node_setup_local_provision_effect.runtime_cost_estimator_build"
    )


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A scripted docker. ``state.events`` is the ordered record of mutations."""
    state = SimpleNamespace(
        events=[], declared=DECLARED, exit_codes={SCRAM: "0", ENABLE: "0"}
    )

    def run(
        argv: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        out = ""
        if argv[:2] == ["docker", "compose"] and "config" in argv:
            if "--services" in argv:
                out = state.declared
            else:
                out = json.dumps(
                    {
                        "name": PROJECT,
                        "services": {
                            argv[-1]: {"build": {}},
                            "redpanda": {"image": BROKER_IMAGE},
                        },
                    }
                )
        elif argv[:2] == ["docker", "compose"] and "ps" in argv:
            out = f"id-{argv[-1]}"
        elif argv[:2] == ["docker", "compose"] and "up" in argv:
            service = argv[-1]
            kind = "oneshot" if service in (SCRAM, ENABLE) else "writers"
            state.events.append(f"up:{kind}:{service}")
        elif argv[:2] == ["docker", "wait"]:
            service = argv[-1].removeprefix("id-")
            state.events.append(f"wait:{service}")
            out = state.exit_codes[service]
        elif argv[:2] == ["docker", "inspect"]:
            out = "running|healthy|0"
        elif argv[:3] == ["docker", "image", "inspect"]:
            out = SHA if argv[-1].startswith(PROJECT) else ""
        return subprocess.CompletedProcess(argv, 0, out, "")

    monkeypatch.setattr(execution, "_run", run)
    return state


def _roll() -> tuple[str, ...]:
    step = _effect().writer_roll(execution, LANE.value, Scope.RUNTIME, SHA)[
        "after_migrations"
    ]
    rolled: tuple[str, ...] = step()
    return rolled


def test_the_broker_is_flipped_before_any_writer_is_recreated(
    docker: SimpleNamespace,
) -> None:
    _roll()
    flip = [
        f"up:oneshot:{SCRAM}",
        f"wait:{SCRAM}",
        f"up:oneshot:{ENABLE}",
        f"wait:{ENABLE}",
    ]
    assert docker.events[:4] == flip
    assert any(e.startswith("up:writers:") for e in docker.events[4:]), (
        "positive control: the writers are recreated after the flip"
    )


def test_a_one_shot_that_does_not_exit_0_fails_the_roll_and_recreates_no_writer(
    docker: SimpleNamespace,
) -> None:
    docker.exit_codes[ENABLE] = "1"
    with pytest.raises(RuntimeError, match=f"{ENABLE} exited 1"):
        _roll()
    assert f"wait:{ENABLE}" in docker.events, "positive control: the one-shot ran"
    assert not any(e.startswith("up:writers:") for e in docker.events)


def test_a_failed_principal_one_shot_stops_before_the_flip(
    docker: SimpleNamespace,
) -> None:
    docker.exit_codes[SCRAM] = "1"
    with pytest.raises(RuntimeError, match=f"{SCRAM} exited 1"):
        _roll()
    assert f"up:oneshot:{ENABLE}" not in docker.events


def test_a_lane_whose_compose_declares_no_sasl_one_shot_starts_none(
    docker: SimpleNamespace,
) -> None:
    docker.declared = f"redpanda\n{WRITER}\n"
    _roll()
    assert not [e for e in docker.events if e.startswith("up:oneshot")]
    assert any(e.startswith("up:writers:") for e in docker.events), (
        "positive control: the roll itself still ran"
    )
