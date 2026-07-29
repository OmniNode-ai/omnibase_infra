# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Census ratchet for the runtime-worker (OMN-12988, config-drift OMN-12945).

The stability-test lane's required state includes a running ``runtime-worker``
container (4-container census: main, effects, worker, projection-api). The base
``docker-compose.infra.yml`` used to default the worker to ``replicas: 0`` via a
bare ``${WORKER_REPLICAS:-0}``, so a plain compose ``up``/recreate that did not
set replicas silently dropped the worker — zero errors, zero signal. OMN-14968
made that base line lane-prefixed and fail-closed (``${DEV_WORKER_REPLICAS:?}``)
as well; the census ratchets below stay the runtime-side backstop.

Two ratchets guard against that:

1. ``runtime-worker`` MUST remain in the deploy-agent RUNTIME-scope census so
   ``verify_containers_up`` flags it ``missing`` (deploy failure) rather than
   silently tolerating its absence. A ``replicas: 0`` worker produces no
   container, so ``docker compose ps`` omits it and the census check fails —
   which is the desired loud signal.
2. The stability-test compose override MUST pin the worker to ``replicas: 1`` as
   a literal value (no ``${VAR:-default}`` env indirection), so a lane recreate
   cannot silently scale it to 0 via a stray exported env var.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from deploy_agent.events import SCOPE_SERVICES, Scope, services_for_scope

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
STABILITY_COMPOSE_PATH = REPO_ROOT / "docker" / "docker-compose.stability-test.yml"


def test_runtime_worker_in_runtime_scope_census() -> None:
    """A missing worker must be a deploy failure, not silence.

    ``runtime-worker`` is part of the required runtime census; the deploy-agent
    verifier polls for every service in this list and raises when any is
    missing. A ``replicas: 0`` worker is absent from ``docker compose ps`` and
    is therefore reported missing — the loud signal we want.
    """
    assert "runtime-worker" in SCOPE_SERVICES[Scope.RUNTIME]
    assert "runtime-worker" in services_for_scope(Scope.RUNTIME)
    assert "runtime-worker" in services_for_scope(Scope.FULL)


def test_stability_override_pins_worker_replicas_fail_closed() -> None:
    """The stability override pins worker replicas fail-closed on the policy value.

    A SOFT env indirection (``${STABILITY_TEST_WORKER_REPLICAS:-1}``) is the
    silent-drop surface: an exported ``STABILITY_TEST_WORKER_REPLICAS=0``, or a
    future edit removing the ``:-1`` fallback, scales the worker to 0 with no
    signal.

    OMN-12988 first closed that with a LITERAL ``1``. OMN-12990 then deliberately
    replaced the literal with the ledgered, fail-fast
    ``${STABILITY_TEST_WORKER_REPLICAS:?...}`` form so the value comes from
    ``contracts/services/runtime_policy.contract.yaml`` and a recreate that omits
    the policy env ABORTS instead of dropping the worker. This test kept asserting
    the superseded literal and has been RED since 2026-06-23 — invisible because
    ``scripts/deploy-agent/tests/`` is a separate pytest root that the repo suite
    does not collect. Corrected under OMN-14968, which applied the same fail-closed
    form to the base compose (dev lane) as ``${DEV_WORKER_REPLICAS:?...}``.

    Accepted: the literal ``1``, or a ``:?`` fail-fast reference to the ledgered
    lane value. Rejected: any ``:-`` soft default, and a literal ``0``.
    """
    # The stability override uses docker compose custom tags (!override,
    # !!merge) that yaml.safe_load cannot parse, so assert against the raw text:
    # locate the runtime-worker service block and its deploy.replicas line.
    raw = STABILITY_COMPOSE_PATH.read_text(encoding="utf-8")

    worker_block = re.search(
        r"^  runtime-worker:\n((?:    .*\n|\n)*)",
        raw,
        re.MULTILINE,
    )
    assert worker_block is not None, (
        f"could not locate the runtime-worker service block in {STABILITY_COMPOSE_PATH}"
    )

    replicas_line = re.search(
        r"^      replicas:\s*(.+?)\s*$",
        worker_block.group(1),
        re.MULTILINE,
    )
    assert replicas_line is not None, (
        "stability-test runtime-worker must declare deploy.replicas explicitly; "
        f"none found in:\n{worker_block.group(1)}"
    )

    value = replicas_line.group(1)
    assert value == "1" or value.startswith("${STABILITY_TEST_WORKER_REPLICAS:?"), (
        "stability-test runtime-worker must pin deploy.replicas either to the "
        "literal integer 1 or to the ledgered policy value with the ':?' "
        "fail-fast form (OMN-12990); a soft ':-' default can silently scale the "
        f"worker to 0, got {value!r}"
    )
    assert ":-" not in value, (
        "a soft ':-' default re-opens the silent-drop hole: a recreate that omits "
        f"the policy env would scale the worker without any signal, got {value!r}"
    )
