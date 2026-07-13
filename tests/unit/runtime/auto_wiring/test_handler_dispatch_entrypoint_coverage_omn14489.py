# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14135 — dispatch-ENTRYPOINT coverage for every contract-declared handler.

INVARIANT: a handler named in a contract's ``handler_routing.handlers[]`` MUST expose a
callable ``handle`` or ``handle_async``. Auto-wiring binds one of them at wiring time
(``_make_dispatch_callback``); a handler exposing NEITHER is bound to ``_missing_handle``,
which raises ``ModelOnexError`` on EVERY dispatch.

WHY THIS GATE EXISTS (the defect class it makes impossible):
``HandlerA2ATask`` (node_remote_agent_invoke_effect) was contract-declared, wired,
ingress-valid, and CI-green while exposing only ``submit()``/``watch()`` and no dispatch
entrypoint. The A2A remote-agent branch passed ingress validation and died at the FIRST
dispatch. Nothing caught it:

  * contract tests   — assert the contract's declared models, never that the handler can
                       be INVOKED;
  * OMN-14488 route-coverage — proves a payload ROUTES to a registered dispatcher, but a
                       dispatcher bound to ``_missing_handle`` still registers and still
                       routes, and that gate's fake handler HAS a ``handle``;
  * OMN-14488 operation_match ingress coverage — validates the declared ``input_model``
                       against the producer payload, which is upstream of dispatch.

"Registered" and "routable" are NOT "executable". This gate closes that seam.

WHAT THIS FILE IS (and is not): the detector lives in the canonical validator
``omnibase_infra.validators.handler_dispatch_entrypoint``, which is what the CI step and
the pre-commit hook actually execute. These tests import that module rather than
re-implementing the scan, so they exercise THE ARTIFACT THAT RUNS — a test carrying its
own copy of the predicate could stay green while the shipped validator was broken.
"""

from __future__ import annotations

import pytest

from omnibase_infra.validators.handler_dispatch_entrypoint import (
    DEFAULT_BASELINE,
    DEFAULT_SCAN_ROOT,
    MIN_EXPECTED_DECLARED_HANDLERS,
    declared_handlers,
    entrypointless,
    has_dispatch_entrypoint,
    load_baseline,
    main,
)


@pytest.mark.unit
def test_contracts_were_actually_scanned() -> None:
    """Non-vacuity: an empty/failed scan must not make the gate below pass silently."""
    declared = declared_handlers(DEFAULT_SCAN_ROOT)
    assert len(declared) >= MIN_EXPECTED_DECLARED_HANDLERS, (
        f"Only {len(declared)} contract-declared handlers found — the contract scan is "
        f"broken (expected >= {MIN_EXPECTED_DECLARED_HANDLERS} under "
        f"{DEFAULT_SCAN_ROOT}). A gate over an empty set is vacuous."
    )


@pytest.mark.unit
def test_no_new_entrypointless_handlers() -> None:
    """LOAD-BEARING: a contract-declared handler MUST expose handle/handle_async.

    Goes RED against the EXISTS-but-WRONG state: HandlerA2ATask was declared, wired, and
    green while exposing only submit()/watch(). It is deliberately NOT in the baseline.
    """
    live = entrypointless(declared_handlers(DEFAULT_SCAN_ROOT))
    violations = live - load_baseline(DEFAULT_BASELINE)
    assert not violations, (
        "Contract-declared handler(s) expose NEITHER handle() nor handle_async(). "
        "Auto-wiring binds these to _missing_handle, so EVERY dispatch raises "
        "ModelOnexError at runtime while CI stays green:\n"
        + "\n".join(f"  - {c}: {h}" for c, h in sorted(violations))
        + "\n\nAdd a def-B `handle(request) -> response` entrypoint. Do NOT add the "
        f"handler to {DEFAULT_BASELINE} — that baseline is frozen and shrink-only."
    )


@pytest.mark.unit
def test_baseline_is_shrink_only_and_never_stale() -> None:
    """A handler that GAINS an entrypoint must be REMOVED from the baseline.

    This is what makes the baseline a ratchet rather than a permanent exemption: it can
    only shrink, and it can never go stale (a fixed handler still listed = FAIL).
    """
    live = entrypointless(declared_handlers(DEFAULT_SCAN_ROOT))
    stale = load_baseline(DEFAULT_BASELINE) - live
    assert not stale, (
        f"These handlers now HAVE a dispatch entrypoint but are still listed in "
        f"{DEFAULT_BASELINE}. Remove them — the baseline is shrink-only:\n"
        + "\n".join(f"  - {c}: {h}" for c, h in sorted(stale))
    )


@pytest.mark.unit
def test_a2a_task_is_not_baselined() -> None:
    """The handler that exposed this defect class must never be granted an exemption.

    Guards the repair itself: baselining HandlerA2ATask would silently re-open the exact
    hole this ticket closes while every other assertion here stayed green.
    """
    baselined = {handler for _contract, handler in load_baseline(DEFAULT_BASELINE)}
    assert "HandlerA2ATask" not in baselined, (
        "HandlerA2ATask is in the entrypointless baseline. It is the victim that exposed "
        "this defect class (OMN-14489) and must carry a real dispatch entrypoint, not an "
        "exemption."
    )


@pytest.mark.unit
def test_entrypoint_predicate_discriminates() -> None:
    """Discriminator: the predicate must actually distinguish the two states.

    Without this, a predicate that returned True unconditionally would make every
    assertion above vacuously green — the exact failure mode this gate exists to catch.
    """

    class _NoEntrypoint:
        async def submit(self, command: object) -> None: ...  # the HandlerA2ATask shape

    class _DefB:
        async def handle(self, request: object) -> None: ...

    class _AsyncVariant:
        async def handle_async(self, request: object) -> None: ...

    assert not has_dispatch_entrypoint(_NoEntrypoint)
    assert has_dispatch_entrypoint(_DefB)
    assert has_dispatch_entrypoint(_AsyncVariant)


@pytest.mark.unit
def test_validator_cli_passes_on_current_tree() -> None:
    """The shipped CI/pre-commit entrypoint exits 0 on the repo as committed.

    Drives ``main()`` — the same callable the CI step and the pre-commit hook invoke — so
    a validator that crashed or mis-parsed its own baseline cannot ship green.
    """
    assert main([str(DEFAULT_SCAN_ROOT), "--baseline", str(DEFAULT_BASELINE)]) == 0
