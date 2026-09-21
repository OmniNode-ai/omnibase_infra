# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Keyword parity between the OmniMarket consumer and the runtime port.

THERE ARE THREE DIRECTIONS AND EACH NEEDS ITS OWN ASSERTION (OMN-18938).
"Parity" reads like one property and is not. Naming only some of them is how
this module was green through a fleet-wide outage in the exact seam it covers,
while the protocol beside it cited this file as mechanical proof.

    1. Everything the consumer DECLARES, this repo must accept.
       ``test_runtime_port_accepts_every_keyword_the_consumer_declares``.
       Catches OMN-18321: a keyword added on the consumer side alone.
    2. Everything the consumer DEFAULTS must stay optional here.
       ``test_consumer_optional_keywords_are_optional_on_the_runtime_port``.
       Catches the inverse drift on a name both sides already know.
    3. Everything this repo REQUIRES, the consumer must actually PASS.
       ``test_every_required_keyword_is_one_the_consumer_passes``.
       Catches OMN-18924/OMN-15504, and NOTHING ELSE DOES.

Direction 3 was missing until OMN-18938, and its absence is not a gap in
coverage so much as a gap in shape. Directions 1 and 2 both take the consumer's
own names as their starting set, so a name the consumer has never heard of is
outside the domain of both -- direction 2 looks the closest and still misses it,
because its filter only considers names the consumer already declares. A
newly-REQUIRED keyword on this side is by definition a name the consumer does
not mention, so it fell through a check written to be exactly about this.

WHAT THAT COST. ``omnibase_infra#3882`` added two required keyword-only
arguments to ``dispatch()``; the deployed consumer passed neither. Every
delegation on the dev lane terminalised ``provider_error`` with
``RuntimeDelegationDispatchPort.dispatch() missing 2 required keyword-only
arguments``, chain canary correlation ``b267d3bd-0f60-466e-8c8a-e7c60446e1f0``
at 19:44Z on 2026-09-20. This module ran 3 passed, 0 failed against that tree.

DIRECTION 3 READS THE CALL SITE, NOT THE PROTOCOL, and that distinction is the
assertion rather than an implementation detail. The consumer's protocol is what
it INTENDS; the ``dispatch(`` call in its handler is what actually executes and
what raises the ``TypeError``. A consumer can declare a keyword in its protocol
and not pass it, which is the failure verbatim.


OMN-18321. On 2026-09-12 OMN-18172 (``omnimarket#2494``, squash ``849fdae6``)
added a ``provenance`` keyword to the consumer protocol
``ProtocolDelegationDispatchPort`` and to the call site in
``handler_delegate_skill.py``. The runtime-owned implementation in THIS repo --
``RuntimeDelegationDispatchPort.dispatch`` -- was not moved in the same change,
so on the deployed bus path the call raised

    TypeError: RuntimeDelegationDispatchPort.dispatch() got an unexpected
    keyword argument 'provenance'

which the consumer handler's own ``except Exception`` turned into a
``delegate-skill-failed`` terminal. Because the dispatch never happened,
``node_delegation_orchestrator`` never ran and never wrote its ``state_io`` row,
so ``delegation_workflow_state`` carried nothing. Every scheduled Chain Canary
run on the ``.201`` dev lane went RED with ``projection_row_absent`` from
2026-09-12T17:44Z. One defect, two symptoms.

WHY NOTHING CAUGHT IT, AND WHY THIS TEST IS SHAPED THE WAY IT IS
    OmniMarket declares its own ``ProtocolDelegationDispatchPort`` AND its own
    class *also named* ``RuntimeDelegationDispatchPort``, which does accept
    ``provenance``. The in-memory delegation seam gate drives that one and
    stayed green throughout. Repo layering puts ``omnimarket`` above
    ``omnibase_infra``, so no type checker on either side ever sees the two
    signatures together, and structural Protocol conformance is never verified
    across the boundary.

    ``tests/fixtures/seams/core_release/dispatch_kwargs_frozen.json`` (OMN-14628)
    freezes the keywords somebody already knew about. It cannot notice a keyword
    the consumer has newly STARTED sending -- which is exactly this defect.

    So this test reads the consumer's declaration as its input rather than a
    hand-maintained list. It parses the consumer source with ``ast`` and never
    imports it: importing ``omnimarket`` from ``omnibase_infra`` would be the
    layering inversion this repo forbids (CLAUDE.md rule 7), and the check must
    work whether or not the package is installed.

FAIL-CLOSED
    Every unresolvable case FAILS. A parity check that skips when it cannot find
    the consumer source reports green on precisely the machine where nobody is
    watching, which is how an empty result becomes a false clean bill of health.
"""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

import pytest

from omnibase_infra.runtime.protocols.protocol_delegation_dispatch_port import (
    ProtocolDelegationDispatchPort,
)
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)

# Deselected from the generic test splits and selected by the job that owns it.
# These tests FAIL CLOSED on an unresolvable consumer source -- which is the whole
# point, and which makes them wrong to collect in a split that never checks that
# source out: they would go red there for a reason that has nothing to do with the
# parity they assert. Same shape as `live_github_api` (OMN-16096). Deselected in the
# splits is NOT skipped: `consumer-kwarg-parity` in delegation-seam-gate.yml and the
# always_run `onex-delegation-dispatch-consumer-kwarg-parity` pre-commit hook both
# run them unconditionally, and both provide the source.
pytestmark = [pytest.mark.integration, pytest.mark.cross_repo_consumer]

_CONSUMER_RELPATH = Path(
    "src/omnimarket/nodes/node_delegate_skill_orchestrator/handlers"
    "/handler_delegate_skill.py"
)
_CONSUMER_PROTOCOL = "ProtocolDelegationDispatchPort"
_CONSUMER_METHOD = "dispatch"
_CONSUMER_SOURCE_ENV = "ONEX_DELEGATION_CONSUMER_SOURCE"
_SIBLING_ENV = "OMNI_HOME"


def _resolve_consumer_source() -> Path:
    """Locate the OmniMarket consumer source, or fail naming every path tried.

    Resolution order, most explicit first:

    1. ``ONEX_DELEGATION_CONSUMER_SOURCE`` -- the file itself. CI sets this
       after a sparse checkout of the consumer repo, the same mechanism
       ``chain-canary.yml`` already uses to read ``config/ci_bus_lanes.yaml``
       from that repo.
    2. ``$OMNI_HOME/omnimarket/<relpath>`` -- the canonical sibling clone, which
       is what a local pre-commit run resolves.
    3. The installed ``omnimarket`` distribution, when one is importable.

    A miss is an error, never a skip.
    """
    tried: list[str] = []

    explicit = os.environ.get(_CONSUMER_SOURCE_ENV, "").strip()
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            return candidate
        tried.append(f"{_CONSUMER_SOURCE_ENV}={explicit} (not a file)")

    omni_home = os.environ.get(_SIBLING_ENV, "").strip()
    if omni_home:
        candidate = Path(omni_home) / "omnimarket" / _CONSUMER_RELPATH
        if candidate.is_file():
            return candidate
        tried.append(f"${_SIBLING_ENV}/omnimarket/{_CONSUMER_RELPATH} (absent)")
    else:
        tried.append(f"${_SIBLING_ENV} is unset")

    try:
        import omnimarket
    except ImportError:
        tried.append("omnimarket is not importable in this interpreter")
    else:
        for root in getattr(omnimarket, "__path__", []):
            candidate = (
                Path(root)
                / "nodes"
                / "node_delegate_skill_orchestrator"
                / "handlers"
                / "handler_delegate_skill.py"
            )
            if candidate.is_file():
                return candidate
            tried.append(f"{candidate} (absent)")

    raise AssertionError(
        "cannot resolve the OmniMarket delegation consumer source, so keyword "
        "parity cannot be proven. This FAILS rather than skipping: a parity "
        "check that skips when it cannot see the consumer reports green on the "
        "one machine where the drift would land. Set "
        f"{_CONSUMER_SOURCE_ENV} to the file, or ${_SIBLING_ENV} to the "
        "workspace root holding the omnimarket clone. Paths tried: " + "; ".join(tried)
    )


def _consumer_declared_keywords(source: Path) -> dict[str, bool]:
    """Keyword-only parameter names the consumer protocol declares.

    Returns ``{name: has_default}``. Parsed with ``ast`` -- the consumer module
    is never imported, because importing it here would invert the repo layering.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != _CONSUMER_PROTOCOL:
            continue
        for member in node.body:
            if (
                isinstance(member, (ast.AsyncFunctionDef, ast.FunctionDef))
                and member.name == _CONSUMER_METHOD
            ):
                args = member.args
                defaults = list(args.kw_defaults)
                return {
                    arg.arg: defaults[index] is not None
                    for index, arg in enumerate(args.kwonlyargs)
                }

    raise AssertionError(
        f"{source} declares no {_CONSUMER_PROTOCOL}.{_CONSUMER_METHOD}. The "
        "consumer protocol moved or was renamed; repoint this check in the "
        "same change rather than deleting it."
    )


@pytest.mark.parametrize(
    "dispatch_method",
    [
        pytest.param(ProtocolDelegationDispatchPort.dispatch, id="protocol"),
        pytest.param(RuntimeDelegationDispatchPort.dispatch, id="implementation"),
    ],
)
def test_runtime_port_accepts_every_keyword_the_consumer_declares(
    dispatch_method: object,
) -> None:
    """Both halves of this repo's boundary accept the consumer's full keyword set.

    The consumer's declaration is the input. A keyword added on that side and
    not here is the OMN-18321 defect, and it is a live-lane outage rather than a
    type error, because the call site is only ever resolved at runtime.
    """
    source = _resolve_consumer_source()
    declared = _consumer_declared_keywords(source)
    assert declared, f"{source} declares a dispatch() with no keyword parameters"

    accepted = set(inspect.signature(dispatch_method).parameters)  # type: ignore[arg-type]
    missing = sorted(name for name in declared if name not in accepted)

    assert not missing, (
        f"{dispatch_method.__qualname__} does not accept "  # type: ignore[attr-defined]
        f"{missing}, which {source.name}'s {_CONSUMER_PROTOCOL} declares and its "
        "handler passes on every delegation. On the deployed bus path this is a "
        "TypeError the consumer swallows into a failed terminal, so the chain "
        "dies silently and the FSM row is never written (OMN-18321). Add the "
        "keyword here and thread it onto the published payload in the same "
        "change -- accepting it and dropping it is the silent-drop defect "
        "OMN-18172 exists to close."
    )


def test_consumer_optional_keywords_are_optional_on_the_runtime_port() -> None:
    """A keyword the consumer may omit must not become required here.

    The inverse drift: the consumer defaults a keyword, this repo makes it
    mandatory, and every caller that omits it fails the same invisible way.
    """
    source = _resolve_consumer_source()
    declared = _consumer_declared_keywords(source)
    parameters = inspect.signature(RuntimeDelegationDispatchPort.dispatch).parameters

    required_here = sorted(
        name
        for name, has_default in declared.items()
        if has_default
        and name in parameters
        and parameters[name].default is inspect.Parameter.empty
    )

    assert not required_here, (
        f"{required_here} carry a default in {source.name}'s {_CONSUMER_PROTOCOL} "
        "but are required by RuntimeDelegationDispatchPort.dispatch. A consumer "
        "that omits one raises TypeError on the deployed bus path (OMN-18321)."
    )


_CONSUMER_CALL_ATTRIBUTE = "_dispatch_port"


def _consumer_passed_keywords(source: Path) -> set[str]:
    """Keyword names the consumer's handler actually PASSES at its call site.

    Deliberately NOT the same input as ``_consumer_declared_keywords``. The
    protocol declaration states intent; this states what runs. A newly-required
    keyword on our side raises ``TypeError`` against what the handler passes,
    whatever its protocol says.

    Finds every ``<something>._dispatch_port.dispatch(...)`` call and unions
    their keywords. A ``**kwargs`` splat at a call site makes the passed set
    unknowable by reading, so it FAILS rather than returning a set it cannot
    stand behind -- the same fail-closed posture as an unresolvable source.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    passed: set[str] = set()
    found = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != _CONSUMER_METHOD:
            continue
        inner = func.value
        if (
            not isinstance(inner, ast.Attribute)
            or inner.attr != _CONSUMER_CALL_ATTRIBUTE
        ):
            continue
        found = True
        for keyword in node.keywords:
            if keyword.arg is None:
                raise AssertionError(
                    f"{source} splats **kwargs into its {_CONSUMER_METHOD}() call, "
                    "so the keywords it passes cannot be read statically and this "
                    "direction cannot be proven. Pass them explicitly, or move "
                    "this check to something that can see the runtime call."
                )
            passed.add(keyword.arg)

    if not found:
        raise AssertionError(
            f"{source} contains no self.{_CONSUMER_CALL_ATTRIBUTE}."
            f"{_CONSUMER_METHOD}(...) call. The consumer's call site moved or was "
            "renamed; repoint this check in the same change rather than deleting "
            "it -- an empty result here is indistinguishable from parity."
        )
    return passed


@pytest.mark.parametrize(
    "dispatch_method",
    [
        pytest.param(ProtocolDelegationDispatchPort.dispatch, id="protocol"),
        pytest.param(RuntimeDelegationDispatchPort.dispatch, id="implementation"),
    ],
)
def test_every_required_keyword_is_one_the_consumer_passes(
    dispatch_method: object,
) -> None:
    """Direction 3. Nothing here may be required that the consumer does not pass.

    The only direction that can see a keyword added as REQUIRED on this side,
    because it starts from THIS repo's parameter list instead of the consumer's.
    Both directions above start from the consumer's names, so a name the
    consumer has never heard of is outside their domain entirely.

    The remedy when this fails is a defaulted keyword, not a consumer edit: the
    two repos deploy independently, so a required argument is broken for as long
    as the consumer is one release behind, which is always.
    """
    source = _resolve_consumer_source()
    passed = _consumer_passed_keywords(source)

    parameters = inspect.signature(dispatch_method).parameters  # type: ignore[arg-type]
    unmet = sorted(
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        and name not in passed
    )

    assert not unmet, (
        f"{dispatch_method.__qualname__} REQUIRES {unmet}, which "  # type: ignore[attr-defined]
        f"{source.name} does not pass at its dispatch call site. On the deployed "
        "bus path this raises TypeError: dispatch() missing required keyword-only "
        "arguments, the consumer swallows it into a failed terminal, and every "
        "delegation on the lane dies with provider_error (OMN-18924, chain canary "
        "b267d3bd-0f60-466e-8c8a-e7c60446e1f0). Give the keyword a default here "
        "rather than requiring the consumer to catch up -- the two repos deploy "
        "independently and the consumer is behind by construction."
    )
