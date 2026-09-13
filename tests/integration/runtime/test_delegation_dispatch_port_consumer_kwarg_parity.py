# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Keyword parity between the OmniMarket consumer protocol and the runtime port.

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

pytestmark = pytest.mark.integration

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
