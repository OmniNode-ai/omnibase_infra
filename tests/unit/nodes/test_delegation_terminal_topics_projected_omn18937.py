# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every DECLARED delegation terminal must reach the ledger (OMN-18937).

Why this gate exists
--------------------
A delegation has two terminals, not one. ``node_delegate_skill_orchestrator``
in omnimarket declares both under ``runtime_dispatch.terminal_events``
(``success`` and ``failure``) and publishes both. Every omnimarket-side
projection subscribes to both. The two omnibase_infra consumers subscribed to
the SUCCESS terminal only, so a delegation that FAILED left no trace at all:
chain-canary run 35533286353 (2026-09-20T19:44:30Z, correlation
``b267d3bd-0f60-466e-8c8a-e7c60446e1f0``) put ``delegate-skill-failed.v1`` on
the dev-lane broker and wrote zero rows to ``public.event_ledger``,
``public.ledger_chain`` and ``delegation_workflow_state``.

Nothing detected that for as long as it was true, because both consumer
contracts were internally consistent: each paired its one subscription with
its one routing entry, so the OMN-14594 pairing rule passed, and the chain
writer's read came back empty BY CONSTRUCTION rather than erroneously. The
absent half was invisible to every check that reads only what a contract
declares.

So this gate reads the PRODUCER side and asserts the consumers cover it. The
producer-derived anchor available in this repo is
``src/omnibase_infra/runtime/topics.yaml`` -- the contract-declarative mirror
of ``node_delegate_skill_orchestrator``'s ``published_events`` that OMN-13202
introduced precisely because omnimarket is not a build dependency of
omnibase_infra and its contract cannot be read in CI. That manifest already
generates ``EnumOmnimarketTopic.EVT_DELEGATE_SKILL_*_V1``, so it is the
existing source of truth, not a second one minted here.

This is the local instance of the class epic OMN-18906 (seam-defect
prevention) and its child OMN-18907 (producer-derived fixtures) name: a seam
where the consumer's declaration is self-consistent and still covers less than
the producer emits. No gate from that epic has landed yet -- when a general
producer-derived fixture surface does land, this module is a candidate to fold
into it rather than a second mechanism to keep.

Falsifier for this module itself: if the derivation ever returned an empty or
one-sided set, both assertions below would pass vacuously while the defect
stood. ``test_the_terminal_derivation_is_not_vacuous`` is the positive control
that refuses that outcome, and it is the reason a zero here can be read as
evidence of absence rather than as a check that did not run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TERMINAL_MANIFEST = _REPO_ROOT / "src" / "omnibase_infra" / "runtime" / "topics.yaml"
_NODES = _REPO_ROOT / "src" / "omnibase_infra" / "nodes"
_LEDGER_PROJECTION = _NODES / "node_ledger_projection_compute" / "contract.yaml"
_CHAIN_LEDGER = _NODES / "node_delegation_chain_ledger_effect" / "contract.yaml"

# The delegate-skill chain's terminals are the members of the producer mirror
# that name that command. Matching on the command's own name rather than on a
# ``-completed``/``-failed`` suffix keeps a future third terminal (refused,
# timed-out, cancelled) in scope automatically: the manifest is the thing that
# has to change for a terminal to exist at all.
_DELEGATE_SKILL_TERMINAL_PREFIX = "onex.evt.omnimarket.delegate-skill-"


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _declared_delegation_terminals() -> tuple[str, ...]:
    """Every delegate-skill terminal the PRODUCER mirror declares."""
    manifest = _load(_TERMINAL_MANIFEST)
    return tuple(
        sorted(
            topic
            for topic in manifest["topics"]
            if topic.startswith(_DELEGATE_SKILL_TERMINAL_PREFIX)
        )
    )


def _subscribed(contract: dict[str, Any]) -> set[str]:
    return set(contract["event_bus"]["subscribe_topics"])


def _dispatched(contract: dict[str, Any]) -> set[str]:
    return {entry["topic"] for entry in contract["handler_routing"]["handlers"]}


def test_the_terminal_derivation_is_not_vacuous() -> None:
    """Positive control: the derivation finds BOTH terminals, or it is broken.

    An empty or single-element derivation would make every other assertion in
    this module pass while saying nothing. A zero-finding sweep is evidence
    only when a control proves the sweep can find something.
    """
    terminals = _declared_delegation_terminals()
    assert len(terminals) >= 2, (
        f"the producer mirror {_TERMINAL_MANIFEST} yielded {terminals!r}; a "
        "delegation has at least a success and a failure terminal, so a set "
        "this small means the derivation is broken and every coverage "
        "assertion in this module is passing vacuously"
    )
    assert "onex.evt.omnimarket.delegate-skill-completed.v1" in terminals
    assert "onex.evt.omnimarket.delegate-skill-failed.v1" in terminals


def test_ledger_projection_records_every_declared_delegation_terminal() -> None:
    """AC2/AC3: ``public.event_ledger`` must carry BOTH terminals.

    ``node_ledger_projection_compute`` is the only writer of
    ``public.event_ledger``. A terminal it does not subscribe to can never
    appear in the relation, so every reader downstream -- the chain writer,
    the canary's ledger leg, any forensic query -- is empty by construction
    for that terminal however it behaves.
    """
    contract = _load(_LEDGER_PROJECTION)
    subscribed = _subscribed(contract)
    dispatched = _dispatched(contract)
    terminals = _declared_delegation_terminals()

    unrecorded = [t for t in terminals if t not in subscribed]
    assert not unrecorded, (
        f"node_ledger_projection_compute does not subscribe to {unrecorded!r}, "
        "which the producer mirror declares as delegation terminals; a "
        "delegation reaching one of them leaves zero rows in "
        "public.event_ledger (OMN-18937)"
    )
    undispatched = [t for t in terminals if t not in dispatched]
    assert not undispatched, (
        f"{undispatched!r} are subscribed but carry no handler_routing entry, "
        "which is subscribed-but-never-dispatched (OMN-14594 pairing rule)"
    )


def test_chain_ledger_effect_records_every_declared_delegation_terminal() -> None:
    """AC3: ``public.ledger_chain`` must be reachable from BOTH terminals.

    The chain writer is dispatched BY the terminal event. Subscribed to one
    terminal only, it never runs at all for the other, so no ledger_chain row
    exists for a failed delegation no matter what the projection recorded.

    The terminal must also be DECLARED in ``chain_topology``, as a hop topic
    or as one of that hop's ``alternatives``. Tier 2 grades positionally and
    the canary reads the whole chain's verdict from the last row, so a
    terminal that is subscribed but undeclared grades the terminal row FAIL
    rather than leaving it ungraded.
    """
    contract = _load(_CHAIN_LEDGER)
    subscribed = _subscribed(contract)
    dispatched = _dispatched(contract)
    terminals = _declared_delegation_terminals()

    unsubscribed = [t for t in terminals if t not in subscribed]
    assert not unsubscribed, (
        f"node_delegation_chain_ledger_effect does not subscribe to "
        f"{unsubscribed!r}; the chain writer is never dispatched for a "
        "delegation that reaches that terminal (OMN-18937)"
    )
    undispatched = [t for t in terminals if t not in dispatched]
    assert not undispatched, (
        f"{undispatched!r} are subscribed but carry no handler_routing entry "
        "(OMN-14594 pairing rule)"
    )

    declared: set[str] = set()
    for hop in contract["chain_topology"]:
        declared.add(hop["topic"])
        declared.update(hop.get("alternatives") or ())
    undeclared = [t for t in terminals if t not in declared]
    assert not undeclared, (
        f"{undeclared!r} are dispatched to the chain writer but are not "
        "declared in chain_topology, as a hop topic or as an alternative of "
        "one; an undeclared observed topic grades the terminal row FAIL and "
        "the canary reads its tier-2 verdict from that row (OMN-18937)"
    )
