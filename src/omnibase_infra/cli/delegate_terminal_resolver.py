# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Resolve the delegation terminal a delegate receipt carries (OMN-18569).

``onex delegate`` has to find one thing in its own receipt before it can write
the three customer artifacts: the delegation terminal, i.e. the
``ModelDelegateSkillResponse`` the orchestrator produced. That terminal reaches
the CLI inside one of **two carriers**, and which one depends on where the
orchestrator ran:

* **in-process** (``--locus in-process``) -- the runtime records the terminal
  payload bare, so ``ModelReceiptRuntimeSummary.terminal_payload`` IS the
  delegation terminal.
* **dispatched** (``--locus deployed-lane``) -- the terminal arrives off the
  bus as a full event envelope, so the same field holds the ENVELOPE and the
  delegation terminal sits one level down, under ``payload``.

Before OMN-18569 the CLI looked for an ``attempts`` list directly on
``terminal_payload`` and, finding none on the dispatched path, returned
``None`` and wrote nothing -- silently. A real dispatched run (correlation
``83aa8b6c-8189-49f0-953d-80c8f015ed0a``, 2026-09-17) exited 0 with a correct
answer and produced no ``result.txt``, ``receipt.json`` or ``run.json``, which
is the customer-facing half of the delegation product simply missing on the
path the product is demonstrated from.

:func:`resolve_delegate_terminal` is the ONE accessor both carriers resolve
through. Selection is made by validating two models with DISJOINT REQUIRED
FIELDS, not by probing a dict key for a list: an envelope is identified by
carrying ``envelope_id`` AND a ``payload`` that is itself a delegation
terminal, a bare terminal by carrying ``attempts``. Neither shape can validate
as the other, so there is nothing to guess and no order-dependent tie-break.
When neither validates the resolver RAISES, naming the fields that were absent
-- the silence is what let the defect survive from 2026-09-06 to 2026-09-17,
so the silence is what is removed.

**Layering, stated rather than implied.** The authoritative schema is
``omnimarket``'s ``ModelDelegateSkillResponse`` /
``ModelDelegateSkillAttemptRecord``. ``omnimarket`` sits ABOVE
``omnibase_infra`` and is resolved at runtime by this CLI, never imported at
module scope (see ``cli_delegate``'s own note on ``DELEGATE_NODE_NAME``), so
the models here are a deliberate READ-SIDE MIRROR of the fields this CLI reads,
not a second definition of the wire contract. Two consequences, both intended:
``extra="ignore"``, so a field omnimarket ADDS can never stop a customer's
answer reaching disk; and optional-with-``None`` defaults, so a field
omnimarket REMOVES surfaces in the artifact as an explicit ``null`` rather than
vanishing from it.

.. versionadded:: OMN-18569
"""

from __future__ import annotations

from pydantic import ValidationError

from omnibase_infra.cli.model_delegate_terminal import ModelDelegateTerminal
from omnibase_infra.cli.model_delegate_terminal_envelope import (
    ModelDelegateTerminalEnvelope,
)

__all__ = [
    "DelegateTerminalUnresolvedError",
    "resolve_delegate_terminal",
]


class DelegateTerminalUnresolvedError(ValueError):
    """A delegate receipt carried no resolvable delegation terminal.

    Raised -- never swallowed -- because the caller is the run-file writer, and
    a writer that cannot find its terminal and returns quietly is
    indistinguishable, from outside, from one that had nothing to write.
    """


def _absent_fields(error: ValidationError) -> tuple[str, ...]:
    """Name the required fields a carrier did not supply, in report order.

    Only ``missing`` errors are reported. A type or value error on a field that
    IS present is a different failure and is reported through the raw error
    text instead, because "``attempts`` is absent" and "``attempts`` is a
    string" want different fixes.
    """
    names: list[str] = []
    for detail in error.errors():
        if detail.get("type") != "missing":
            continue
        location = ".".join(str(part) for part in detail.get("loc", ()))
        if location and location not in names:
            names.append(location)
    return tuple(names)


def _describe(error: ValidationError) -> str:
    """Say why one carrier shape did not validate, preferring absent field names."""
    absent = _absent_fields(error)
    if absent:
        return "absent required field(s) " + ", ".join(absent)
    return f"validation failed: {error.errors(include_url=False)}"


def resolve_delegate_terminal(carrier: object) -> ModelDelegateTerminal:
    """Return the delegation terminal, whichever carrier shape it arrived in.

    This is the single accessor for both the in-process (bare) and the
    dispatched (envelope-wrapped) shapes. The two candidate models have
    disjoint required fields, so at most one of them can validate any given
    carrier; the order below is a readability choice, not a tie-break.

    Raises:
        DelegateTerminalUnresolvedError: the carrier is neither shape. The
            message names the absent required fields of BOTH candidates, so the
            failure says what the thing on disk actually was rather than only
            that something was wrong.
    """
    try:
        return ModelDelegateTerminalEnvelope.model_validate(carrier).payload
    except ValidationError as envelope_error:
        as_envelope = _describe(envelope_error)

    try:
        return ModelDelegateTerminal.model_validate(carrier)
    except ValidationError as terminal_error:
        raise DelegateTerminalUnresolvedError(
            f"delegation terminal is neither a bare terminal nor an "
            f"envelope-carried one ({type(carrier).__name__}); "
            f"as a bare terminal, {_describe(terminal_error)}; "
            f"as an envelope, {as_envelope}"
        ) from terminal_error
