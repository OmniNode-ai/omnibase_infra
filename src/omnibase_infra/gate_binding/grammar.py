# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read the declared ``Gate:`` binding grammar and resolve a description
against it (OMN-18414).

This module deliberately spells NO pattern of its own. Every regex it compiles
is read from ``omnibase_infra/contracts/gate_binding_grammar.json``, because a
second spelling here is precisely the defect the contract closes: the admission
guard and this consumer each carried a private grammar for the same key, and
the two sets of accepted forms did not overlap at all.
``tests/unit/nodes/node_evidence_autoclose_sweep_effect/test_gate_binding_grammar.py``
asserts the absence by reading this module's own source.

Case-insensitivity arrives as an inline flag inside each declared pattern
rather than as a ``flags`` argument, so nothing here needs
``re.IGNORECASE | re.MULTILINE`` -- a form this repository's union-usage
ratchet miscounts as a type union. The one ``re.MULTILINE`` below is a single
flag and is unaffected.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Final

from omnibase_infra.gate_binding.enum_gate_binding_probe import EnumGateBindingProbe
from omnibase_infra.gate_binding.model_gate_binding import ModelGateBinding
from omnibase_infra.gate_binding.model_gate_binding_grammar import (
    ModelGateBindingGrammar,
)

#: Resolved relative to this file so an installed wheel and a source checkout
#: read the same declaration. A hardcoded absolute path is a cross-machine bug
#: (omni_home CLAUDE.md rule 6) and an environment variable with a default
#: would silently pick a wrong contract (rule 8).
_CONTRACT_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent / "contracts" / "gate_binding_grammar.json"
)


@lru_cache(maxsize=1)
def load_gate_binding_grammar() -> ModelGateBindingGrammar:
    """Read and validate the contract.

    Fails LOUDLY on a missing or malformed contract rather than degrading to a
    built-in default. A resolver that silently falls back to a grammar nobody
    declared is indistinguishable from one reading the contract, which is how a
    gate goes dark without a repo-visible signal.
    """
    text = _CONTRACT_PATH.read_text(encoding="utf-8")
    return ModelGateBindingGrammar.model_validate(json.loads(text))


@lru_cache(maxsize=1)
def _reading_line_pattern() -> re.Pattern[str]:
    return re.compile(load_gate_binding_grammar().line_pattern_reading, re.MULTILINE)


@lru_cache(maxsize=1)
def _normalizations() -> tuple[tuple[re.Pattern[str], str], ...]:
    return tuple(
        (re.compile(rule.pattern), rule.replacement)
        for rule in load_gate_binding_grammar().normalizations
    )


@lru_cache(maxsize=1)
def _compiled_forms() -> tuple[tuple[str, EnumGateBindingProbe, re.Pattern[str]], ...]:
    return tuple(
        (form.id, form.probe, re.compile(form.pattern))
        for form in load_gate_binding_grammar().forms
    )


def gate_binding_line(description: str) -> str:
    """Return the binding text the description declares, empty when it declares none.

    The LAST declaration wins: a description edited to re-point its binding
    should not be judged against the line it replaced.

    An empty return means "this ticket names no binding", which is a different
    fact from "this ticket names one that cannot be read" -- the caller must be
    able to tell those apart, so the two are never collapsed here.
    """
    matches = _reading_line_pattern().findall(description)
    if not matches:
        return ""
    return str(matches[-1]).strip()


def normalize_gate_binding(raw: str) -> str:
    """Apply the declared rewrites, in the order the contract declares them."""
    text = raw
    for pattern, replacement in _normalizations():
        text = pattern.sub(replacement, text)
    return text.strip()


def resolve_gate_binding(description: str) -> ModelGateBinding | None:
    """Resolve a description's binding line against the declared forms.

    ``None`` means one of two things and the caller must distinguish them with
    :func:`gate_binding_line`: the description declared no binding at all, or it
    declared one matching no form. The first is silence; the second is a defect
    and must be held on, never passed.
    """
    raw = gate_binding_line(description)
    if not raw:
        return None
    normalized = normalize_gate_binding(raw)
    if not normalized:
        return None
    for form_id, probe, pattern in _compiled_forms():
        match = pattern.match(normalized)
        if match is None:
            continue
        return ModelGateBinding(
            form=form_id,
            probe=probe,
            raw=raw,
            normalized=normalized,
            groups={k: v for k, v in match.groupdict().items() if v is not None},
        )
    return None


def declared_form_summary() -> str:
    """Render the accepted forms for a refusal message.

    A hold that does not say what WOULD have been readable sends whoever reads
    it to guess, and guessing is how the two grammars drifted apart.
    """
    grammar = load_gate_binding_grammar()
    examples: dict[str, str] = {}
    for fixture in grammar.fixtures.accepted:
        examples.setdefault(fixture.form, fixture.binding)
    return ", ".join(
        f"`{examples[form.id]}`" for form in grammar.forms if form.id in examples
    )
