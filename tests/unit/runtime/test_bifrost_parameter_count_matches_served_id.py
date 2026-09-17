# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18570: tie ``parameter_count`` to the id the endpoint actually serves.

This is the third field in ``AuthorizedLabBinding`` to drift on the .201 rungs,
and the first one nothing was watching.

OMN-16419 and OMN-16999 between them established that ``served_model_id`` and
``context_window`` are probe results, and
``test_bifrost_served_model_probe_fixture.py`` pins both to a recorded
``/v1/models`` readback. ``parameter_count`` was left out of that pass, because
``/v1/models`` does not report a parameter count — so there was no external
referent to pin it to and it was treated as a free-text declaration.

The consequence showed up on 2026-09-17. Both .201 rungs carried
``parameter_count="27B"`` while advertising ``served_model_id="Qwen3.6-35B-A3B"``
— a leftover from the window (2026-08-23 to 2026-09-03) when the RTX 5090 really
did serve a Qwen3.8 **27B** under SGLang. OMN-16999 corrected the served id when
the box was redeployed to vLLM on the 35B and left the parameter count reading
from the retired deployment. Every existing test stayed green, in perfect
agreement, about a number that was wrong by 8B.

Nothing refuses a call over this field, so the cost is not a broken rung — it is
a contract that misdescribes the hardware to every human and every surface that
reads it. That is not hypothetical: it is the most plausible origin of the
2026-09-17 operator question "why are you using the 35b model instead of qwen
3.8 27b on the 5090", asked about a box that had been serving the 35B for two
weeks.

THE EXTERNAL REFERENT THIS USES. A parameter count cannot be probed, but it does
not have to be invented either: the served id names it. ``Qwen3.6-35B-A3B`` is a
mixture-of-experts id in the ``<total>B-A<active>B`` form the vendor publishes —
35B total, 3B active per token. So where the served id carries a parameter
figure, that figure IS the referent, and ``parameter_count`` restating it
differently is a contradiction inside a single row.

Where the served id carries no figure the rule does not bind and the field stays
a declaration — ``deepseek-v4-flash`` is the standing example, whose 284B comes
from the OMN-12492 contract and not from its id. Skipping is deliberate: a test
that demanded a figure there would push someone to encode a guess.
"""

from __future__ import annotations

import re

import pytest

from omnibase_infra.runtime.models.model_bifrost_lane_backend_binding import (
    _AUTHORIZED_BINDINGS,
    ACTIVE_BACKEND_KEYS,
)

pytestmark = pytest.mark.unit

#: A parameter figure inside a model id: ``35B``, ``3B``, ``1.5B``.
#:
#: The lookbehind stops the version segment of ``Qwen3.6`` from being read as a
#: parameter count, and keeps the ``3B`` of an ``A3B`` active-parameter suffix
#: (preceded by ``A``, not by a digit or a dot) matchable.
_PARAM_FIGURE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)B\b", re.IGNORECASE)


def _expected_parameter_count(served_model_id: str) -> str | None:
    """The parameter count a served id states about itself, or None.

    The first figure is total parameters; a second, when present, is the
    active-per-token count of a mixture-of-experts model and is rendered back in
    the vendor's own ``-A<active>B`` form.
    """
    figures = _PARAM_FIGURE.findall(served_model_id)
    if not figures:
        return None
    total = f"{figures[0]}B"
    if len(figures) >= 2:
        return f"{total}-A{figures[1]}B"
    return total


def test_the_referent_helper_reads_a_moe_id_and_declines_an_opaque_one() -> None:
    """Positive control for the rule itself (CLAUDE.md rule 16).

    Every assertion below is a comparison against ``_expected_parameter_count``.
    If that helper silently returned ``None`` for everything, each parametrized
    case would skip and the suite would report all-green while checking nothing
    — a zero that reads exactly like a pass. This pins both branches so the
    skips below are trustworthy.
    """
    assert _expected_parameter_count("Qwen3.6-35B-A3B") == "35B-A3B"
    assert _expected_parameter_count("Qwen3.8-27B") == "27B"
    assert _expected_parameter_count("mistral-7B-instruct") == "7B"
    assert _expected_parameter_count("Qwen2.5-1.5B") == "1.5B"
    # No figure to read: the rule must decline rather than guess.
    assert _expected_parameter_count("deepseek-v4-flash") is None
    assert _expected_parameter_count("gpt-4o") is None


@pytest.mark.parametrize("backend_key", sorted(ACTIVE_BACKEND_KEYS))
def test_parameter_count_agrees_with_the_served_model_id(backend_key: str) -> None:
    """A row may not state a parameter count its own served id contradicts."""
    binding = _AUTHORIZED_BINDINGS[backend_key]
    expected = _expected_parameter_count(binding.served_model_id)
    if expected is None:
        pytest.skip(
            f"{binding.served_model_id!r} carries no parameter figure, so "
            f"parameter_count {binding.parameter_count!r} stays a declaration"
        )
    assert binding.parameter_count == expected, (
        f"{backend_key!r} declares parameter_count "
        f"{binding.parameter_count!r}, but its own served_model_id "
        f"{binding.served_model_id!r} says {expected!r}. One of the two is "
        "left over from a previous deployment of this endpoint — re-probe "
        "GET /v1/models, bind the id it reports, and restate the parameter "
        "count from that id in the same commit. This is the OMN-16419 / "
        "OMN-16999 / OMN-18570 drift class, now on its third field."
    )
