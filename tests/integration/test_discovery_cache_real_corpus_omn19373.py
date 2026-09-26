# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The memo, measured against the contracts actually installed here (OMN-19373).

The unit half of this change drives ``discover_contracts()`` with a single
fabricated entry point. That proves the memo's logic — validated by entry-point
identity and by contract-file mtime and size — but it cannot prove the premise
this ticket rests on, because it never walks the real corpus.

This file does. It calls the real ``discover_contracts()`` over the
``onex.nodes`` entry points installed in this environment, reading the real
``contract.yaml`` beside each, and asserts that a repeat scan parses nothing.

WHY IT COUNTS PARSES RATHER THAN SECONDS. The duration is a property of how
many contracts happen to be installed, not of the code: 24.036s for 544 entry
points in the deployed ``omninode-runtime-effects`` pod on onex-dev
(2026-09-26), 0.943s for the 144 in a local checkout, and milliseconds on a CI
image carrying a handful. A threshold in seconds would therefore be either
flaky or vacuous depending on the runner. The quantity the sweep's cost is
actually made of is ``_parse_contract`` calls, and that is invariant: before
this change the second scan re-parsed every contract, after it the second scan
parses none.

WHERE IT REFUSES TO PASS. If the real corpus is empty, every "did not
re-parse" assertion below would hold vacuously, so the test skips with the
count rather than reporting green.
"""

from __future__ import annotations

import time

import pytest

from omnibase_infra.runtime.auto_wiring import discovery as discovery_mod
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts

pytestmark = pytest.mark.integration


def discover_contracts_cache_clear() -> None:
    """Clear the memo if this build has one.

    Resolved at call time rather than imported at module scope so that running
    this file against a build without the memo fails on the BEHAVIOUR under
    test -- a repeat scan that re-parses -- instead of on an ImportError, which
    would prove only that a symbol is missing.
    """
    clear = getattr(discovery_mod, "discover_contracts_cache_clear", None)
    if clear is not None:
        clear()


class _ParseCounter:
    """Wraps the real ``_parse_contract`` and counts genuine parses."""

    def __init__(self) -> None:
        self._real = discovery_mod._parse_contract
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        return self._real(*args, **kwargs)


@pytest.fixture(autouse=True)
def _isolate_memo():
    discover_contracts_cache_clear()
    yield
    discover_contracts_cache_clear()


def test_the_real_repeat_scan_parses_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = _ParseCounter()
    monkeypatch.setattr(discovery_mod, "_parse_contract", counter)

    t0 = time.perf_counter()
    first = discover_contracts()
    first_seconds = time.perf_counter() - t0
    parsed_on_first = counter.calls

    if parsed_on_first == 0:
        pytest.skip(
            "no onex.nodes contracts are installed in this environment, so a "
            "'did not re-parse' assertion would hold vacuously "
            f"(discovered={first.total_discovered}, errors={first.total_errors})"
        )

    t1 = time.perf_counter()
    second = discover_contracts()
    second_seconds = time.perf_counter() - t1

    assert counter.calls == parsed_on_first, (
        f"the repeat scan re-parsed {counter.calls - parsed_on_first} of "
        f"{parsed_on_first} real contracts; the sweep's cost is unchanged. "
        f"first={first_seconds:.3f}s second={second_seconds:.3f}s"
    )
    # The memo must return the same answer, not merely return quickly.
    assert second.total_discovered == first.total_discovered
    assert second.total_errors == first.total_errors
    assert [c.name for c in second.contracts] == [c.name for c in first.contracts]


def test_clearing_the_memo_reparses_the_real_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positive control for the test above.

    It proves the counter is wired to the real scan, so the zero-growth
    assertion means "the memo served it" rather than "nothing was measured".
    """
    counter = _ParseCounter()
    monkeypatch.setattr(discovery_mod, "_parse_contract", counter)

    discover_contracts()
    parsed_on_first = counter.calls
    if parsed_on_first == 0:
        pytest.skip("no onex.nodes contracts are installed in this environment")

    discover_contracts_cache_clear()
    discover_contracts()

    assert counter.calls == parsed_on_first * 2, (
        "after an explicit cache clear the real corpus must be parsed again; "
        f"parse calls went {parsed_on_first} -> {counter.calls}"
    )
