# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed manifest fetching with a shared, bounded retry budget (OMN-16753).

Both refresh verifiers fetch ``/v1/introspection/manifest`` and both need the
same boot tolerance, so both of them live here rather than being copied. Two
review findings on omnibase_infra#3334 are the reason this module exists at
all, and each is a property of the code below rather than a convention:

**Retry eligibility is a typed signal, never an error-string prefix.** The
first revision decided whether a failure was retriable with
``err.startswith("manifest fetch failed")`` -- a string-typed contract between
two private functions in two different modules. Rewording either message would
have silently stopped every retry and regressed a booting lane back to
``INFRA_ERROR``, which is the exact defect OMN-16753 closes, with no test
failing. :class:`EnumManifestFetchFailure` is that decision now: ``TRANSPORT``
is retriable because waiting can fix it, ``UNREADABLE`` is terminal because a
200 carrying a non-manifest body will not become one by waiting.

**The retry window is bounded ONCE for the whole gate run, not per URL.** The
first revision gave each URL its own 24 x 15 s window, and
``run_health_gate`` fetches the main manifest and then the effects manifest
serially -- a worst case near 720 s of manifest retrying before the health leg
had even started its own 360 s wait. :class:`RetryBudget` is a single
monotonic deadline shared by every fetch in one run: the second manifest
inherits whatever the first left, so the total can never exceed
:data:`MANIFEST_FETCH_WINDOW_SECONDS`, which is sized to the health leg's own
window rather than to a multiple of it. The per-attempt socket timeout is
clamped to the remaining budget for the same reason -- a bound that only
counts sleeps is not a bound on wall clock.

Related Tickets:
    - OMN-16753: this ticket
    - OMN-15837: the declared-groups derivation the effects manifest feeds
    - OMN-17624: the bounded health-verdict wait whose window this matches
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import Protocol

#: Total wall-clock ceiling shared by EVERY manifest fetch inside one gate run.
#: Deliberately equal to the health leg's own derived wait (24 x 15 s) rather
#: than a per-URL allowance: the manifest fetch and the health probe are
#: waiting on the SAME runtime to finish booting, so waiting longer than the
#: health probe would buys nothing and only delays a verdict a human is
#: standing by for.
MANIFEST_FETCH_WINDOW_SECONDS = 360.0

#: Gap between attempts inside that window.
MANIFEST_FETCH_INTERVAL_SECONDS = 15.0

#: Per-attempt socket timeout, clamped down to whatever budget remains.
MANIFEST_FETCH_TIMEOUT_SECONDS = 10.0

#: Attempt ceiling for the whole run, shared exactly like the deadline is.
#: The deadline is the bound that matters, but it is measured on a clock the
#: caller can inject, and a caller that injects a no-op ``sleep_fn`` without
#: also injecting the clock would otherwise spin against wall time. Two
#: independent bounds, whichever binds first; the loop cannot outrun both.
MANIFEST_FETCH_MAX_ATTEMPTS = 24

#: How many failed attempts are carried into the receipt. The history exists so
#: a reader can tell a persistent failure from an intermittent one; the whole
#: list on a fully expired window is 24 near-identical lines, which is a log
#: dump rather than evidence.
MANIFEST_FETCH_HISTORY_LIMIT = 5


class ManifestResponse(Protocol):
    """The shape both ``urlopen`` and every test double already satisfy."""

    def read(self) -> bytes: ...

    def __enter__(self) -> ManifestResponse: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


#: A ``urlopen``-shaped callable. Typed rather than ``object``, so the call
#: below needs no ``type: ignore`` to say what every caller already passes.
ManifestOpener = Callable[..., ManifestResponse]


class EnumManifestFetchFailure(str, Enum):
    """Why a manifest fetch did not produce a payload.

    The retry decision reads THIS, never the rendered message. A reworded
    message must keep retrying; a genuinely different failure must not start.
    """

    #: The runtime did not answer: connection reset/refused, DNS, timeout. The
    #: one class a bounded wait can fix, because it is what a booting lane
    #: looks like from outside.
    TRANSPORT = "transport"

    #: The runtime answered with something that is not a manifest: not JSON, or
    #: JSON of a shape no manifest has. Terminal -- waiting cannot turn a
    #: served error page into a contract list.
    UNREADABLE = "unreadable"

    #: The runtime answered with an HTTP status that will not become a manifest
    #: by waiting -- a 404/410/401 says this URL is wrong or forbidden, not that
    #: the process is still booting. Split out of TRANSPORT because
    #: ``HTTPError`` is a ``URLError`` subclass: a broad catch would spend the
    #: entire window re-asking a question already answered.
    HTTP_TERMINAL = "http_terminal"

    #: The shared window was already spent by an earlier fetch in this run, so
    #: this URL was never attempted. Reported as its own class rather than
    #: dressed up as a transport error, because "we did not ask" and "it did
    #: not answer" are different facts and a receipt should be able to say
    #: which one it saw. Fail-closed either way: it is still an error.
    BUDGET_EXPIRED = "budget_expired"


@dataclass(frozen=True)
class ManifestFetchOutcome:
    """The typed result of one manifest fetch attempt."""

    payload: dict[str, object] | None = None
    failure: EnumManifestFetchFailure | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True iff a manifest payload was obtained."""
        return self.failure is None and self.payload is not None

    @property
    def retriable(self) -> bool:
        """True iff waiting could plausibly change this outcome."""
        return self.failure is EnumManifestFetchFailure.TRANSPORT


@dataclass
class RetryBudget:
    """One monotonic deadline shared by every manifest fetch in a gate run.

    Started lazily on first use so constructing it costs nothing, and driven
    through injected ``sleep_fn``/``monotonic_fn`` so a test can pin the total
    bound without sleeping through it.
    """

    total_seconds: float = MANIFEST_FETCH_WINDOW_SECONDS
    interval_seconds: float = MANIFEST_FETCH_INTERVAL_SECONDS
    max_attempts: int = MANIFEST_FETCH_MAX_ATTEMPTS
    sleep_fn: Callable[[float], None] = time.sleep
    monotonic_fn: Callable[[], float] = time.monotonic
    slept_seconds: float = field(default=0.0, init=False)
    attempts_made: int = field(default=0, init=False)
    _deadline: float | None = field(default=None, init=False)

    def remaining_seconds(self) -> float:
        """Budget left, starting the clock on the first call."""
        if self._deadline is None:
            self._deadline = self.monotonic_fn() + self.total_seconds
        return max(0.0, self._deadline - self.monotonic_fn())

    def attempt_timeout_seconds(
        self, ceiling: float = MANIFEST_FETCH_TIMEOUT_SECONDS
    ) -> float:
        """Socket timeout for the next attempt, clamped to the budget.

        Clamped rather than fixed because a bound that counts only sleeps is
        not a bound on wall clock: a hung socket would otherwise add its full
        timeout on top of an already-spent window.
        """
        remaining = self.remaining_seconds()
        return ceiling if self.attempts_made == 0 else min(ceiling, remaining)

    def may_attempt(self) -> bool:
        """Whether another fetch attempt fits inside the shared window.

        The FIRST attempt of a run always fits -- a budget nobody has spent yet
        is not a reason to report a failure nobody tried. After that the
        deadline is absolute, which is what makes the total bound exact rather
        than "the window plus one more timeout per URL".
        """
        if self.attempts_made == 0:
            return True
        return self.attempts_made < self.max_attempts and self.remaining_seconds() > 0.0

    def sleep_before_retry(self) -> bool:
        """Sleep one interval iff the shared budget can still afford it.

        A sleep that would consume the ENTIRE remainder is refused rather than
        shortened: it would buy no attempt, and returning from it with an
        expired window would replace the real last error with a bookkeeping
        one. Returning False here is the loop's only exit besides success and
        a terminal failure, so expiry fails closed with the failure it
        actually observed.
        """
        if self.attempts_made >= self.max_attempts:
            return False
        if self.remaining_seconds() <= self.interval_seconds:
            return False
        self.sleep_fn(self.interval_seconds)
        self.slept_seconds += self.interval_seconds
        return True


def fetch_manifest_once(
    manifest_url: str,
    *,
    opener: ManifestOpener | None = None,
    timeout_seconds: float = MANIFEST_FETCH_TIMEOUT_SECONDS,
) -> ManifestFetchOutcome:
    """Fetch one ``/v1/introspection/manifest`` payload, classifying failures.

    Split out of ``check_manifest_count`` by OMN-15837 so the SAME fetched
    payload feeds both the contract-count floor and the consumer-group
    derivation -- the declared set must describe the image that is actually
    running, and a second fetch could observe a different one.
    """
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(manifest_url, timeout=timeout_seconds) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        # An HTTP status IS an answer, and only some answers are worth waiting
        # on. 5xx and 429 are what a booting or overloaded runtime returns, so
        # they stay retriable; every other status is terminal, because
        # re-asking a 404 for six minutes delays the verdict without changing
        # it. ``HTTPError`` must be caught BEFORE ``URLError`` -- it is a
        # subclass, and the broad catch alone put every status in TRANSPORT.
        retriable = exc.code >= 500 or exc.code == 429
        return ManifestFetchOutcome(
            failure=(
                EnumManifestFetchFailure.TRANSPORT
                if retriable
                else EnumManifestFetchFailure.HTTP_TERMINAL
            ),
            error=f"manifest fetch failed ({manifest_url}): HTTP {exc.code} {exc.reason}",
        )
    except (urllib.error.URLError, OSError) as exc:
        return ManifestFetchOutcome(
            failure=EnumManifestFetchFailure.TRANSPORT,
            error=f"manifest fetch failed ({manifest_url}): {exc}",
        )
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        return ManifestFetchOutcome(
            failure=EnumManifestFetchFailure.UNREADABLE,
            error=f"manifest not valid JSON ({manifest_url}): {exc}",
        )
    if isinstance(payload, list):
        return ManifestFetchOutcome(payload={"contracts": payload})
    if isinstance(payload, dict):
        return ManifestFetchOutcome(payload=payload)
    return ManifestFetchOutcome(
        failure=EnumManifestFetchFailure.UNREADABLE,
        error=f"manifest payload has unexpected shape ({manifest_url})",
    )


@dataclass(frozen=True)
class ManifestFetchReport:
    """The outcome of a budgeted fetch, plus the failures it went through."""

    outcome: ManifestFetchOutcome
    #: Every failed attempt's message, oldest first. Kept because a first
    #: attempt that reset and a last attempt that timed out are a different
    #: story from twenty-four identical resets, and the receipt could not tell
    #: them apart while only the last error survived.
    history: tuple[str, ...] = ()

    @property
    def payload(self) -> dict[str, object] | None:
        return self.outcome.payload

    @property
    def error(self) -> str | None:
        return self.outcome.error

    def recent_history(
        self, limit: int = MANIFEST_FETCH_HISTORY_LIMIT
    ) -> tuple[str, ...]:
        """The last ``limit`` failure messages, for the log and the receipt."""
        return self.history[-limit:] if limit > 0 else ()


def fetch_manifest_with_budget(
    manifest_url: str,
    *,
    opener: ManifestOpener | None = None,
    budget: RetryBudget,
) -> ManifestFetchReport:
    """Fetch a manifest, retrying transport failures until the budget expires.

    The budget is shared, so a second call with the same object inherits only
    what the first left. Expiry returns the last failure -- a runtime that
    never served a manifest inside the window is a genuine finding, never a
    reason to pass.
    """
    history: list[str] = []
    while True:
        if not budget.may_attempt():
            expired = ManifestFetchOutcome(
                failure=EnumManifestFetchFailure.BUDGET_EXPIRED,
                error=(
                    f"manifest fetch window expired before this URL was "
                    f"attempted ({manifest_url}); the shared "
                    f"{budget.total_seconds:.0f}s / "
                    f"{budget.max_attempts}-attempt budget was spent by an "
                    f"earlier fetch in this run"
                ),
            )
            history.append(expired.error or "")
            return ManifestFetchReport(outcome=expired, history=tuple(history))
        outcome = fetch_manifest_once(
            manifest_url,
            opener=opener,
            timeout_seconds=budget.attempt_timeout_seconds(),
        )
        budget.attempts_made += 1
        if outcome.ok:
            return ManifestFetchReport(outcome=outcome, history=tuple(history))
        history.append(outcome.error or "manifest fetch failed with no message")
        if not outcome.retriable or not budget.sleep_before_retry():
            return ManifestFetchReport(outcome=outcome, history=tuple(history))


def count_manifest_contracts(payload: dict[str, object]) -> int:
    """Count the contracts a manifest payload declares."""
    contracts = payload.get("contracts", [])
    return len(contracts) if isinstance(contracts, list) else 0


def fetch_manifest_contract_count(
    manifest_url: str,
    *,
    opener: ManifestOpener | None = None,
    budget: RetryBudget,
) -> tuple[int | None, str | None, tuple[str, ...]]:
    """Fetch a manifest and count its contracts, within the shared budget.

    Lives here rather than in either verifier so the two gates cannot drift
    apart again -- the same reason this module exists. Renamed from
    ``check_manifest_count``, which never checked anything: it discarded its
    ``min_contracts`` argument and returned the raw count while the caller did
    the floor comparison, a signature promising a check its body did not
    perform. The floor stays with the caller that owns the report field.

    Returns:
        ``(count, error, failure_history)`` -- the history is every failed
        attempt's message, so a persistent failure is distinguishable from an
        intermittent one in the receipt.
    """
    fetched = fetch_manifest_with_budget(manifest_url, opener=opener, budget=budget)
    history = fetched.recent_history()
    if fetched.error is not None or fetched.payload is None:
        return (
            None,
            fetched.error or f"manifest fetch returned no payload ({manifest_url})",
            history,
        )
    return count_manifest_contracts(fetched.payload), None, history
