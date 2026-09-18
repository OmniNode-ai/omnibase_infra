# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18757: the verbatim replay flags on scripts/ledger_lock.py.

WHY THIS FILE EXISTS AT ALL. OMN-18433 shipped ``--replay-before`` and
``--replay-source`` on 2026-09-16 and used them in anger: five ``| REPLAY |``
marker rows stamped ``2026-09-16T11:26:21Z`` are in the live rolling work
ledger, landed by ``omni_home#326``. They existed only in the GITIGNORED
``omni_home`` copy of the tool, so no commit ever carried them, and when
OMN-18554 ported that copy into this committed one (``omnibase_infra#3688``,
squash ``80dc408ba``) it carried the six guards and not the two flags. Three
tests in ``omni_home``'s ``tests/test_ledger_stranded_clone.py`` went red and
stayed red for two days, and nothing in THIS repository noticed, because
nothing in this repository tested the flags.

So the point of this module is not only that the flags work. It is that the
next port of this file cannot drop them silently a second time. Deleting the
two ``parser.add_argument`` calls turns this file red in this repo's own CI,
which is the loop that was missing.

TWO SUBJECTS, AND WHERE EACH LIVES. The deciding logic --
``replay_refusal()`` and ``replay_marker_row()`` -- is in
``docs/workflows/_shared/stranded_clone_guard.py``, which is tracked in
``omni_home`` and unit-tested there. This file's subject is the CLI WIRING in
``scripts/ledger_lock.py``, which is tracked HERE. The split is deliberate and
it is also why there is no inline fallback for the waiver: see
``test_the_waiver_is_refused_when_the_deciding_module_is_out_of_reach``.

The module-present legs resolve the real guard from an ``omni_home`` clone and
skip LOUDLY, naming the path they looked for, when there is not one on the
machine -- the same convention ``test_ledger_cost_sentence_omn18554.py`` and
``omni_home``'s own end-to-end classes use. The legs that matter most in a
runner with no such clone -- the argparse contract, and every refusal -- need
no module and run unconditionally.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "ledger_lock.py"

assert SCRIPT.is_file(), (
    f"scripts/ledger_lock.py must exist at {SCRIPT}. The script is COMMITTED in this "
    "repository (OMN-18554), so absence is a failure, never a skip."
)

GUARD_RELPATH = Path("docs") / "workflows" / "_shared" / "stranded_clone_guard.py"

#: A row recovered from the 2026-09-16 stranded tree. Two days old relative to
#: the window below, and carrying no ``friction=`` field, which is what the
#: OMN-18274 guard refused it for.
RECOVERED = (
    "2026-09-16T08:50:20Z | TERMINAL | lane=integration-plan | "
    "ticket=OMN-17195 | result=DONE"
)
BEFORE = "2026-09-16T10:41:37Z"


# --------------------------------------------------------------------------
# Resolving the committed deciding logic
# --------------------------------------------------------------------------


def _omni_home_guard() -> Path | None:
    """The real committed guard module, from an ``omni_home`` clone, or None.

    Never synthesizes a stand-in. A stub would let this file pass while the
    real module and the wiring disagreed, which is the failure mode the whole
    two-subject arrangement exists to prevent.
    """
    candidates: list[Path] = []
    env = os.environ.get("OMNI_HOME")
    if env:
        candidates.append(Path(env))
    # The canonical registry holds this clone as a sibling of omni_home's root.
    candidates.append(Path(__file__).resolve().parents[4])
    for root in candidates:
        guard = root / GUARD_RELPATH
        if guard.is_file():
            return guard
    return None


_GUARD = _omni_home_guard()
_NO_GUARD = (
    "the committed deciding logic is not on this machine: looked for "
    f"{GUARD_RELPATH} under $OMNI_HOME and under {Path(__file__).resolve().parents[4]}. "
    "The wiring legs that need no module run regardless; this leg needs the real "
    "module and will not substitute a stub for it."
)


@pytest.fixture
def omni_home(tmp_path: Path) -> Path:
    """A throwaway ``OMNI_HOME`` carrying the REAL guard module.

    ``ledger_lock.py`` resolves the module at ``$OMNI_HOME/docs/workflows/
    _shared/``, so pointing ``OMNI_HOME`` at a tmp tree is how a test decides
    whether the module is reachable for a given invocation.
    """
    if _GUARD is None:  # pragma: no cover - guarded by the skipif
        pytest.skip(_NO_GUARD)
    shared = tmp_path / GUARD_RELPATH.parent
    shared.mkdir(parents=True)
    (shared / GUARD_RELPATH.name).write_bytes(_GUARD.read_bytes())
    return tmp_path


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    path = tmp_path / "L.md"
    path.write_text("# L\n\n## section\n\n", encoding="utf-8")
    return path


def _run(
    ledger_path: Path, *args: str, omni_home_root: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Drive the real CLI.

    ``OMNI_HOME`` is set explicitly in every call, never inherited: an
    inherited one would point at the developer's real registry and make the
    module-absent legs pass or fail by accident of the machine.
    """
    env = dict(os.environ)
    env["OMNI_HOME"] = str(omni_home_root if omni_home_root else ledger_path.parent)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(ledger_path), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _append(
    ledger_path: Path, row: str, *extra: str, omni_home_root: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return _run(ledger_path, "--append", row, *extra, omni_home_root=omni_home_root)


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# The argparse contract. No module needed -- these are the anti-drop tests.
# --------------------------------------------------------------------------


class TestTheFlagsAreDeclared:
    """AC1/AC6. Removing either flag turns this class red in this repo's CI.

    The two-day outage this ticket closes was invisible here precisely
    because no test in this repository named either flag.
    """

    @pytest.mark.parametrize("flag", ["--replay-before", "--replay-source"])
    def test_the_flag_appears_in_help(self, ledger: Path, flag: str) -> None:
        helped = _run(ledger, "--help")
        assert helped.returncode == 0, helped.stderr
        assert flag in helped.stdout, (
            f"{flag} is not declared. OMN-18554's port of this file dropped both flags "
            "once already; this assertion is what makes a second drop loud."
        )

    def test_the_parser_accepts_the_pair(self, ledger: Path, tmp_path: Path) -> None:
        """A positive control on the parser itself.

        Without it, every refusal assertion below could be satisfied by
        argparse rejecting an unknown argument rather than by the refusal
        under test -- which is exactly how
        ``test_the_flag_requires_a_named_source`` passed vacuously in
        ``omni_home`` for two days.
        """
        proc = _append(
            ledger, RECOVERED, "--replay-before", BEFORE, "--replay-source", "s.txt"
        )
        assert "unrecognized arguments" not in proc.stderr, proc.stderr
        assert proc.returncode != 2, (proc.returncode, proc.stderr)


class TestPairingIsAUsageError:
    """AC3. Exit 2, and the message names the missing flag.

    These are usage errors rather than refusals because nothing about the
    payload or the ledger was consulted to reach them.
    """

    def test_a_window_with_no_source_names_the_source_flag(self, ledger: Path) -> None:
        proc = _append(ledger, RECOVERED, "--replay-before", BEFORE)
        assert proc.returncode == 2, (proc.returncode, proc.stderr)
        assert "--replay-source" in proc.stderr
        assert "unrecognized arguments" not in proc.stderr, (
            "this must be the pairing refusal, not argparse failing to recognize the "
            "flag -- the two are indistinguishable by exit code alone"
        )
        assert RECOVERED not in ledger.read_text(encoding="utf-8")

    def test_a_source_with_no_window_names_the_window_flag(self, ledger: Path) -> None:
        proc = _append(ledger, RECOVERED, "--replay-source", "s.txt")
        assert proc.returncode == 2, (proc.returncode, proc.stderr)
        assert "--replay-before" in proc.stderr
        assert RECOVERED not in ledger.read_text(encoding="utf-8")

    @pytest.mark.parametrize("empty", ["", "   "])
    def test_a_whitespace_only_source_is_no_source(
        self, ledger: Path, empty: str
    ) -> None:
        """Provenance that is present-but-blank is the bypass wearing a hat."""
        proc = _append(
            ledger, RECOVERED, "--replay-before", BEFORE, "--replay-source", empty
        )
        assert proc.returncode == 2, (proc.returncode, proc.stderr)
        assert RECOVERED not in ledger.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "malformed",
        ["2026-09-16", "yesterday", "2026-09-16T10:41:37", "2026-09-16T10:41:37+00:00"],
    )
    def test_a_malformed_window_stops_the_command(
        self, ledger: Path, malformed: str
    ) -> None:
        """A mistyped window must stop the command, never be coerced.

        Coercing it towards some nearby instant would silently widen the
        window the waiver is granted under.
        """
        proc = _append(
            ledger, RECOVERED, "--replay-before", malformed, "--replay-source", "s.txt"
        )
        assert proc.returncode == 2, (proc.returncode, proc.stderr)
        assert "--replay-before" in proc.stderr
        assert RECOVERED not in ledger.read_text(encoding="utf-8")


class TestTheWaiverFailsClosedWithoutItsDecidingLogic:
    """AC4, and the half of the design that is opposite to the stranded signal.

    ``signal_stranded_clone`` carries inline fallback constants so that a
    clone missing the module still SIGNALS -- there the danger is silence.
    Here the danger is a WAIVER, so the same absence must REFUSE. An inline
    copy of the rule would be a second implementation of it, free to drift
    towards permissive, which is the direction that costs something.
    """

    def test_the_waiver_is_refused_when_the_deciding_module_is_out_of_reach(
        self, ledger: Path, tmp_path: Path
    ) -> None:
        island = tmp_path / "no-guard-here"
        island.mkdir()
        proc = _append(
            ledger,
            RECOVERED,
            "--replay-before",
            BEFORE,
            "--replay-source",
            "s.txt",
            omni_home_root=island,
        )
        assert proc.returncode == 65, (proc.returncode, proc.stderr)
        assert "REFUSED" in proc.stderr
        assert RECOVERED not in ledger.read_text(encoding="utf-8")

    def test_the_refusal_names_the_module_it_could_not_read(
        self, ledger: Path, tmp_path: Path
    ) -> None:
        """A refusal an operator cannot act on gets routed around."""
        island = tmp_path / "no-guard-here"
        island.mkdir()
        proc = _append(
            ledger,
            RECOVERED,
            "--replay-before",
            BEFORE,
            "--replay-source",
            "s.txt",
            omni_home_root=island,
        )
        assert "stranded_clone_guard.py" in proc.stderr, proc.stderr

    def test_the_script_holds_no_inline_replay_fallback(self) -> None:
        """Static, and deliberately so.

        The behavioural test above proves today's build refuses. This one
        refuses the SHAPE that would make a future build stop refusing:
        a ``_FALLBACK_`` constant for the replay path, mirroring the two the
        stranded signal legitimately carries.
        """
        source = SCRIPT.read_text(encoding="utf-8")
        offenders = [
            line
            for line in source.splitlines()
            if line.startswith("_FALLBACK_") and "REPLAY" in line.upper()
        ]
        assert not offenders, (
            "the replay waiver must be granted by the committed module or not at all; "
            f"an inline fallback would be a second copy of the rule: {offenders}"
        )


# --------------------------------------------------------------------------
# Behaviour against the real committed deciding logic.
# --------------------------------------------------------------------------


@pytest.mark.skipif(_GUARD is None, reason=_NO_GUARD)
class TestTheRestoreIsVerbatim:
    def test_the_recovered_row_is_refused_without_the_flag(
        self, ledger: Path, omni_home: Path
    ) -> None:
        """The positive control for the entire feature.

        If this row would have landed anyway, every assertion below proves
        nothing: the flag would be waving through something no guard objected
        to. It is refused by the clock guard for being two days old.
        """
        proc = _append(ledger, RECOVERED, omni_home_root=omni_home)
        assert proc.returncode == 65, (proc.returncode, proc.stderr)
        assert RECOVERED not in ledger.read_text(encoding="utf-8")

    def test_with_the_flag_it_is_restored_byte_for_byte(
        self, ledger: Path, omni_home: Path
    ) -> None:
        proc = _append(
            ledger,
            RECOVERED,
            "--replay-before",
            BEFORE,
            "--replay-source",
            "recovered.txt",
            omni_home_root=omni_home,
        )
        assert proc.returncode == 0, proc.stderr
        assert RECOVERED + "\n" in ledger.read_text(encoding="utf-8")

    def test_the_marker_lands_immediately_before_the_restored_row(
        self, ledger: Path, omni_home: Path
    ) -> None:
        _append(
            ledger,
            RECOVERED,
            "--replay-before",
            BEFORE,
            "--replay-source",
            "recovered.txt",
            omni_home_root=omni_home,
        )
        lines = [
            ln for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        index = lines.index(RECOVERED)
        marker = lines[index - 1]
        assert "| REPLAY |" in marker
        assert "recovered.txt" in marker

    def test_the_marker_records_provenance_and_the_waiver(
        self, ledger: Path, omni_home: Path
    ) -> None:
        _append(
            ledger,
            RECOVERED,
            "--replay-before",
            BEFORE,
            "--replay-source",
            "recovered.txt",
            omni_home_root=omni_home,
        )
        marker = next(
            ln
            for ln in ledger.read_text(encoding="utf-8").splitlines()
            if "| REPLAY |" in ln
        )
        assert "original-timestamp=2026-09-16T08:50:20Z" in marker
        assert "restores-lane=integration-plan" in marker
        assert "guard=waived-on-replay" in marker

    def test_the_marker_makes_no_claim_about_a_guard_effective_date(
        self, ledger: Path, omni_home: Path
    ) -> None:
        """The recovery lane's "these rows predate the guard" reading was
        falsified in OMN-18433 -- the friction guard is blob-identical on both
        branches and landed two days earlier. The marker records where the
        bytes came from, never when a guard took effect."""
        _append(
            ledger,
            RECOVERED,
            "--replay-before",
            BEFORE,
            "--replay-source",
            "recovered.txt",
            omni_home_root=omni_home,
        )
        marker = next(
            ln
            for ln in ledger.read_text(encoding="utf-8").splitlines()
            if "| REPLAY |" in ln
        )
        assert "predate" not in marker.lower()

    def test_a_replayed_row_mints_no_claim_token(
        self, ledger: Path, omni_home: Path
    ) -> None:
        """A token minted now could be cited to authorize a mutation TODAY.

        Whatever the restored row's own claim authorized was settled at its
        own timestamp; re-minting it is the bypass this path must not open.
        """
        claim = (
            "- 2026-09-16T08:50:20Z [integration-plan] CLAIM OMN-17195 - restore; "
            "est ~1 lane-hours; displaces nothing; (OMN-17195)"
        )
        proc = _append(
            ledger,
            claim,
            "--replay-before",
            BEFORE,
            "--replay-source",
            "recovered.txt",
            omni_home_root=omni_home,
        )
        assert proc.returncode == 0, proc.stderr
        assert claim + "\n" in ledger.read_text(encoding="utf-8")
        assert "CLAIM-TOKEN" not in proc.stdout, proc.stdout


@pytest.mark.skipif(_GUARD is None, reason=_NO_GUARD)
class TestTheFlagIsNotAGeneralWaiver:
    """Every refusal writes nothing at all -- not the row, and not the marker.

    A refusal that left an orphan marker behind would assert in the ledger
    that a restore happened when none did.
    """

    def test_a_present_day_row_cannot_ride_the_flag(
        self, ledger: Path, omni_home: Path
    ) -> None:
        row = f"{_stamp(datetime.now(UTC))} | TERMINAL | lane=l | no friction field"
        proc = _append(
            ledger,
            row,
            "--replay-before",
            BEFORE,
            "--replay-source",
            "x.txt",
            omni_home_root=omni_home,
        )
        assert proc.returncode == 65, (proc.returncode, proc.stderr)
        text = ledger.read_text(encoding="utf-8")
        assert row not in text
        assert "| REPLAY |" not in text

    def test_a_row_stamped_exactly_at_the_window_is_refused(
        self, ledger: Path, omni_home: Path
    ) -> None:
        """The boundary, from the closed side. "Before" is strict."""
        row = f"{BEFORE} | TERMINAL | lane=l | result=DONE"
        proc = _append(
            ledger,
            row,
            "--replay-before",
            BEFORE,
            "--replay-source",
            "x.txt",
            omni_home_root=omni_home,
        )
        assert proc.returncode == 65, (proc.returncode, proc.stderr)
        assert row not in ledger.read_text(encoding="utf-8")

    def test_a_row_one_second_before_the_window_is_accepted(
        self, ledger: Path, omni_home: Path
    ) -> None:
        """The same boundary from the open side, so the test above is a
        boundary and not an accidental blanket refusal."""
        moment = datetime.strptime(BEFORE, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        row = f"{_stamp(moment - timedelta(seconds=1))} | TERMINAL | lane=l | done"
        proc = _append(
            ledger,
            row,
            "--replay-before",
            BEFORE,
            "--replay-source",
            "x.txt",
            omni_home_root=omni_home,
        )
        assert proc.returncode == 0, proc.stderr
        assert row + "\n" in ledger.read_text(encoding="utf-8")

    def test_a_payload_with_no_timestamp_is_refused(
        self, ledger: Path, omni_home: Path
    ) -> None:
        """With no timestamp there is no fact to compare against the window,
        and no way to tell a restore from a fresh write."""
        row = "TERMINAL | lane=l | no stamp at all"
        proc = _append(
            ledger,
            row,
            "--replay-before",
            BEFORE,
            "--replay-source",
            "x.txt",
            omni_home_root=omni_home,
        )
        assert proc.returncode == 65, (proc.returncode, proc.stderr)
        text = ledger.read_text(encoding="utf-8")
        assert row not in text
        assert "| REPLAY |" not in text

    def test_a_future_window_is_refused_as_a_blanket_waiver(
        self, ledger: Path, omni_home: Path
    ) -> None:
        """A window in the future admits every row ever written."""
        future = _stamp(datetime.now(UTC) + timedelta(days=365))
        proc = _append(
            ledger,
            RECOVERED,
            "--replay-before",
            future,
            "--replay-source",
            "x.txt",
            omni_home_root=omni_home,
        )
        assert proc.returncode == 65, (proc.returncode, proc.stderr)
        assert RECOVERED not in ledger.read_text(encoding="utf-8")


@pytest.mark.skipif(_GUARD is None, reason=_NO_GUARD)
class TestTheOrdinaryPathIsUnchanged:
    """The flags are additive. Every append that names neither behaves exactly
    as it did before OMN-18757, which is what makes this change safe to land
    under a fleet that is appending right now."""

    def test_an_ordinary_row_still_lands_and_exits_zero(
        self, ledger: Path, omni_home: Path
    ) -> None:
        row = f"{_stamp(datetime.now(UTC))} | NOTE | lane=omn18757-test | ordinary"
        proc = _append(ledger, row, omni_home_root=omni_home)
        assert proc.returncode == 0, proc.stderr
        assert row + "\n" in ledger.read_text(encoding="utf-8")
        assert "| REPLAY |" not in ledger.read_text(encoding="utf-8")

    def test_the_clock_guard_still_refuses_a_backdated_row(
        self, ledger: Path, omni_home: Path
    ) -> None:
        """The guard the replay path waives must still fire when it is not
        waived -- otherwise the waiver is indistinguishable from the guard
        having been removed."""
        proc = _append(ledger, RECOVERED, omni_home_root=omni_home)
        assert proc.returncode == 65, (proc.returncode, proc.stderr)
