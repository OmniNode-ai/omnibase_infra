# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""TLC evidence for the start-first runtime redeploy and first-install model (OMN-22151).

The redeploy and install configurations pass every invariant, each mutant
configuration fails the one property it is named for, each witness is violated
(the path it names completes), and the verdict is recorded, against the content
digest of the spec and cfg files.

Where ``TLA2TOOLS_JAR`` names tla2tools.jar and ``java`` is on PATH, each config
is run through TLC here and the assertions read that run. Everywhere else
(hosted CI has neither) they read the committed ``results/<cfg stem>.out``, which
the digest test ties to the spec and cfgs as they stand. To regenerate
``results/`` on a lab host with Java::

    TLA2TOOLS_JAR=/path/to/tla2tools.jar \\
        python tests/formal/test_runtime_start_first_model.py --regenerate

It runs each cfg in formal/runtime_start_first into results/<cfg stem>.out
and writes the sha256 of the spec then the cfg files in sorted order to
results/model.sha256.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parents[2] / "formal" / "runtime_start_first"
RESULTS = MODEL_DIR / "results"
SPEC = "RuntimeStartFirst.tla"

PASSING = ("Redeploy", "Install")
EXPECTED_VIOLATIONS = {
    "mut_no_fence": "Invariant AtMostOneServing is violated.",
    "mut_rollback_no_fence": "Invariant AtMostOneServing is violated.",
    "mut_stand_unpaused": "Invariant AtMostOneServing is violated.",
    "mut_install_resume_early": "Invariant AtMostOneServing is violated.",
    "mut_fwd_no_stop": "Invariant AtMostOneServing is violated.",
    "mut_no_epoch": "Invariant AtMostOneServing is violated.",
    "mut_no_seek": "Invariant Contiguous is violated.",
    "mut_abort_no_seek": "Invariant NoDoubleEffect is violated.",
    "mut_rollback_no_copy": "Invariant NoDoubleEffect is violated.",
    "mut_resume_replay": "Invariant NoDoubleEffect is violated.",
    "mut_src_from_events": "Invariant NoDoubleEffect is violated.",
    "mut_no_retry": "Invariant RefusalHasCause is violated.",
    "mut_stop_early": "Invariant RefusalHasCause is violated.",
    "mut_fence_unready": "Invariant RefusalHasCause is violated.",
    "mut_stop_unverified": "Invariant StopOnlyWhenLit is violated.",
    "mut_abort_stops_old": "Invariant FailedLeavesOldServing is violated.",
    "mut_abort_after_stop": "Invariant FailedLeavesOldServing is violated.",
    "mut_open_early": "Invariant WindowStartsWithNewStandby is violated.",
    "mut_no_bound": "Invariant WindowBounded is violated.",
    "mut_rollback_keeps_front": "Invariant RolledBackIsOld is violated.",
    "mut_stop_before_record": "Invariant WindowRecordedBeforeStop is violated.",
    "mut_front_one_port": "Invariant FrontOnFreePorts is violated.",
    "mut_rollback_no_restart": "Invariant InstallFailedNeedsTwoFaults is violated.",
    "mut_no_fwd_reserve": "Invariant RollForwardHasItsTime is violated.",
    "mut_no_rollback": "Deadlock reached.",
    "wit_redeploy_done": "Invariant RedeployDoneWitness is violated.",
    "wit_redeploy_failed": "Invariant RedeployFailedWitness is violated.",
    "wit_install_done": "Invariant InstallDoneWitness is violated.",
    "wit_install_rolled_back": "Invariant InstallRolledBackWitness is violated.",
    "wit_install_forward_done": "Invariant InstallForwardDoneWitness is violated.",
    "wit_install_failed": "Invariant InstallFailedWitness is violated.",
    "wit_overlap": "Invariant OverlapWitness is violated.",
}

#: Each invariant the cfgs state, with a mutant that must break it. DoneIsNew is
#: a postcondition the agent's Complete step establishes; it has no mutant. The
#: temporal properties Resolves and WindowResolves are checked under FairSpec in
#: the two passing configs and have no mutant.
INVARIANT_MUTANTS = {
    "AtMostOneServing": "mut_no_fence",
    "Contiguous": "mut_no_seek",
    "NoDoubleEffect": "mut_abort_no_seek",
    "RefusalHasCause": "mut_no_retry",
    "StopOnlyWhenLit": "mut_stop_unverified",
    "FailedLeavesOldServing": "mut_abort_stops_old",
    "WindowStartsWithNewStandby": "mut_open_early",
    "WindowBounded": "mut_no_bound",
    "RolledBackIsOld": "mut_rollback_keeps_front",
    "WindowRecordedBeforeStop": "mut_stop_before_record",
    "FrontOnFreePorts": "mut_front_one_port",
    "InstallFailedNeedsTwoFaults": "mut_rollback_no_restart",
    "RollForwardHasItsTime": "mut_no_fwd_reserve",
}

#: The line the design quotes; the verdict file must carry it with the checked bounds.
VERDICT_PREFIX = "VERDICT: SAFE under the stated bounds"


def _digest() -> str:
    h = hashlib.sha256()
    h.update((MODEL_DIR / SPEC).read_bytes())
    for cfg in sorted(MODEL_DIR.glob("*.cfg")):
        h.update(cfg.read_bytes())
    return h.hexdigest()


def _jar() -> str | None:
    jar = os.environ.get("TLA2TOOLS_JAR")
    if jar and Path(jar).is_file() and shutil.which("java"):
        return jar
    return None


def _run_tlc(jar: str, cfg_stem: str) -> str:
    """Run TLC on a copy of the model, so its trace files never land in the tree."""
    with tempfile.TemporaryDirectory() as work:
        for src in [MODEL_DIR / SPEC, *MODEL_DIR.glob("*.cfg")]:
            shutil.copy(src, work)
        proc = subprocess.run(
            [
                "java",
                "-XX:+UseParallelGC",
                "-cp",
                jar,
                "tlc2.TLC",
                "-noGenerateSpecTE",
                "-config",
                f"{cfg_stem}.cfg",
                "-workers",
                "4",
                "-metadir",
                str(Path(work) / "states"),
                SPEC,
            ],
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
            timeout=3000,
        )
    return proc.stdout + proc.stderr


@functools.cache
def _output(cfg_stem: str) -> str:
    """TLC's output for one cfg: a live run where Java is present, else the record."""
    jar = _jar()
    if jar is not None:
        return _run_tlc(jar, cfg_stem)
    return (RESULTS / f"{cfg_stem}.out").read_text()


def _cfg_invariants(cfg_stem: str) -> list[str]:
    text = (MODEL_DIR / f"{cfg_stem}.cfg").read_text()
    block = re.search(r"^INVARIANTS[ \t]*\n((?:[ \t]+\w+[ \t]*(?:\n|\Z))+)", text, re.M)
    assert block is not None, f"{cfg_stem}.cfg lists no INVARIANTS"
    return block.group(1).split()


@pytest.mark.unit
def test_results_bind_to_current_model_digest() -> None:
    assert (RESULTS / "model.sha256").read_text().strip() == _digest()


@pytest.mark.unit
@pytest.mark.parametrize("name", PASSING)
def test_passing_configs_hold_every_invariant(name: str) -> None:
    out = _output(name)
    assert "Model checking completed. No error has been found." in out
    assert "0 states left on queue." in out


@pytest.mark.unit
@pytest.mark.parametrize(("name", "message"), sorted(EXPECTED_VIOLATIONS.items()))
def test_each_mutant_fails_its_property(name: str, message: str) -> None:
    out = _output(name)
    assert f"Error: {message}" in out
    assert "No error has been found" not in out


@pytest.mark.unit
def test_every_cfg_has_a_recorded_result() -> None:
    cfgs = {p.stem for p in MODEL_DIR.glob("*.cfg")}
    assert cfgs == set(PASSING) | set(EXPECTED_VIOLATIONS)
    for name in cfgs:
        assert (RESULTS / f"{name}.out").is_file()


@pytest.mark.unit
def test_model_states_every_invariant_with_a_mutant_that_breaks_it() -> None:
    stated = set(_cfg_invariants("Redeploy")) - {"TypeOK", "DoneIsNew"}
    assert stated == set(INVARIANT_MUTANTS)
    for invariant, mutant in INVARIANT_MUTANTS.items():
        assert EXPECTED_VIOLATIONS[mutant] == f"Invariant {invariant} is violated."
    assert _cfg_invariants("Install") == _cfg_invariants("Redeploy")


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutant", sorted(n for n in EXPECTED_VIOLATIONS if n.startswith("mut_"))
)
def test_mutant_keeps_every_invariant_checked(mutant: str) -> None:
    assert _cfg_invariants(mutant) == _cfg_invariants("Redeploy")


@pytest.mark.unit
def test_verdict_recorded() -> None:
    verdict = RESULTS / "verdict.txt"
    assert verdict.is_file()
    lines = verdict.read_text().splitlines()
    assert lines[0].startswith(VERDICT_PREFIX)
    assert "OMN-22151" in verdict.read_text()


@pytest.mark.unit
def test_every_mutant_is_stopped_on_its_first_violation() -> None:
    """A counterexample names one property; no mutant reports two."""
    for name in EXPECTED_VIOLATIONS:
        out = _output(name)
        assert out.count("is violated.") + out.count("Deadlock reached.") == 1


def _regenerate() -> int:
    jar = os.environ.get("TLA2TOOLS_JAR")
    if not jar or not shutil.which("java"):
        print(
            "set TLA2TOOLS_JAR to tla2tools.jar and put java on PATH", file=sys.stderr
        )
        return 2
    RESULTS.mkdir(exist_ok=True)
    for cfg in sorted(MODEL_DIR.glob("*.cfg")):
        out = _run_tlc(jar, cfg.stem)
        (RESULTS / f"{cfg.stem}.out").write_text(out)
        print(f"{cfg.name}: {out.strip().splitlines()[-1] if out.strip() else ''}")
    (RESULTS / "model.sha256").write_text(_digest() + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(_regenerate() if "--regenerate" in sys.argv else 2)
