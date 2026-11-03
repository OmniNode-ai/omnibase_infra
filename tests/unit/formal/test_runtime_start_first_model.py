# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""TLC evidence for the shared-listener start-first handover model (OMN-22376, T0).

Design revision 4 of OMN-22151 stage 2. Three base configurations, each with its
bounds in the cfg, hold the six properties between them: ``Http`` holds
AcceptedIsReady and NoRefusalWithoutFault, ``Bus`` holds AtMostOneConsuming,
NoSkip and ReplayOnlyX2 over two redeploys, ``Install`` holds WindowBounded (and
the three bus properties through rollback and roll-forward) with two agent faults.
Every property has a mutant configuration that must make TLC report a violation
of exactly that property; a mutant that stops failing fails this file. Witness
configurations show the paths the passing runs rely on (a redeploy, a hold, a
grant, an X2 replay, an install, a rollback, a roll-forward) are reachable.

Where ``TLA2TOOLS_JAR`` names tla2tools.jar and ``java`` is on PATH, each config
is run through TLC here and the assertions read that run. Everywhere else
(hosted CI has neither) they read the committed ``results/<cfg stem>.out``, which
the digest test ties to the spec and cfgs as they stand. To regenerate
``results/`` on a lab host with Java::

    TLA2TOOLS_JAR=/path/to/tla2tools.jar \\
        python tests/unit/formal/test_runtime_start_first_model.py --regenerate

It runs each cfg in formal/runtime_start_first into results/<cfg stem>.out and
writes the sha256 of the spec then the cfg files in sorted order to
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

MODEL_DIR = Path(__file__).resolve().parents[3] / "formal" / "runtime_start_first"
RESULTS = MODEL_DIR / "results"
SPEC = "RuntimeStartFirst.tla"

PASSING = ("Http", "Bus", "Install", "Install1")

#: property -> (base cfg that holds it, mutant cfg that must break exactly it).
PROPERTY_CHECKS = {
    "AcceptedIsReady": ("Http", "mut_teardown_first"),
    "NoRefusalWithoutFault": ("Http", "mut_shutdown_socket"),
    "AtMostOneConsuming": ("Bus", "mut_no_standby"),
    "NoSkip": ("Bus", "mut_autocommit"),
    "ReplayOnlyX2": ("Bus", "mut_seed_own_group"),
    "WindowBounded": ("Install", "mut_no_watchdog"),
}

#: mutant -> (base cfg, property). The first six are PROPERTY_CHECKS; the rest
#: are further breaks of rules the design names.
MUTANTS = {
    **{mutant: (base, prop) for prop, (base, mutant) in PROPERTY_CHECKS.items()},
    "mut_no_grants": ("Http", "NoRefusalWithoutFault"),
    "mut_no_verify_exit": ("Bus", "AtMostOneConsuming"),
    "mut_first_sight_latest": ("Install", "NoSkip"),
}

#: witness -> the invariant it violates: the path it names completes, so the
#: passing runs are not vacuous. ``wit_single_fault_bound`` is the design's
#: one-fault bound (WMax + RB + RF + W + G) against two agent faults: it is
#: breached, so the bound the model checks is the cumulative one.
WITNESSES = {
    "wit_redeployed": "NotRedeployed",
    "wit_hold": "NotHolding",
    "wit_grant": "NotGranted",
    "wit_two_redeploys": "NotRedeployed",
    "wit_x2_replay": "NotReplayed",
    "wit_installed": "NotInstalled",
    "wit_rolled_back": "NotRolledBack",
    "wit_rolled_forward": "NotRolledForward",
    "wit_failed": "NotFailed",
    "wit_two_agent_faults": "NotTwoAgentFaults",
    "wit_single_fault_bound": "WindowBounded",
}


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
@pytest.mark.parametrize(("prop", "check"), sorted(PROPERTY_CHECKS.items()))
def test_each_property_holds_in_its_base_and_fails_in_its_mutant(
    prop: str, check: tuple[str, str]
) -> None:
    base, mutant = check
    assert prop in _cfg_invariants(base)
    assert "No error has been found" in _output(base)
    out = _output(mutant)
    assert f"Error: Invariant {prop} is violated." in out
    assert "No error has been found" not in out


@pytest.mark.unit
@pytest.mark.parametrize(("mutant", "spec"), sorted(MUTANTS.items()))
def test_each_mutant_fails_exactly_its_property(
    mutant: str, spec: tuple[str, str]
) -> None:
    _base, prop = spec
    out = _output(mutant)
    assert f"Error: Invariant {prop} is violated." in out
    assert out.count("is violated.") == 1
    assert "No error has been found" not in out


@pytest.mark.unit
@pytest.mark.parametrize("mutant", sorted(MUTANTS))
def test_mutant_keeps_every_invariant_of_its_base_checked(mutant: str) -> None:
    base, _prop = MUTANTS[mutant]
    assert _cfg_invariants(mutant) == _cfg_invariants(base)


@pytest.mark.unit
@pytest.mark.parametrize(("witness", "inv"), sorted(WITNESSES.items()))
def test_each_witness_is_violated(witness: str, inv: str) -> None:
    out = _output(witness)
    assert f"Error: Invariant {inv} is violated." in out
    assert out.count("is violated.") == 1


@pytest.mark.unit
def test_every_cfg_has_a_recorded_result() -> None:
    cfgs = {p.stem for p in MODEL_DIR.glob("*.cfg")}
    assert cfgs == set(PASSING) | set(MUTANTS) | set(WITNESSES)
    for name in cfgs:
        assert (RESULTS / f"{name}.out").is_file()


@pytest.mark.unit
def test_the_six_properties_are_all_stated_by_a_base_config() -> None:
    stated = {inv for base in PASSING for inv in _cfg_invariants(base)} - {"TypeOK"}
    assert stated == set(PROPERTY_CHECKS)


@pytest.mark.unit
def test_stated_bounds_are_in_every_cfg() -> None:
    for cfg in MODEL_DIR.glob("*.cfg"):
        text = cfg.read_text()
        for key in ("MaxRedeploy", "LogLen", "MaxCrash", "MaxAgentCrash", "WBound"):
            assert re.search(rf"^\s+{key} = \d+", text, re.M), (cfg.name, key)


@pytest.mark.unit
def test_design_review_verdict_recorded() -> None:
    verdict = RESULTS / "design_review_verdict.txt"
    assert verdict.is_file()
    text = verdict.read_text()
    assert text.splitlines()[0].startswith("# Design-review verdict")
    assert "OMN-22376" in text
    assert "831a0f81" in text
    for residual in ("R1", "R2", "R3", "R4", "R5", "X1", "X2"):
        assert residual in text, residual


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
