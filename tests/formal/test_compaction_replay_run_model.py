# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""TLC evidence for the compaction replay-run model (OMN-21090, plan task CM.0).

The model passes every invariant, each mutant configuration fails the invariant
it is named for with every invariant still enabled, and a design-review verdict
is recorded, against the content digest of the spec and cfg files. The
assertions read the committed ``results/<cfg stem>.out``; the digest test ties
those files to the spec and cfgs as they stand, so a changed model with stale
results fails here. To regenerate ``results/`` on a lab host with java or
docker (a shell script has no canonical location in this repository, so the
runner lives here, as it does for the host bring-up model)::

    TLA2TOOLS_JAR=/path/to/tla2tools.jar \\
        uv run python tests/formal/test_compaction_replay_run_model.py --regenerate

It runs Model.cfg (which must pass) and each mut_*.cfg (each must violate the
invariant it is named for) into results/<cfg stem>.out, writes the sha256 of the
spec then the cfg files in sorted order to results/model.sha256, prints one
verdict line per config, and exits 1 when any config does not behave as expected.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parents[2] / "formal" / "compaction_replay_run"
RESULTS = MODEL_DIR / "results"
SPEC = "CompactionReplayRun.tla"

EXPECTED_VIOLATIONS = {
    "mut_terminal_unique": "Invariant UniqueTerminal is violated.",
    "mut_killswitch": "Invariant NoOnAfterTrip is violated.",
    "mut_boundary_read": "Invariant NoReadAfterBoundary is violated.",
    "mut_infra_removal": "Invariant MatchedRemoval is violated.",
    "mut_fork_restore": "Invariant ForkRestore is violated.",
    "mut_terminal_kind": "Invariant KindMatchesCause is violated.",
    "mut_prefix_once": "Invariant OnePrefixOneSnapshot is violated.",
}

#: The invariants the plan's CM.0 names, each with the mutant that must break it.
INVARIANT_MUTANTS = {
    "UniqueTerminal": "mut_terminal_unique",
    "NoReadAfterBoundary": "mut_boundary_read",
    "MatchedRemoval": "mut_infra_removal",
    "ForkRestore": "mut_fork_restore",
    "NoOnAfterTrip": "mut_killswitch",
    "KindMatchesCause": "mut_terminal_kind",
    "OnePrefixOneSnapshot": "mut_prefix_once",
}


def _digest() -> str:
    h = hashlib.sha256()
    h.update((MODEL_DIR / SPEC).read_bytes())
    for cfg in sorted(MODEL_DIR.glob("*.cfg")):
        h.update(cfg.read_bytes())
    return h.hexdigest()


def _cfg_invariants(cfg_stem: str) -> list[str]:
    text = (MODEL_DIR / f"{cfg_stem}.cfg").read_text()
    match = re.search(r"^INVARIANTS[ \t]+(.+)$", text, re.M)
    assert match is not None, f"{cfg_stem}.cfg lists no INVARIANTS"
    return match.group(1).split()


@pytest.mark.unit
def test_results_bind_to_current_model_digest() -> None:
    assert (RESULTS / "model.sha256").read_text().strip() == _digest()


@pytest.mark.unit
def test_model_cfg_holds_every_invariant_and_terminates() -> None:
    out = (RESULTS / "Model.out").read_text()
    assert "Model checking completed. No error has been found." in out
    assert "0 states left on queue." in out
    assert "PROPERTIES Terminates" in (MODEL_DIR / "Model.cfg").read_text()


@pytest.mark.unit
@pytest.mark.parametrize(("name", "message"), sorted(EXPECTED_VIOLATIONS.items()))
def test_each_mutant_fails_its_invariant(name: str, message: str) -> None:
    out = (RESULTS / f"{name}.out").read_text()
    assert f"Error: {message}" in out
    assert "No error has been found" not in out


@pytest.mark.unit
def test_every_cfg_has_a_recorded_result() -> None:
    cfgs = {p.stem for p in MODEL_DIR.glob("*.cfg")}
    assert cfgs == {"Model", *EXPECTED_VIOLATIONS}
    for name in cfgs:
        assert (RESULTS / f"{name}.out").is_file()


@pytest.mark.unit
def test_model_states_every_invariant_with_a_mutant_that_breaks_it() -> None:
    stated = set(_cfg_invariants("Model")) - {"TypeOK"}
    assert stated == set(INVARIANT_MUTANTS)
    for invariant, mutant in INVARIANT_MUTANTS.items():
        assert EXPECTED_VIOLATIONS[mutant] == f"Invariant {invariant} is violated."


@pytest.mark.unit
@pytest.mark.parametrize("mutant", sorted(EXPECTED_VIOLATIONS))
def test_mutant_keeps_every_invariant_checked(mutant: str) -> None:
    assert _cfg_invariants(mutant) == _cfg_invariants("Model")


@pytest.mark.unit
def test_design_review_verdict_recorded() -> None:
    verdict = RESULTS / "design_review_verdict.txt"
    assert verdict.is_file()
    text = verdict.read_text()
    assert text.splitlines()[0].startswith("# Design-review verdict")
    assert "OMN-21090" in text


@pytest.mark.unit
def test_design_review_verdict_names_the_gated_tasks() -> None:
    text = (RESULTS / "design_review_verdict.txt").read_text()
    assert "CM.10" in text
    assert "CM.16" in text


def _run_tlc(jar: str, cfg_stem: str) -> str:
    """Run TLC on a copy of the model, so its trace files never land in the tree."""
    with tempfile.TemporaryDirectory() as work:
        for src in [MODEL_DIR / SPEC, *MODEL_DIR.glob("*.cfg")]:
            shutil.copy(src, work)
        tlc = ["tlc2.TLC", "-config", f"{cfg_stem}.cfg"]
        tlc += ["-workers", "2", "-metadir", str(Path(work) / "states"), SPEC]
        if shutil.which("java"):
            cmd = ["java", "-XX:+UseParallelGC", "-cp", jar, *tlc]
        else:
            user = f"{os.getuid()}:{os.getgid()}"
            cmd = ["docker", "run", "--rm", "--user", user, "-w", work]
            cmd += ["-v", f"{work}:{work}", "-v", f"{jar}:/tla2tools.jar:ro"]
            cmd += [
                "eclipse-temurin:17-jre-alpine",
                "java",
                "-cp",
                "/tla2tools.jar",
                *tlc,
            ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=3000
        )
    return proc.stdout + proc.stderr


def _regenerate() -> int:
    jar = os.environ.get("TLA2TOOLS_JAR")
    if not jar or not Path(jar).is_file():
        print("set TLA2TOOLS_JAR to tla2tools.jar", file=sys.stderr)
        return 2
    if not (shutil.which("java") or shutil.which("docker")):
        print("java or docker is required", file=sys.stderr)
        return 2
    RESULTS.mkdir(exist_ok=True)
    status = 0
    for cfg in sorted(MODEL_DIR.glob("*.cfg")):
        out = _run_tlc(jar, cfg.stem)
        (RESULTS / f"{cfg.stem}.out").write_text(out)
        if cfg.stem == "Model":
            ok = "Model checking completed. No error has been found." in out
        else:
            ok = f"Error: {EXPECTED_VIOLATIONS[cfg.stem]}" in out
        status |= 0 if ok else 1
        print(f"{cfg.name}: {'ok' if ok else 'UNEXPECTED'}")
    (RESULTS / "model.sha256").write_text(_digest() + "\n")
    return status


if __name__ == "__main__":
    sys.exit(_regenerate() if "--regenerate" in sys.argv else 2)
