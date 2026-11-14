# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""TLC evidence for the hourly check's send decisions and dedupe model (OMN-23732, AC1, AC2).

The design holds the no-double-send invariant, each mutant configuration fails it, the
witnesses are violated, and a design-review verdict is recorded, against the content digest of
the spec and cfg files.

Where ``TLA2TOOLS_JAR`` names tla2tools.jar and ``java`` is on PATH, each config is run through
TLC here and the assertions read that run. Everywhere else (hosted CI has neither) they read the
committed ``results/<cfg stem>.out``, which the digest test ties to the spec and cfgs as they
stand. To regenerate ``results/`` on a lab host with Java::

    TLA2TOOLS_JAR=/path/to/tla2tools.jar \\
        python tests/formal/test_hourly_send_model.py --regenerate

It runs each cfg in formal/hourly_send into results/<cfg stem>.out (the last line is
``exit=<TLC exit code>``) and writes the sha256 of the spec then the cfg files in sorted order to
results/model.sha256.
"""

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

MODEL_DIR = Path(__file__).resolve().parents[2] / "formal" / "hourly_send"
RESULTS = MODEL_DIR / "results"
SPEC = "HourlySend.tla"

PASSING = ("Model", "once")
EXPECTED_VIOLATIONS = {
    "mut_as_built": "Invariant NoKeyDoubleSend is violated.",
    "mut_nonatomic_claim": "Invariant NoKeyDoubleSend is violated.",
    "mut_record_after_send": "Invariant NoKeyDoubleSend is violated.",
    "mut_unfenced_send": "Invariant NoKeyDoubleSend is violated.",
    "mut_unguarded_rollback": "Invariant NoKeyDoubleSend is violated.",
    "mut_w2_replace": "Invariant NoKeyDoubleSend is violated.",
    "mut_unreadable_empty": "Invariant NoKeyDoubleSend is violated.",
    "wit_sent": "Invariant NothingSent is violated.",
    "wit_resend": "Invariant NoResend is violated.",
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
def test_each_mutant_fails_its_invariant(name: str, message: str) -> None:
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
def test_design_states_the_invariant_and_the_literal_form() -> None:
    assert set(_cfg_invariants("Model")) - {"TypeOK"} == {"NoKeyDoubleSend"}
    assert set(_cfg_invariants("once")) - {"TypeOK"} == {"NoKeySentTwice"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutant", sorted(n for n in EXPECTED_VIOLATIONS if n.startswith("mut_"))
)
def test_mutant_keeps_every_invariant_checked(mutant: str) -> None:
    assert _cfg_invariants(mutant) == _cfg_invariants("Model")


@pytest.mark.unit
def test_design_review_verdict_recorded() -> None:
    verdict = RESULTS / "design_review_verdict.txt"
    assert verdict.is_file()
    text = verdict.read_text()
    assert text.splitlines()[0].startswith("# Design-review verdict")
    assert "OMN-23732" in text


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
