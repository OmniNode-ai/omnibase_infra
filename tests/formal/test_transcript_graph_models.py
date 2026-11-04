# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""TLC evidence for the four transcript-graph models (OMN-22258, task T0).

The models are formal/transcript_ingest_reducer (ingest reducer, terminal outbox
and the downstream source-version update), formal/mirror_conditional_upload,
formal/fallback_store_once and formal/finding_publication. For each, the passing
configs hold every invariant and property, each mutant config fails the one
property it is named for, each witness is violated (its path is reachable), and
the committed results are bound to the sha256 of the spec and cfg files.

Where ``TLA2TOOLS_JAR`` names tla2tools.jar and ``java`` is on PATH, each config
is run through TLC here and the assertions read that run. Everywhere else
(hosted CI has neither) they read the committed ``results/<cfg stem>.out``, which
the digest test ties to the spec and cfgs as they stand. To regenerate
``results/`` on a lab host with Java::

    TLA2TOOLS_JAR=/path/to/tla2tools.jar \\
        python tests/formal/test_transcript_graph_models.py --regenerate

It runs each cfg of each model into results/<cfg stem>.out and writes the sha256
of the spec then the cfg files in sorted order to results/model.sha256.
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
from dataclasses import dataclass
from pathlib import Path

import pytest

FORMAL = Path(__file__).resolve().parents[2] / "formal"


def _inv(name: str) -> str:
    return f"Error: Invariant {name} is violated."


def _temporal(*names: str) -> str:
    if len(names) == 1:
        return f"Error: Temporal property {names[0]} was violated."
    return f"Error: Temporal properties {' and '.join(names)} were violated."


@dataclass(frozen=True)
class Model:
    directory: str
    spec: str
    passing: tuple[str, ...]
    #: cfg stem -> the first line TLC prints for it
    violations: dict[str, str]
    #: property of Model.cfg -> the cfg stems that must break it
    breakers: dict[str, tuple[str, ...]]

    @property
    def path(self) -> Path:
        return FORMAL / self.directory

    @property
    def results(self) -> Path:
        return self.path / "results"


INGEST = Model(
    directory="transcript_ingest_reducer",
    spec="TranscriptIngestReducer",
    passing=("Model", "NoConflict"),
    violations={
        "mut_accept_any_decreases": _inv("RowNeverDecreases"),
        "mut_accept_any_highest": _inv("RowIsHighestApplied"),
        "mut_term_on_publish": _inv("TerminalOnlyWhenSatisfied"),
        "mut_term_on_publish_emitted": _inv("EmittedOnlyWhenSatisfied"),
        "mut_no_digest": _inv("TerminalOnlyWhenSatisfied"),
        "mut_two_step": _inv("CloseCarriesOutbox"),
        "mut_two_step_lost": _temporal("ClosedEventuallyEffected"),
        "mut_mark_before_publish": _inv("SentWasPublished"),
        "mut_no_dedup": _inv("EffectAtMostOnce"),
        "mut_apply_by_arrival": _inv("DerivedNeverOlder"),
        "mut_apply_by_arrival_newest": _inv("DerivedIsNewest"),
        "mut_no_fairness": _temporal("EventuallyApplied"),
        "wit_both_effected": _inv("BothEffectedWitness"),
        "wit_conflict": _inv("ConflictWitness"),
        "wit_duplicate_terminal": _inv("DuplicateWitness"),
        "wit_ignored_older": _inv("IgnoredOlderWitness"),
        "wit_redelivery": _inv("RedeliveryWitness"),
    },
    breakers={
        "RowNeverDecreases": ("mut_accept_any_decreases",),
        "RowIsHighestApplied": ("mut_accept_any_highest",),
        "TerminalOnlyWhenSatisfied": ("mut_term_on_publish", "mut_no_digest"),
        "EmittedOnlyWhenSatisfied": ("mut_term_on_publish_emitted",),
        "CloseCarriesOutbox": ("mut_two_step",),
        "SentWasPublished": ("mut_mark_before_publish",),
        "EffectAtMostOnce": ("mut_no_dedup",),
        "DerivedNeverOlder": ("mut_apply_by_arrival",),
        "DerivedIsNewest": ("mut_apply_by_arrival_newest",),
        "EventuallyApplied": ("mut_no_fairness",),
        "ClosedEventuallyEffected": ("mut_two_step_lost",),
    },
)

MIRROR = Model(
    directory="mirror_conditional_upload",
    spec="MirrorConditionalUpload",
    passing=("Model",),
    violations={
        "mut_put_unconditional": _inv("RecordedNeverDecreases"),
        "mut_check_then_put": _inv("RecordedNeverDecreases"),
        "mut_put_unconditional_longest": _inv("RecordedIsLongestPut"),
        "mut_no_fairness": _temporal("EverySnapshotCovered"),
        "wit_refused": _inv("RefusedWitness"),
        "wit_full": _inv("FullWitness"),
    },
    breakers={
        "RecordedNeverDecreases": ("mut_put_unconditional", "mut_check_then_put"),
        "RecordedIsLongestPut": ("mut_put_unconditional_longest",),
        "EverySnapshotCovered": ("mut_no_fairness",),
    },
)

FALLBACK = Model(
    directory="fallback_store_once",
    spec="FallbackStoreOnce",
    passing=("Model",),
    violations={
        "mut_write_on_arrival": _inv("OneStoredResult"),
        "mut_lookup_then_put": _inv("OneStoredResult"),
        "mut_lookup_then_put_writes": _inv("OneWrite"),
        "mut_write_on_arrival_answered": _inv("DifferentRefused"),
        "mut_no_fairness": _temporal("EveryCallAnswered", "SomeResultStored"),
        "wit_refused": _inv("RefusedWitness"),
        "wit_same": _inv("SameWitness"),
    },
    breakers={
        "OneStoredResult": ("mut_write_on_arrival", "mut_lookup_then_put"),
        "OneWrite": ("mut_lookup_then_put_writes",),
        "DifferentRefused": ("mut_write_on_arrival_answered",),
        "EveryCallAnswered": ("mut_no_fairness",),
        "SomeResultStored": ("mut_no_fairness",),
    },
)

FINDING = Model(
    directory="finding_publication",
    spec="FindingPublication",
    passing=("Model",),
    violations={
        "mut_lookup_then_create": _inv("AtMostOnePr"),
        "mut_branch_is_done": _inv("BranchGetsPr"),
        "mut_branch_is_done_liveness": _temporal("BranchGetsPrEventually"),
        "mut_stage_by_arrival": _inv("RecordedNeverBack"),
        "mut_no_fairness": _temporal("BranchGetsPrEventually"),
        "mut_no_fairness_recorded": _temporal("PrGetsRecorded"),
        "wit_adopt_branch": _inv("AdoptBranchWitness"),
        "wit_adopt_pr": _inv("AdoptPrWitness"),
        "wit_reorder": _inv("ReorderWitness"),
    },
    breakers={
        "AtMostOnePr": ("mut_lookup_then_create",),
        "BranchGetsPr": ("mut_branch_is_done",),
        "RecordedNeverBack": ("mut_stage_by_arrival",),
        "BranchGetsPrEventually": ("mut_branch_is_done_liveness", "mut_no_fairness"),
        "PrGetsRecorded": ("mut_no_fairness_recorded",),
    },
)

MODELS = (INGEST, MIRROR, FALLBACK, FINDING)

#: Properties that a model's Model.cfg states and that no mutant can be asked
#: to break on its own, with the reason.
UNBROKEN = {
    # RecordedBackedByForge holds in every mutant too: it states what the passing
    # model's recorded stage is backed by, and the stage events are only ever
    # emitted after the forge call they name.
    "finding_publication": {"RecordedBackedByForge"},
}


def _digest(model: Model) -> str:
    h = hashlib.sha256()
    h.update((model.path / f"{model.spec}.tla").read_bytes())
    for cfg in sorted(model.path.glob("*.cfg")):
        h.update(cfg.read_bytes())
    return h.hexdigest()


def _jar() -> str | None:
    jar = os.environ.get("TLA2TOOLS_JAR")
    if jar and Path(jar).is_file() and shutil.which("java"):
        return jar
    return None


def _run_tlc(jar: str, model: Model, cfg_stem: str) -> str:
    """Run TLC on a copy of the model, so its trace files never land in the tree."""
    with tempfile.TemporaryDirectory() as work:
        for src in [model.path / f"{model.spec}.tla", *model.path.glob("*.cfg")]:
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
                f"{model.spec}.tla",
            ],
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
            timeout=3000,
        )
    return proc.stdout + proc.stderr


@functools.cache
def _output(directory: str, cfg_stem: str) -> str:
    """TLC's output for one cfg: a live run where Java is present, else the record."""
    model = next(m for m in MODELS if m.directory == directory)
    jar = _jar()
    if jar is not None:
        return _run_tlc(jar, model, cfg_stem)
    return (model.results / f"{cfg_stem}.out").read_text()


def _cfg_section(model: Model, cfg_stem: str, section: str) -> list[str]:
    text = (model.path / f"{cfg_stem}.cfg").read_text()
    block = re.search(rf"^{section}[ \t]*\n((?:[ \t]+\w+[ \t]*(?:\n|\Z))+)", text, re.M)
    return block.group(1).split() if block else []


_PASSING = [(m, name) for m in MODELS for name in m.passing]
_VIOLATING = [(m, n, msg) for m in MODELS for n, msg in sorted(m.violations.items())]
_IDS = [f"{m.directory}:{n}" for m, n, _ in _VIOLATING]


@pytest.mark.unit
@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.directory)
def test_results_bind_to_current_model_digest(model: Model) -> None:
    assert (model.results / "model.sha256").read_text().strip() == _digest(model)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "name"), _PASSING, ids=lambda x: getattr(x, "directory", x)
)
def test_passing_configs_hold_every_property(model: Model, name: str) -> None:
    out = _output(model.directory, name)
    assert "Model checking completed. No error has been found." in out
    assert "0 states left on queue." in out


@pytest.mark.unit
@pytest.mark.parametrize(("model", "name", "message"), _VIOLATING, ids=_IDS)
def test_each_control_and_witness_fails_as_named(
    model: Model, name: str, message: str
) -> None:
    out = _output(model.directory, name)
    assert message in out
    assert "No error has been found" not in out


@pytest.mark.unit
@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.directory)
def test_every_cfg_has_a_recorded_result(model: Model) -> None:
    cfgs = {p.stem for p in model.path.glob("*.cfg")}
    assert cfgs == set(model.passing) | set(model.violations)
    for name in cfgs:
        assert (model.results / f"{name}.out").is_file()


@pytest.mark.unit
@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.directory)
def test_model_states_every_property_with_a_control_that_breaks_it(
    model: Model,
) -> None:
    stated = set(_cfg_section(model, "Model", "INVARIANTS")) - {"TypeOK"}
    stated |= set(_cfg_section(model, "Model", "PROPERTIES"))
    unbroken = UNBROKEN.get(model.directory, set())
    assert stated - unbroken == set(model.breakers)
    for prop, controls in model.breakers.items():
        for control in controls:
            checked = set(_cfg_section(model, control, "INVARIANTS")) | set(
                _cfg_section(model, control, "PROPERTIES")
            )
            assert prop in checked, f"{control}.cfg does not check {prop}"


@pytest.mark.unit
@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.directory)
def test_witnesses_check_only_their_witness(model: Model) -> None:
    for name in model.violations:
        if name.startswith("wit_"):
            invariants = _cfg_section(model, name, "INVARIANTS")
            assert len(invariants) == 1
            assert invariants[0].endswith("Witness")
            assert not _cfg_section(model, name, "PROPERTIES")


@pytest.mark.unit
@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.directory)
def test_mutants_use_the_model_constants_except_the_mutation(model: Model) -> None:
    base = (model.path / "Model.cfg").read_text()
    for name in model.violations:
        if not name.startswith("mut_"):
            continue
        cfg = (model.path / f"{name}.cfg").read_text()
        for line in base.splitlines():
            key = line.split("=")[0].strip()
            if "=" in line and key not in {"MUT", "ConflictDig"}:
                assert line in cfg, f"{name}.cfg differs from Model.cfg on {key}"


@pytest.mark.unit
@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.directory)
def test_verdict_recorded_and_cites_the_model(model: Model) -> None:
    verdict = model.results / "design_review_verdict.txt"
    text = verdict.read_text()
    first = text.splitlines()[0]
    assert "OMN-22258" in first
    assert re.search(
        r"^Verdict: \*\*(PASS|PASS with conditions|FAIL)\.\*\*", text, re.M
    )
    assert f"formal/{model.directory}/" in text
    assert "## Findings" in text
    assert "## Conditions" in text
    assert "Codex" in text


def _regenerate() -> int:
    jar = os.environ.get("TLA2TOOLS_JAR")
    if not jar or not shutil.which("java"):
        print(
            "set TLA2TOOLS_JAR to tla2tools.jar and put java on PATH", file=sys.stderr
        )
        return 2
    for model in MODELS:
        model.results.mkdir(exist_ok=True)
        cfgs = sorted(
            model.path.glob("*.cfg"),
            key=lambda p: (not p.stem.startswith(("mut_", "wit_")), p.name),
        )
        for cfg in cfgs:
            out = _run_tlc(jar, model, cfg.stem)
            (model.results / f"{cfg.stem}.out").write_text(out)
            last = out.strip().splitlines()[-1] if out.strip() else ""
            print(f"{model.directory}/{cfg.name}: {last}")
        (model.results / "model.sha256").write_text(_digest(model) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(_regenerate() if "--regenerate" in sys.argv else 2)
