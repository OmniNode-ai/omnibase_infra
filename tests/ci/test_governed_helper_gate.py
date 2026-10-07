# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18629 — the governed-helper primitive gate refuses a NEW bare primitive.

A repository that already ships a governed helper keeps acquiring fresh call
sites of the bare primitive that helper supersedes. Four of the twelve verified
defects of the 2026-09-17 evening window were exactly that, and three of the
four are in this repository:

* OMN-18608 — a bare ``mkdir`` lock with no holder record. A crash leaked it and
  every reconcile tick for the next 35 minutes printed "another reconcile-host
  is running" and exited 0 while ``ps`` showed no such process.
* OMN-18613 — a bare Kafka consumer ``commit()`` with no argument, which commits
  the consumer's POSITION for every assigned partition, past records the agent
  never looked at. One poll returned offsets 262 and 263; accepting 262
  committed 264.
* OMN-18606 — a bare interpreter running a repo-local script with the project
  venv not on PATH. The workflow merged green and failed on every real run.

This module is the gate. It is run by ``.github/workflows/``
``governed-helper-primitive-gate.yml`` and by the ``governed-helper-primitive``
pre-commit hook, both invoking the same ``scripts/ci/governed_helper_gate.py``,
so a local verdict and a CI verdict cannot diverge. That construction is taken
from ``tests/test_no_raw_prod_bypass_policy.py``, the repository's established
shape for a committed-recipe gate.

**Why the checker is standard library only.** It sits on a fail-closed
enforcement path that runs in pre-commit, where the resolved interpreter may be
a bare system python. A missing third-party parser there would turn into a
refusal of every commit on the machine, so the policy and baseline are JSON and
are read with ``json``. This is the reason ``ticket_creation_policy.json``
records for the same choice.

**Why there is no suppression annotation.** The class produced four defects in
one night. A per-call-site annotation would let the lane introducing the defect
also grant itself the exemption, which is the shape that left roughly 234
self-written suppressions unreviewed in one repository. A pair that is wrong is
corrected in the policy file, where the change is reviewable.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_MODULE = REPO_ROOT / "scripts" / "ci" / "governed_helper_gate.py"
POLICY_PATH = REPO_ROOT / "config" / "governed_helper_policy.json"
BASELINE_PATH = REPO_ROOT / "config" / "governed_helper_baseline.json"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))

import governed_helper_gate as gate

# ---------------------------------------------------------------------------
# AC1 — one checker, one policy file, a pair is data
# ---------------------------------------------------------------------------


def test_the_shipped_policy_declares_the_three_increment_one_pairs() -> None:
    """The policy is data the checker reads, not literals the checker carries."""
    policy = gate.load_policy(POLICY_PATH)
    assert {pair.id for pair in policy.pairs} == {
        "lock-without-holder-record",
        "kafka-consumer-commit-unbounded",
        "interpreter-without-project-venv",
    }


def test_every_pair_declares_all_four_required_fields() -> None:
    """A pair is (bare primitive, governed helper, scope, establishing ticket)."""
    for pair in gate.load_policy(POLICY_PATH).pairs:
        assert pair.bare_pattern, pair.id
        assert pair.governed_helper, pair.id
        assert pair.scope_globs, pair.id
        assert re.fullmatch(r"OMN-\d+", pair.established_by), pair.id


def test_a_pair_the_checker_has_never_heard_of_still_produces_findings(
    tmp_path: Path,
) -> None:
    """AC1's falsifier: a fourth pair is a data edit, with no checker change.

    The pair below names a primitive this repository's policy does not carry.
    If the checker had the three shipped pairs compiled into it, this would
    find nothing.
    """
    scanned = tmp_path / "some_script.sh"
    scanned.write_text("#!/bin/bash\nnohup frobnicate --forever &\n", encoding="utf-8")
    policy = gate.Policy(
        pairs=(
            gate.Pair(
                id="a-pair-invented-for-this-test",
                bare_pattern=r"\bnohup\b",
                governed_helper="scripts/supervise.sh:1",
                governed_form="supervise it",
                scope_globs=("*.sh",),
                exclude_globs=(),
                governed_prefix_pattern=None,
                established_by="OMN-18629",
            ),
        )
    )
    findings = gate.scan(policy, gate.Baseline(entries=()), tmp_path, [scanned.name])
    assert [f.pair_id for f in findings.new] == ["a-pair-invented-for-this-test"]


# ---------------------------------------------------------------------------
# AC2 — a new occurrence is refused, per pair, with a negative control
# ---------------------------------------------------------------------------

#: One row per shipped pair: the bare form that must be refused, and the
#: governed form in the same file that must NOT be.
PAIR_FIXTURES: tuple[tuple[str, str, str, str], ...] = (
    (
        "lock-without-holder-record",
        "lockme.sh",
        'if mkdir "${LOCK_DIR}" 2>/dev/null; then :; fi\n',
        'if acquire_lock_with_holder "${LOCK_DIR}"; then :; fi\n',
    ),
    (
        "kafka-consumer-commit-unbounded",
        "consume_it.py",
        "async def f(self):\n    await self._consumer.commit()\n",
        "async def f(self):\n    await self._commit_through(msg)\n",
    ),
    (
        "interpreter-without-project-venv",
        "runit.sh",
        '#!/bin/bash\npython3 "${SCRIPT_DIR}/lane_census_inventory.py"\n',
        '#!/bin/bash\nuv run python "${SCRIPT_DIR}/lane_census_inventory.py"\n',
    ),
)


@pytest.mark.parametrize(
    ("pair_id", "filename", "bare", "governed"),
    PAIR_FIXTURES,
    ids=[row[0] for row in PAIR_FIXTURES],
)
def test_a_new_bare_occurrence_is_refused_and_the_governed_form_is_not(
    pair_id: str, filename: str, bare: str, governed: str, tmp_path: Path
) -> None:
    """AC2, both directions: the bare form fails, the governed form passes."""
    policy = gate.load_policy(POLICY_PATH)
    empty = gate.Baseline(entries=())

    (tmp_path / filename).write_text(bare, encoding="utf-8")
    refused = gate.scan(policy, empty, tmp_path, [filename])
    assert [f.pair_id for f in refused.new] == [pair_id], (
        f"the bare form for {pair_id} was not refused"
    )

    # NEGATIVE CONTROL: the governed form, same pair, same file name.
    (tmp_path / filename).write_text(governed, encoding="utf-8")
    allowed = gate.scan(policy, empty, tmp_path, [filename])
    assert allowed.new == [], (
        f"the GOVERNED form for {pair_id} was refused: {allowed.new}"
    )


def test_the_refusal_names_the_file_the_line_the_pair_and_the_helper(
    tmp_path: Path,
) -> None:
    """AC2: a refusal a reader cannot act on is a refusal that gets routed around."""
    policy = gate.load_policy(POLICY_PATH)
    (tmp_path / "lockme.sh").write_text(
        '#!/bin/bash\n\nif mkdir "${LOCK_DIR}"; then :; fi\n', encoding="utf-8"
    )
    findings = gate.scan(policy, gate.Baseline(entries=()), tmp_path, ["lockme.sh"])
    rendered = gate.render(findings)
    assert "lockme.sh" in rendered
    assert ":3" in rendered
    assert "lock-without-holder-record" in rendered
    assert "scripts/reconcile-host.sh" in rendered
    assert "OMN-18608" in rendered


# ---------------------------------------------------------------------------
# AC3 — a baseline entry with no ticket is refused by the gate itself
# ---------------------------------------------------------------------------


def test_a_baseline_entry_without_a_ticket_is_refused(tmp_path: Path) -> None:
    """AC3's falsifier: growing the baseline without filing work is unavailable."""
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "pair": "lock-without-holder-record",
                        "path": "scripts/deploy-runtime.sh",
                        "line_sha256_12": "0" * 12,
                        "occurrences": 1,
                        "ticket": "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(gate.PolicyError) as excinfo:
        gate.load_baseline(path)
    assert "ticket" in str(excinfo.value)
    assert "scripts/deploy-runtime.sh" in str(excinfo.value)


def test_every_shipped_baseline_entry_cites_a_ticket() -> None:
    """AC3 on the committed artifact, not only on a fixture."""
    for entry in gate.load_baseline(BASELINE_PATH).entries:
        assert re.fullmatch(r"OMN-\d+", entry.ticket), entry


# ---------------------------------------------------------------------------
# AC4 — no suppression annotation, no skip surface
# ---------------------------------------------------------------------------


def test_the_checker_declares_no_skip_or_force_option() -> None:
    """AC4: read the parser's own option strings, so a return is a red test."""
    options = {
        option
        for action in gate.build_parser()._actions
        for option in action.option_strings
    }
    forbidden = {"--force", "--skip", "--no-fail", "--warn-only", "--allow", "--ignore"}
    assert not (options & forbidden), options


def test_the_checker_honours_no_annotation_and_no_environment_override() -> None:
    """AC4: an annotation on the offending line does not suppress the finding."""
    policy = gate.load_policy(POLICY_PATH)
    (tmp := Path(gate.__file__).parent).exists()  # keep the import meaningful
    del tmp
    source = GATE_MODULE.read_text(encoding="utf-8")
    for token in ("-ok:", "noqa: governed", "governed-helper-ok", "GOVERNED_HELPER_"):
        assert token not in source, (
            f"the checker carries a suppression or override surface: {token!r}"
        )
    assert "os.environ" not in source, (
        "an environment variable read is a skip surface by another name"
    )
    assert policy.pairs, "sanity: the policy loaded"


# ---------------------------------------------------------------------------
# AC5 — wired as a pre-commit hook and a CI job, asserted in the summary tuple
# ---------------------------------------------------------------------------


def test_the_gate_is_wired_as_a_precommit_hook() -> None:
    config = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "id: governed-helper-primitive" in config


def test_the_gate_is_wired_as_a_ci_workflow() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "governed-helper-primitive-gate.yml"
    )
    assert workflow.is_file()
    body = workflow.read_text(encoding="utf-8")
    assert "governed_helper_gate.py" in body
    # No conditional on the gating step, and no skip input: an `if:` on the
    # enforcing step is how a required context quietly stops reporting.
    assert "continue-on-error" not in body


def test_the_gate_context_is_asserted_in_the_summary_gate_tuple() -> None:
    """AC5: on this repo the summary umbrella IS the enforcement surface."""
    summary = (REPO_ROOT / "scripts" / "ci" / "ci_summary_gate.py").read_text(
        encoding="utf-8"
    )
    assert "Governed helper primitive gate" in summary


# ---------------------------------------------------------------------------
# AC6 (labelled criterion) — the baseline may shrink and may never widen
# ---------------------------------------------------------------------------


def test_a_second_identical_occurrence_in_a_baselined_file_is_refused(
    tmp_path: Path,
) -> None:
    """The ratchet: baselining one call site does not baseline the next one."""
    policy = gate.load_policy(POLICY_PATH)
    line = 'if mkdir "${LOCK_DIR}"; then :; fi\n'
    (tmp_path / "lockme.sh").write_text(line, encoding="utf-8")
    baseline = gate.Baseline(
        entries=(
            gate.BaselineEntry(
                pair="lock-without-holder-record",
                path="lockme.sh",
                line_sha256_12=gate.line_digest(line.strip()),
                occurrences=1,
                ticket="OMN-18608",
            ),
        )
    )
    assert gate.scan(policy, baseline, tmp_path, ["lockme.sh"]).new == []

    (tmp_path / "lockme.sh").write_text(line + line, encoding="utf-8")
    widened = gate.scan(policy, baseline, tmp_path, ["lockme.sh"])
    assert len(widened.new) == 1, "a second occurrence rode in on the first's entry"


def test_a_baseline_entry_the_scanner_no_longer_matches_is_refused(
    tmp_path: Path,
) -> None:
    """AC6's falsifier: a fixed occurrence may not leave cover behind."""
    policy = gate.load_policy(POLICY_PATH)
    line = 'if mkdir "${LOCK_DIR}"; then :; fi\n'
    (tmp_path / "lockme.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    baseline = gate.Baseline(
        entries=(
            gate.BaselineEntry(
                pair="lock-without-holder-record",
                path="lockme.sh",
                line_sha256_12=gate.line_digest(line.strip()),
                occurrences=1,
                ticket="OMN-18608",
            ),
        )
    )
    findings = gate.scan(policy, baseline, tmp_path, ["lockme.sh"])
    assert findings.new == []
    assert len(findings.stale) == 1
    assert "lockme.sh" in gate.render(findings)


def test_the_committed_baseline_matches_the_committed_tree_exactly() -> None:
    """The shipped artifacts agree: no stale entry, no unbaselined occurrence.

    This is the test that fails the day someone fixes a baselined call site and
    forgets to delete its entry, and the day someone adds a new one.
    """
    result = subprocess.run(
        [sys.executable, str(GATE_MODULE)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"the gate refuses the committed tree:\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# The gate fails closed on its own inputs
# ---------------------------------------------------------------------------


def test_an_unreadable_policy_is_a_refusal_not_a_pass(tmp_path: Path) -> None:
    """A gate that cannot read its own rules has not passed; it has not run."""
    with pytest.raises(gate.PolicyError):
        gate.load_policy(tmp_path / "does-not-exist.json")


def test_a_pair_with_an_uncompilable_pattern_is_a_refusal(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "id": "broken",
                        "bare_pattern": "(unclosed",
                        "governed_helper": "x:1",
                        "governed_form": "y",
                        "scope_globs": ["*.sh"],
                        "established_by": "OMN-18629",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(gate.PolicyError):
        gate.load_policy(path)


def test_the_gate_excludes_only_its_own_surfaces() -> None:
    """The self-exclusion list is closed, enumerated, and about this gate only.

    Every entry must be a file this gate owns. An unrelated path appearing here
    would be an allowlist entry wearing a different hat, which is the thing the
    baseline's ticket requirement and the absence of an annotation both exist to
    prevent.
    """
    for path in gate.ALWAYS_EXCLUDED:
        assert "governed_helper" in path or "omn18629" in path, path
        assert (REPO_ROOT / path).is_file(), f"{path} is excluded but does not exist"
