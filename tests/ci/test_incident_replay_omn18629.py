# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18629 incident replay — the real guard against the real pre-fix bytes.

The guard is ``scripts/ci/governed_helper_gate.py``. This module drives it with
the exact bytes of the two files that produced the incidents it exists to catch,
fetched from this repository's own object store, not retyped.

**The replay already earned its keep.** Written against the first draft of the
``kafka-consumer-commit-unbounded`` pattern, this module returned ZERO findings
on the file that lost the record. The draft pattern required at least one
character before ``consumer``, so it matched ``self._consumer.commit()`` and
missed ``self.consumer.commit()`` -- which is what the deploy agent actually
wrote, on all seven of its commit paths. The gate would have shipped green,
enforcing nothing on the very incident it was built for, and every unit test
would have passed because they were written against the pattern rather than
against the failure. That is the exact shape OMN-15547 exists to refuse, and the
prefix is optional now because this test said so.

**Why a discriminator is mandatory.** A pattern that flagged every ``.commit()``
would replay this incident perfectly and would also refuse every database commit
in the repository, which is both wrong far more often than right and much harder
to notice, because a gate that fires everywhere gets read as noise and then gets
routed around. ``test_the_same_guard_does_not_fire_on_an_ordinary_database_commit``
drives the SAME function over the SAME captured bytes and requires the opposite
verdict on the lines that are not consumer commits.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn18629"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))

import governed_helper_gate as gate

CONSUMER_PRE_FIX = FIXTURES / "deploy-agent-consumer-prefix-omn18613.py.captured"
CONSUMER_POST_FIX = FIXTURES / "deploy-agent-consumer-postfix-omn18613.py.captured"
RECONCILE_PRE_FIX = FIXTURES / "reconcile-host-prefix-omn18608.sh.captured"

#: Recorded so a reformatted or regenerated fixture is a failure rather than a
#: silent substitution. An artifact that has been touched is no longer the
#: artifact that failed.
EXPECTED_SHA256 = {
    CONSUMER_PRE_FIX: "320e666f9e2e640d275076b920342b80001b312faab777b0da13ced59106fb64",
    CONSUMER_POST_FIX: "b4f80138df34bc24623e5cee5079520ad30abfc9fc34f7b3441dd8d06191e969",
    RECONCILE_PRE_FIX: "15b444900f1a691095adcf3407eefcac9b97b9a76002c53157e5120ab5420720",
}


def _scan_captured(fixture: Path, as_name: str) -> gate.Findings:
    """Drive the REAL guard over the captured bytes, under a scanned filename.

    The fixture is copied to its original basename because the guard's scope is
    declared in globs over the path, so bytes scanned under `.captured` would
    exercise nothing. Scanned in a temporary directory rather than in place so
    the exclude globs that cover the helper's own file do not decide the result.
    """
    policy = gate.load_policy(REPO_ROOT / "config" / "governed_helper_policy.json")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shutil.copy(fixture, root / as_name)
        return gate.scan(policy, gate.Baseline(entries=()), root, [as_name])


@pytest.mark.parametrize("fixture", sorted(EXPECTED_SHA256), ids=lambda p: p.name)
def test_the_fixture_is_the_bytes_that_were_captured(fixture: Path) -> None:
    """R3: an unmodified capture, or the replay proves nothing."""
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256[fixture], (
        f"{fixture.name} is not the captured artifact any more"
    )


def test_the_real_guard_refuses_the_deploy_agent_that_lost_the_record() -> None:
    """OMN-18613 replayed: seven unbounded commits, none of them caught before.

    These are the bytes of `scripts/deploy-agent/deploy_agent/consumer.py` at
    5be72e12f^, the commit before the fix. One poll returned offsets 262 and
    263; accepting 262 committed 264, past the still-buffered 263, which was
    then never executed and left no job record, no acceptance line, no rejection
    line and no quarantine record.
    """
    findings = _scan_captured(CONSUMER_PRE_FIX, "consumer.py")
    commits = [
        f for f in findings.new if f.pair_id == "kafka-consumer-commit-unbounded"
    ]
    assert len(commits) == 7, (
        f"expected all seven unbounded commit paths, got {[f.line_no for f in commits]}"
    )
    assert all("consumer.commit()" in f.line_text for f in commits)
    # The refusal has to point somewhere a reader can go.
    rendered = gate.render(findings)
    assert "_commit_through" in rendered
    assert "OMN-18613" in rendered


def test_the_real_guard_refuses_the_lock_that_blocked_reconciles_for_35_minutes() -> (
    None
):
    """OMN-18608 replayed, against `scripts/reconcile-host.sh` at 914672948^.

    A bare mkdir released only by a trap on EXIT INT TERM is leaked permanently
    by a SIGKILL. A session died at about 15:25Z holding a lock taken at 15:23Z;
    every tick for the next 35 minutes printed "another reconcile-host is
    running; nothing to do" and exited 0 while `ps` showed no such process.
    """
    findings = _scan_captured(RECONCILE_PRE_FIX, "reconcile-host.sh")
    locks = [f for f in findings.new if f.pair_id == "lock-without-holder-record"]
    assert len(locks) == 1, [f.line_no for f in locks]
    assert "mkdir" in locks[0].line_text
    assert "LOCK_DIR" in locks[0].line_text


def test_the_same_guard_does_not_fire_on_an_ordinary_database_commit() -> None:
    """The discriminator: narrow, or the gate is noise that gets routed around.

    The pre-fix deploy agent is a file full of `.commit(` calls. Only the ones
    on a consumer are findings; a guard that flagged them all would refuse every
    database commit in the repository.
    """
    source = CONSUMER_PRE_FIX.read_text(encoding="utf-8")
    all_commit_lines = [line for line in source.splitlines() if ".commit(" in line]
    findings = _scan_captured(CONSUMER_PRE_FIX, "consumer.py")
    assert len(all_commit_lines) > len(findings.new), (
        "every .commit( line was flagged; the pair is not discriminating at all"
    )
    for finding in findings.new:
        assert "consumer.commit()" in finding.line_text


def test_the_fixed_file_has_no_unbounded_commit_left_in_its_code() -> None:
    """The other direction: the shipped fix genuinely removed the call sites.

    The post-fix file still matches ONCE, and deliberately: its own docstring
    quotes the bare form to explain why it is wrong. That line is prose, not a
    call site, and in the live tree the helper's own file is excluded by the
    pair's exclude globs for exactly this reason. Asserting the count here keeps
    that exclusion honest -- if the fixed file ever grows a real unbounded
    commit, this number moves.
    """
    findings = _scan_captured(CONSUMER_POST_FIX, "consumer.py")
    assert len(findings.new) == 1
    assert findings.new[0].line_text.startswith("A bare")
    code_lines = [
        line
        for line in CONSUMER_POST_FIX.read_text(encoding="utf-8").splitlines()
        if "_commit_through(" in line
    ]
    assert len(code_lines) >= 7, "the fix should have replaced every commit path"


def test_the_two_consumer_fixtures_are_not_the_same_bytes() -> None:
    """Neither assertion above can be vacuous if the pre and post differ."""
    assert CONSUMER_PRE_FIX.read_bytes() != CONSUMER_POST_FIX.read_bytes()
