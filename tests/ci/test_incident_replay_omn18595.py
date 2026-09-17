# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay: the release train must not cut on its own bookkeeping (OMN-18595).

THE INCIDENT
------------
``scripts/ci/post_release_dev_bump.py`` opens a pull request after every release
that moves ``[project].version`` to the next patch, so the release-identity gate
is disarmed for the next merge (OMN-13912). That pull request edits
``pyproject.toml``, and ``pyproject.toml`` is a release-relevant path -- it
carries the dependency floor that ships with the wheel.

So the morning after every release, a decision surface that counts commits
touching release-relevant paths sees exactly one, and it is the machinery's own
bookkeeping. A train armed on that count opens a release pull request containing
no product change, that release publishes, its own post-release bump lands, and
the count is one again. Once per release, forever.

This is not a hypothesis about the future. It is the state ``origin/dev`` was in
when this guard was written: ``git rev-list --count v0.38.30..origin/dev -- src
pyproject.toml`` returned 1, and ``git log --format=%s`` over the same range
returned exactly one line, the bump. The sibling decision surface in omnimarket
had to grow its own marker for the identical reason, which its module docstring
records: without one, the post-release commit "re-triggers the workflow, and
releases an empty version -- once per merge, forever".

WHAT IS REPLAYED, AND OVER WHICH BYTES
--------------------------------------
The guard decides on a commit SUBJECT. Every repo in this registry is squash-only,
so a merged pull request's title IS the subject of the commit it produces, plus
the ``(#n)`` suffix git appends. The captured bytes are therefore the two pull
requests themselves, fetched from the REST API rather than retyped:

* ``pulls/3693`` -- the post-release bump to 0.38.31, merge commit
  ``695305b303dfd1e1d1b68a2885ea34641cd251ea``, which is the head of
  ``origin/dev``. This is the bad input.
* ``pulls/3683`` -- ``feat(OMN-18567)``, merge commit
  ``2a4e49511d421c368adb5097131c5d1b985272c3``. Real runtime work, and the
  discriminator.

THE DISCRIMINATOR IS NOT A FORMALITY
------------------------------------
A classifier that called every subject bookkeeping would replay this incident
perfectly and make the train cut nothing, ever -- which is exactly as broken and
much harder to notice, because a train that never cuts looks like a quiet week.
So the same function is driven over the second fixture and required to say the
opposite.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _REPO_ROOT / "scripts" / "ci" / "release_train.py"
_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "omn18595"
_BUMP = _FIXTURES / "pull-3693-post-release-bump.json.captured"
_WORK = _FIXTURES / "pull-3683-runtime-work.json.captured"


def _load() -> Any:
    name = "release_train_incident_replay"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _MODULE)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        msg = f"cannot load {_MODULE}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rt = _load()


def _subject_from(fixture: Path) -> str:
    """Derive the squash subject from the captured pull request, as git does.

    Read out of the captured bytes rather than written into this file, so the
    test cannot drift away from the artifact it claims to replay.
    """
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    return f"{payload['title']} (#{payload['number']})"


def _policy() -> Any:
    """The shipped policy for the repo the incident happened in."""
    return rt.policy_for(rt.load_policy(rt.DEFAULT_POLICY_PATH), "omnibase_infra")


def _facts_from(subjects: tuple[str, ...], head_sha: str) -> Any:
    """Build facts the way the real collector does: filter, then count.

    The collector's only other work is the git call that produces the subject
    list, which these fixtures stand in for.
    """
    kept = tuple(s for s in subjects if not rt.is_release_bookkeeping_subject(s))
    return rt.ModelRepoFacts(
        repo="omnibase_infra",
        latest_tag="v0.38.30",
        dev_version="0.38.31",
        dev_head_sha=head_sha,
        unreleased_count=len(kept),
        unreleased_subjects=kept,
    )


def _no_receipt_seam() -> dict[str, Any]:
    """The lab seam, never reached on the bookkeeping path.

    Supplied so the test cannot silently pass because the run reached the
    network and failed there instead of on the classification under test.
    """

    def _list(repo: str, name: str) -> list[dict[str, Any]]:
        msg = "the lab seam must not be reached when there is no unreleased work"
        raise AssertionError(msg)

    def _download(repo: str, artifact_id: int) -> Any:  # pragma: no cover
        msg = "the lab seam must not be reached when there is no unreleased work"
        raise AssertionError(msg)

    return {"list_artifacts": _list, "download_receipt": _download}


def test_the_real_guard_rejects_the_post_release_bump_as_releasable_work() -> None:
    """The captured bad input, driven through the real decision.

    ``origin/dev`` carried exactly this one commit over release-relevant paths
    when the guard was written. Without the classification the train cuts
    v0.38.31 tonight with nothing in it.
    """
    subject = _subject_from(_BUMP)
    assert subject.startswith("chore(OMN-13912): post-release dev version bump")

    assert rt.is_release_bookkeeping_subject(subject), (
        "the post-release bump was classified as releasable work; a release cut "
        "on it publishes a version containing no product change"
    )

    payload = json.loads(_BUMP.read_text(encoding="utf-8"))
    decision = rt.decide(
        policy=_policy(),
        facts=_facts_from((subject,), payload["merge_commit_sha"]),
        **_no_receipt_seam(),
    )
    assert decision.verdict is rt.EnumTrainVerdict.SKIP
    assert decision.reason is rt.EnumTrainReason.NO_UNRELEASED_RELEASE_RELEVANT_WORK
    assert decision.unreleased_count == 0


def test_the_same_guard_accepts_real_runtime_work() -> None:
    """The discriminator. A classifier that swallowed everything would be worse.

    Same function, same code path, one fixture changed. Required to say the
    opposite, because a train that never cuts looks like a quiet week rather
    than a break.
    """
    subject = _subject_from(_WORK)
    assert subject.startswith("feat(OMN-18567):")

    assert not rt.is_release_bookkeeping_subject(subject), (
        "real runtime work was classified as release bookkeeping; the train "
        "would then never cut, and would look idle rather than broken"
    )

    payload = json.loads(_WORK.read_text(encoding="utf-8"))
    facts = _facts_from((subject,), payload["merge_commit_sha"])
    assert facts.unreleased_count == 1


def test_the_two_fixtures_are_not_the_same_bytes() -> None:
    """Both captures are real and distinct, so neither test is vacuous."""
    assert _BUMP.read_bytes() != _WORK.read_bytes()
    for fixture in (_BUMP, _WORK):
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        assert payload["merged"] is True
        assert payload["base"]["ref"] == "dev"


def test_a_subject_that_merely_mentions_a_release_is_still_work() -> None:
    """The narrow half of the classification, stated as its own obligation.

    The patterns anchor at the conventional-commit prefix. A loose match would
    swallow any commit whose subject contains the word, which is how a guard
    written to stop an empty release ends up stopping every release.
    """
    assert not rt.is_release_bookkeeping_subject(
        "docs(OMN-18595): describe how the release train decides to cut (#1)"
    )
    assert not rt.is_release_bookkeeping_subject(
        "fix(OMN-18010): the release decision reads the wrong branch (#2)"
    )
