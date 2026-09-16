# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17057: a sibling merge delivers ITS OWN revision, named, to staging.

WHAT WAS BROKEN, measured live 2026-09-16.

``deliver-dev-candidate-to-staging.yml`` fires on a runtime-path push to THIS
repository's ``dev``, and its candidate build clones every sibling at branch
``dev`` at that instant. So a merge to ``omnimarket`` could not cause a
delivery at all -- it rode along on whatever the next ``omnibase_infra`` push
happened to build, and the sibling revision that ended up in the image was
never CHOSEN.

    delivery run 35013603978 (2026-09-15T19:26Z, success)
      build manifest per_sibling_vcs_provenance.omnimarket.vcs_ref
        = 3038d1dabadea4973707f0228862b4aef4e09242

which is simply where ``omnimarket`` ``dev`` was pointing eighteen minutes
after an unrelated commit merged here. Twelve omnimarket commits had landed by
the time this was read.

The primitive to fix it already existed and CI was not reaching it.
``deploy_source_ref.py`` has taken ``--ref`` plus ``--fallback-ref`` since
OMN-17135, resolving the primary in whichever repository carries it and taking
the fallback everywhere else. The candidate build passed the same branch name
as both.

WHAT IS PINNED HERE.

The properties, parsed out of the workflow graph -- never a text match, because
every one of these would survive a rename:

  * the delivery workflow accepts a sibling announcement, and only that type
  * the candidate build declares the pin on BOTH of its trigger blocks, with
    empty defaults, so the push path is unchanged
  * the delivery workflow passes the pin through from the announcement and
    from nowhere else
  * the sibling receipt is read before anything is announced, and that read is
    conditioned on a pin being present
  * the concurrency group separates a sibling delivery from a push delivery,
    so the next push cannot silently cancel one
  * the announcement job needs the convergence job, so an unproven revision is
    never announced
  * nothing on the new path carries a force, skip or override input

The last is the one worth stating plainly: rule 24(b) fails delivery closed on
a missing lab receipt, and a path with an override is not fail-closed. A test
that reads the parsed inputs is how "somebody adds one later" becomes a red
test rather than a review catch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

DELIVER = WORKFLOWS / "deliver-dev-candidate-to-staging.yml"
BUILD = WORKFLOWS / "build-workspace-candidate-runtime.yml"
REUSABLE = WORKFLOWS / "runtime-rebuild-trigger-reusable.yml"

#: The announcement this path introduces. Named once, here.
SIBLING_EVENT = "sibling-dev-candidate"

#: Vocabulary that would turn a fail-closed gate into an advisory one.
OVERRIDE_TOKENS = ("force", "skip", "override", "bypass", "ignore_lab")


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(doc: dict[str, Any]) -> dict[str, Any]:
    """`on:` parses as the boolean True in YAML 1.1 — hence the two lookups."""
    raw = doc.get("on", doc.get(True))
    assert isinstance(raw, dict), "workflow declares no mapping of triggers"
    return raw


def _steps(doc: dict[str, Any], job: str) -> list[dict[str, Any]]:
    return list(doc["jobs"][job].get("steps") or [])


def _step_text(step: dict[str, Any]) -> str:
    """Everything a step says, flattened — `run:`, `with:`, `env:`, `if:`."""
    return json.dumps(step, sort_keys=True)


# ---------------------------------------------------------------------------
# the trigger
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_delivery_workflow_accepts_a_sibling_announcement() -> None:
    triggers = _triggers(_load(DELIVER))
    dispatch = triggers.get("repository_dispatch")
    assert dispatch, (
        "deliver-dev-candidate-to-staging.yml declares no repository_dispatch, "
        "so a sibling merge still has no way to cause a delivery of its own sha"
    )
    assert dispatch.get("types") == [SIBLING_EVENT], (
        "the sibling announcement must be the only accepted dispatch type; an "
        f"unfiltered repository_dispatch accepts anything, got {dispatch!r}"
    )


@pytest.mark.unit
def test_the_delivery_workflow_still_delivers_its_own_pushes() -> None:
    """The push path is the one that works today — a regression here is an outage."""
    triggers = _triggers(_load(DELIVER))
    push = triggers.get("push") or {}
    assert push.get("branches") == ["dev"], push


# ---------------------------------------------------------------------------
# the pin, on the candidate build
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("trigger", ["workflow_dispatch", "workflow_call"])
def test_the_candidate_build_declares_the_pin_with_an_empty_default(
    trigger: str,
) -> None:
    """Empty defaults are what keep the push path byte-identical in behaviour."""
    inputs = _triggers(_load(BUILD))[trigger]["inputs"]
    for name in ("pinned_repo", "pinned_revision"):
        assert name in inputs, f"{trigger} does not declare {name}"
        declared = inputs[name]
        assert declared.get("type") == "string", declared
        assert declared.get("required") is False, declared
        assert declared.get("default") == "", (
            f"{trigger}.{name} defaults to {declared.get('default')!r}; a "
            "non-empty default would silently pin every push-triggered build"
        )


@pytest.mark.unit
def test_sibling_ref_is_unchanged_so_the_branch_axis_is_preserved() -> None:
    """The pin ADDS an axis. Repurposing sibling_ref would break the other three."""
    inputs = _triggers(_load(BUILD))["workflow_call"]["inputs"]
    assert inputs["sibling_ref"]["default"] == "dev", inputs["sibling_ref"]


@pytest.mark.unit
def test_the_pin_reaches_the_staging_step_as_the_primary_ref() -> None:
    """The whole point: DEPLOY_REF must become the commit, not the branch."""
    doc = _load(BUILD)
    staging = [
        s
        for s in _steps(doc, "build-workspace-candidate")
        if "stage_workspace.sh" in str(s.get("run") or "")
    ]
    assert staging, "the candidate build no longer runs stage_workspace.sh"
    env = staging[0].get("env") or {}
    deploy_ref = str(env.get("DEPLOY_REF", ""))
    assert "pinned_revision" in deploy_ref, (
        f"DEPLOY_REF is {deploy_ref!r}; with no pinned_revision in it the "
        "staged sibling tree is still resolved from a branch"
    )
    assert "sibling_ref" in deploy_ref, (
        f"DEPLOY_REF is {deploy_ref!r}; it must fall back to sibling_ref so an "
        "unpinned push build is unchanged"
    )
    fallback = str(env.get("DEPLOY_SIBLING_FALLBACK_REF", ""))
    assert "sibling_ref" in fallback, (
        f"DEPLOY_SIBLING_FALLBACK_REF is {fallback!r}; the repositories that "
        "are NOT pinned must still resolve from the branch"
    )


@pytest.mark.unit
def test_the_candidate_build_verifies_what_it_actually_vendored() -> None:
    """A pin nobody checks is a claim, and the image bakes the claim."""
    doc = _load(BUILD)
    body = " ".join(
        str(s.get("run") or "") for s in _steps(doc, "build-workspace-candidate")
    )
    assert "sibling-vcs-provenance.json" in body
    assert "PINNED_REVISION" in body, (
        "no step compares the staged tree against the requested revision, so a "
        "candidate could ship provenance naming a commit it does not contain"
    )


@pytest.mark.unit
def test_the_build_manifest_records_which_revision_was_chosen() -> None:
    """A deliberate pin must be distinguishable from a branch that drifted there."""
    doc = _load(BUILD)
    manifest = [
        s
        for s in _steps(doc, "build-workspace-candidate")
        if "build-manifest.json" in str(s.get("run") or "")
    ]
    assert manifest, "the candidate build no longer writes a build manifest"
    assert "pinned_sibling" in str(manifest[0]["run"]), (
        "the manifest records per_sibling_vcs_provenance but not the pin, so a "
        "reader cannot tell a chosen revision from an incidental one"
    )


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_delivery_passes_the_pin_from_the_announcement_only() -> None:
    doc = _load(DELIVER)
    with_block = doc["jobs"]["build-runtime-candidate"]["with"]
    for name in ("pinned_repo", "pinned_revision"):
        expression = str(with_block[name])
        assert "client_payload" in expression, (
            f"{name} is passed as {expression!r}; a pin sourced from anywhere "
            "but the announcement is a hand-asserted delivery"
        )


@pytest.mark.unit
def test_the_sibling_receipt_is_read_before_anything_is_announced() -> None:
    doc = _load(DELIVER)
    gate_steps = _steps(doc, "lab-pass-gate")
    sibling = [
        s
        for s in gate_steps
        if "lab_pass_receipt.py gate" in " ".join(str(s.get("run") or "").split())
        and "client_payload" in _step_text(s)
    ]
    assert sibling, (
        "lab-pass-gate never reads the sibling repository's receipt, so a "
        "pinned revision could be delivered having been exercised nowhere"
    )
    step = sibling[0]
    assert "pinned_repo" in str(step.get("if", "")), (
        "the sibling read is unconditional; on a push delivery there is no "
        "sibling sha to ask about and it would fail every ordinary merge"
    )
    assert "compose-dev" in str(step["run"]), (
        "the sibling read must name the lane that actually covers the sha; "
        "asking about lanes nothing emits for a sibling reports a missing "
        "receipt for a lane that never claimed to cover it"
    )


@pytest.mark.unit
def test_the_announcement_still_depends_on_the_lab_gate() -> None:
    doc = _load(DELIVER)
    dispatch = doc["jobs"]["dispatch-to-staging"]
    assert "lab-pass-gate" in dispatch["needs"], dispatch["needs"]
    assert "if" not in dispatch, (
        "dispatch-to-staging carries an `if:`; GitHub's default success() is "
        "what makes a failed, skipped or cancelled gate leave it unfired"
    )


@pytest.mark.unit
def test_a_sibling_delivery_is_not_cancelled_by_the_next_push() -> None:
    doc = _load(DELIVER)
    group = str(doc["concurrency"]["group"])
    assert "pinned_repo" in group, (
        f"the concurrency group is {group!r}. On a repository_dispatch "
        "github.ref resolves to the default branch, so without a segment "
        "naming the pinned repo the next ordinary push cancels the sibling's "
        "delivery silently -- the invisible non-delivery this path removes"
    )


@pytest.mark.unit
def test_nothing_on_the_new_path_carries_an_override() -> None:
    """Rule 24(b) is fail-closed; an override input is how that stops being true."""
    for path in (DELIVER, BUILD, REUSABLE):
        doc = _load(path)
        for trigger, block in _triggers(doc).items():
            if not isinstance(block, dict):
                continue
            for name in block.get("inputs") or {}:
                assert not any(token in name.lower() for token in OVERRIDE_TOKENS), (
                    f"{path.name} {trigger} declares input {name!r}, which reads "
                    "as an override of a fail-closed gate"
                )


# ---------------------------------------------------------------------------
# the announcement
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_reusable_announces_only_after_convergence() -> None:
    doc = _load(REUSABLE)
    jobs = doc["jobs"]
    announce = [name for name in jobs if "deliver" in name]
    assert announce, (
        "the sibling rebuild trigger declares no delivery job, so a sibling "
        "merge's proven revision goes nowhere"
    )
    job = jobs[announce[0]]
    assert "verify-sibling-converged" in job["needs"], job["needs"]
    assert "if" not in job, (
        "the announcement carries an `if:`; default success() is what makes a "
        "failed or skipped convergence leave it unfired"
    )


@pytest.mark.unit
def test_the_announcement_names_the_merged_sha_and_refuses_anything_else() -> None:
    doc = _load(REUSABLE)
    job = next(j for name, j in doc["jobs"].items() if "deliver" in name)
    body = " ".join(str(s.get("run") or "") for s in job["steps"])
    assert SIBLING_EVENT in body, body[:400]
    assert "sibling_sha" in json.dumps(job), (
        "the announcement does not carry the merged sibling sha"
    )
    assert "[0-9a-f]{40}" in body, (
        "the announcement does not refuse a non-sha revision; a branch name "
        "here is the floating ref this whole path replaces"
    )


@pytest.mark.unit
def test_the_announcement_refuses_rather_than_skips_without_credentials() -> None:
    """A green run that announced nothing is the failure mode, not the safe state."""
    doc = _load(REUSABLE)
    job = next(j for name, j in doc["jobs"].items() if "deliver" in name)
    guard = [
        s
        for s in job["steps"]
        if "exit 1" in str(s.get("run") or "") and "APP_ID" in _step_text(s)
    ]
    assert guard, (
        "nothing in the announcement job fails on absent credentials, so a "
        "caller that has not been granted them would report a delivery that "
        "did not happen"
    )
