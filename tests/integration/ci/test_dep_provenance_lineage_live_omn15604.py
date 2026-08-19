# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live half of the dep-provenance content-lineage gate (OMN-15604).

Hermetic decision-logic tests (fake resolver, no network) live in
``tests/scripts/test_check_dep_provenance_lineage_omn15604.py``. This module
holds only the tests that hit the live GitHub REST API, under
``tests/integration`` for the same reason as the OMN-15538 pin-reachability
gate's live half: the pre-push selector always ignores this tree, so a
transient network failure here can never make a branch locally unpushable.

What this proves, against the REAL commits (not synthetic values, per memory
``feedback_prove_red_against_exists_but_wrong``):

1. **RED on the actual incident pin.** ``omnibase_core`` rev ``3d51b047`` (the
   ``omnibase_infra@dev`` pyproject.toml:222 pin this ticket exists because
   of) has a ``src/`` tree that genuinely differs from released tag
   ``v0.46.8``'s ``src/`` tree. ``resolve_src_tree_sha`` must resolve both to
   their real GitHub tree SHAs and ``find_lineage_violations`` must flag the
   mismatch.
2. **GREEN on a pin that IS the release.** Pinning tag ``v0.46.8``'s own
   resolved commit against declared version ``0.46.8`` must report no
   violation -- content identical, by construction.

**Merge-path decoupling (OMN-16096).** Every test in this module is marked
``live_github_api`` and is EXCLUDED from the merge-required
``test-parallel`` split in ``.github/workflows/ci.yml`` (see that job's
``-m`` filter). The module still runs, unmodified, on a scheduled canary
(``.github/workflows/dep-provenance-lineage-live-canary.yml``) that alerts on
failure instead of blocking merges -- two canary attempts on PR #2758 burned
on ``transport error: The read operation timed out`` from this module's live
GitHub calls during an egress-degraded window, coupling merge eligibility to
network weather. The actual merge-required gate
(``dep-provenance-lineage-gate`` in ``ci.yml``) now resolves lineage through
``resolve_src_tree_sha_hermetic`` (the OMN-16053 host-level git mirror,
falling back to this module's same live REST resolver only when the mirror
cannot serve the ref) -- so it no longer needs github.com reachability on the
fleet where the mirror is served. Nothing about this module's own assertions
or resolution transport changed; only its position in the CI graph did.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live_github_api

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check_dep_provenance.py"
_IN_CI = bool(os.environ.get("CI"))

# The exact live incident pin (OMN-15604 ticket evidence).
_PINNED_REV = "3d51b047a43ee412a7521502619d35c216dc7811"
_RELEASED_TAG_COMMIT = "105f7ce0a8f4b31f6f01fc94e9b43e75984f166a"  # v0.46.8


def _load_module():
    spec = importlib.util.spec_from_file_location("check_dep_provenance", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return _load_module()


def _require_resolved(sha: str | None, detail: str, label: str) -> str:
    """Fail closed in CI, skip only on a developer machine, on a network miss."""
    if sha is not None:
        return sha
    message = f"{label}: could not resolve via the live GitHub API ({detail})"
    if _IN_CI:
        pytest.fail(f"{message} — an unresolvable pin is not a passing pin")
    pytest.skip(message)
    raise AssertionError("unreachable")  # pragma: no cover


@pytest.mark.integration
def test_live_pinned_and_released_src_trees_are_independently_resolvable(mod) -> None:
    """Sanity: both real refs resolve to real (and DIFFERENT) tree SHAs.

    This is the ground-truth measurement the ticket's own evidence cites
    (`git rev-parse 3d51b047:src` = a74566fd..., `git rev-parse v0.46.8:src`
    = 008efdba...) -- reproduced here via the REST path the CI gate actually
    uses, not the local-clone path the ticket used to discover the defect.
    """
    pinned_sha, pinned_detail = mod.resolve_src_tree_sha("omnibase_core", _PINNED_REV)
    released_sha, released_detail = mod.resolve_src_tree_sha("omnibase_core", "v0.46.8")
    pinned_sha = _require_resolved(pinned_sha, pinned_detail, "pinned rev 3d51b047")
    released_sha = _require_resolved(released_sha, released_detail, "released v0.46.8")

    assert pinned_sha == "a74566fd92b0ca9bb86919df5e7f804cc4307793"
    assert released_sha == "008efdba12b39cf04d90d17468523daa281fe4fd"
    assert pinned_sha != released_sha


@pytest.mark.integration
def test_live_red_on_the_real_incident_pin(mod) -> None:
    """RED-before proof: the exact live pin (3d51b047 + ==0.46.8), not a
    synthetic value, drives find_lineage_violations end to end."""
    pyproject_text = (
        "[project]\n"
        'name = "omnibase-infra"\n'
        'version = "0.0.0"\n'
        "dependencies = [\n"
        '    "omnibase-core==0.46.8",\n'
        "]\n"
        "\n"
        "[tool.uv.sources]\n"
        'omnibase-core = { git = "https://github.com/OmniNode-ai/omnibase_core.git", '
        f'rev = "{_PINNED_REV}" }}  # raw-override-ok: OMN-15414\n'
    )

    violations = mod.find_lineage_violations(pyproject_text)
    undetermined = [v for v in violations if "UNDETERMINED lineage" in v]
    if undetermined:
        _require_resolved(None, undetermined[0], "live lineage check")
        return  # pragma: no cover - _require_resolved always fails/skips above

    assert len(violations) == 1, violations
    assert "omnibase-core" in violations[0]
    assert "differs from" in violations[0]


@pytest.mark.integration
def test_live_green_when_pin_is_the_release_commit(mod) -> None:
    """GREEN-after proof: pinning the release tag's own commit reports clean."""
    pyproject_text = (
        "[project]\n"
        'name = "omnibase-infra"\n'
        'version = "0.0.0"\n'
        "dependencies = [\n"
        '    "omnibase-core==0.46.8",\n'
        "]\n"
        "\n"
        "[tool.uv.sources]\n"
        'omnibase-core = { git = "https://github.com/OmniNode-ai/omnibase_core.git", '
        f'rev = "{_RELEASED_TAG_COMMIT}" }}\n'
    )

    violations = mod.find_lineage_violations(pyproject_text)
    undetermined = [v for v in violations if "UNDETERMINED lineage" in v]
    if undetermined:
        _require_resolved(None, undetermined[0], "live lineage check (green case)")
        return  # pragma: no cover - _require_resolved always fails/skips above

    assert violations == []


@pytest.mark.integration
def test_live_this_repos_own_pyproject_is_lineage_clean() -> None:
    """This repo's own tree (AC1): no forbidden git override present at all,
    so the lineage check has nothing to compare -- zero live calls, exit 0."""
    mod = _load_module()
    repo_root = Path(__file__).resolve().parents[3]
    pyproject_path = repo_root / "pyproject.toml"
    text = pyproject_path.read_text()

    violations = mod.find_lineage_violations(text)
    assert violations == [], (
        "omnibase_infra's own pyproject.toml should carry no forbidden "
        f"git-pinned override post-OMN-14628; got: {violations}"
    )


@pytest.mark.integration
def test_live_omn_15414_resolves_to_done(mod) -> None:
    """AC3 ground truth: the exact ticket the live incident's escape token
    cites (`# raw-override-ok: OMN-15414`) really is Done today, via the
    real Linear API -- not an assumed/stale fact. Skipped (not failed) when
    LINEAR_API_KEY is unavailable, matching the check's own graceful
    degradation posture."""
    if not os.environ.get("LINEAR_API_KEY"):
        pytest.skip("LINEAR_API_KEY not set in this environment")

    status_name, detail = mod.resolve_ticket_status("OMN-15414")
    if status_name is None:
        if _IN_CI:
            pytest.fail(f"could not resolve OMN-15414 via live Linear API: {detail}")
        pytest.skip(f"could not resolve OMN-15414 via live Linear API: {detail}")
    assert status_name.strip().lower() in mod._TICKET_DONE_STATUSES, (
        f"expected OMN-15414 to be closed (Done at ticket-file time), got "
        f"{status_name!r}"
    )


@pytest.mark.integration
def test_live_red_escape_token_end_to_end_against_the_real_incident_token(
    mod,
) -> None:
    """RED end-to-end: the EXACT live incident line (rev 3d51b047, token
    `# raw-override-ok: OMN-15414`, no until=) fails
    find_escape_token_violations, reproducing the ticket's cited proof
    requirement ("RED case where a well-formed token cites a Done ticket")
    against the real production code path.

    Deliberately does NOT skip on a missing LINEAR_API_KEY: LINEAR_API_KEY is
    not provisioned as a repo or org secret anywhere in OmniNode-ai today, so
    a skip-on-unset guard here would mean this proof never actually executes
    in the live enforcing CI environment -- precisely the gap a remediation
    round found ("Both live AC3 proofs also skip in CI"). The mandatory-
    until= enforcement path (`find_escape_token_violations`) fires on this
    exact token shape with zero network calls, so this test is unconditional.
    """
    pyproject_text = (
        "[project]\n"
        'name = "omnibase-infra"\n'
        'version = "0.0.0"\n'
        "dependencies = [\n"
        '    "omnibase-core==0.46.8",\n'
        "]\n"
        "\n"
        "[tool.uv.sources]\n"
        'omnibase-core = { git = "https://github.com/OmniNode-ai/omnibase_core.git", '
        f'rev = "{_PINNED_REV}" }}  # raw-override-ok: OMN-15414\n'
    )

    violations = mod.find_escape_token_violations(pyproject_text)
    assert len(violations) == 1, violations
    assert "OMN-15414" in violations[0]
