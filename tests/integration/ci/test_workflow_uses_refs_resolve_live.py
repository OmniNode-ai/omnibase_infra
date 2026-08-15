# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Live half of the cross-repo ``uses:`` pin resolution gate (OMN-14941).

Static extraction + pin-expectation regression tests live in
``tests/ci/test_workflow_uses_refs_resolve.py`` (hermetic, run at pre-push by
the governed impacted-test selector). This module holds ONLY the live GitHub
contents-API resolution test, and lives under ``tests/integration/`` because:

* the pre-push selector ALWAYS ignores ``tests/integration`` by design (it
  needs live infra — ``scripts/hooks/prepush_smart_tests.sh``), and this test
  needs the live GitHub API;
* the enforcement surface of this gate is CI (the full-suite job runs all of
  ``tests/`` excluding only ``tests/integration/docker`` and the
  slow/chaos/kafka/performance markers), where it FAILS CLOSED: a definitive
  404 fails, and an unverifiable pin (no network / rate-limited) also fails
  when ``CI`` is set;
* a red here must not make the branch unpushable at the local pre-push hook,
  which is exactly what happened while this test sat in ``tests/ci/``.

**This test is expected GREEN. There is no standing authorization for a red
here — if it fails, a pin is broken; fix the pin.** (OMN-15248.) The file
previously carried a born-path pre-authorization: while the omniclaude
OMN-14941 reusable (``call-occ-companion-effect-reusable.yml``) was unmerged,
the ``@dev`` pin in ``call-occ-companion-effect.yml`` was a genuine 404 and
this test was documented as deliberately red. That condition EXPIRED — both
formerly-404 pins resolve on omniclaude ``dev`` (verified 2026-07-27 via
``gh api repos/OmniNode-ai/omniclaude/contents/<path>?ref=dev``:
``call-occ-companion-effect-reusable.yml`` -> ``c540d981``,
``call-occ-autobind-reusable.yml`` -> ``5f8f64e4``). An expired
red-authorization is how a real regression gets waved through: a reader sees
a blessed failure and stops investigating. Never restate one here — if a pin
genuinely cannot resolve yet, the born-path ordering is the fix (land the
upstream ref first), not a docstring waiver. Mechanizing that rule (so the
next stale waiver self-retires instead of aging into a blind spot) is
OMN-15257; until it lands this paragraph is prose, not enforcement.
"""

from __future__ import annotations

import os

import pytest

from tests.ci.test_workflow_uses_refs_resolve import (
    _ORG_PREFIX,
    _TRANSPORT_FAILURE_CIRCUIT_BREAKER,
    WORKFLOWS_DIR,
    _extract_cross_repo_uses,
    _resolve_ref_live,
)


@pytest.mark.integration
def test_every_cross_repo_uses_ref_resolves_live() -> None:
    refs = _extract_cross_repo_uses(WORKFLOWS_DIR)
    assert refs, "no cross-repo uses: pins extracted — extractor is broken"

    unique_targets = sorted({(r.repo, r.path, r.ref) for r in refs})
    broken: list[str] = []
    undetermined: list[str] = []
    transport_failures = 0

    for repo, path, ref in unique_targets:
        if transport_failures >= _TRANSPORT_FAILURE_CIRCUIT_BREAKER:
            undetermined.append(
                f"{_ORG_PREFIX}{repo}/{path}@{ref} (not probed: circuit "
                "breaker open after repeated transport failures)"
            )
            continue
        resolved, detail = _resolve_ref_live(repo, path, ref)
        pin = f"{_ORG_PREFIX}{repo}/{path}@{ref}"
        if resolved is False:
            broken.append(f"{pin} -> {detail}")
        elif resolved is None:
            undetermined.append(f"{pin} -> {detail}")
            if detail.startswith("transport error"):
                transport_failures += 1

    if broken:
        pinned_by = {f"{r.repo}/{r.path}@{r.ref}": r.workflow for r in refs}
        lines = [
            f"  {entry}  (pinned in "
            f"{pinned_by.get(entry.split(' -> ')[0][len(_ORG_PREFIX) :], '?')})"
            for entry in broken
        ]
        pytest.fail(
            "cross-repo `uses:` pins that DO NOT resolve on the live remote "
            "(the OMN-14941/E1 silent-outage class — the workflow fails at "
            "parse time and no job ever runs):\n"
            + "\n".join(lines)
            + "\n\nFIX THE PIN, DO NOT WAIVE THIS TEST. Every red here is a "
            "live broken workflow reference. Take exactly one of:\n"
            "  1. the target ref moved or was never merged -> re-pin the "
            "`uses:` line to a ref that exists on the remote (a merged SHA, "
            "or the branch the file actually lives on), or land the upstream "
            "PR that creates it FIRST (born-path ordering);\n"
            "  2. the path/filename changed upstream -> update the path in "
            "the `uses:` line;\n"
            "  3. the repo is private/unreachable to this token -> that is an "
            "`undetermined`, not a 404; a 404 here means GitHub answered "
            "definitively that the path does not exist at that ref.\n"
            "Adding an xfail/skip or a 'deliberately red pending <ticket>' "
            "docstring is NOT an accepted fix (OMN-15248)."
        )

    if undetermined:
        detail = "\n".join(f"  {entry}" for entry in undetermined)
        if os.environ.get("CI"):
            pytest.fail(
                "could not resolve these cross-repo `uses:` pins against the "
                "live GitHub API — failing CLOSED in CI (an unverifiable pin "
                "is not a passing pin):\n"
                + detail
                + "\n\nFIX THE RESOLUTION PATH, DO NOT WAIVE THIS TEST: "
                "HTTP 401/403/404-on-a-private-repo -> thread a token with "
                "read access to the target repo into the test step "
                "(`GH_TOKEN`/`GITHUB_TOKEN` env, e.g. CROSS_REPO_PAT for "
                "private cross-repo reads); HTTP 403/429 rate-limit -> the "
                "run is unauthenticated, supply the token; transport errors "
                "-> fix runner egress to api.github.com."
            )
        pytest.skip(
            "GitHub API unreachable from this local machine; live resolution "
            "gate enforced in CI. Unverified pins:\n" + detail
        )
