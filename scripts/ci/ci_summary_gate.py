# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed verdict for the ``CI Summary`` required-context poller (OMN-14127).

Why this exists
---------------
``CI Summary`` is the single required branch-protection context for
``omnibase_infra`` (OMN-4497). It used to be a ``needs``-gated aggregator job
depending on ~20 upstream jobs. A ``needs``-gated job gets **no** GitHub
check-run until its ``needs`` reach a terminal state, so under self-hosted
runner-fleet saturation the gate jobs never terminalized and ``CI Summary`` was
**absent** — the PR wedged ``BLOCKED`` forever with 0 failing / 0 pending
checks and no auto-recovery.

The ``ci-summary`` workflow job is now a NO-``needs``, GitHub-hosted poller: its
check-run instantiates immediately (so the required context can never be
absent), and it calls this module in a loop against the current run's job list
until a terminal verdict is reached (or a bounded deadline fires → fail-closed).

Verdict policy — DEFAULT-DENY, FAIL-CLOSED
------------------------------------------
This module reproduces the *exact* strictness of the old needs-based
``ci-summary`` pass/fail condition and then adds a strictly-stronger safety net.
Three independent checks; all must be satisfied for success:

1. **Strict aggregate gates.** :data:`STRICT_GATE_JOBS` must each be *present*,
   *completed*, and conclude ``success`` — a ``skipped``/``failure``/
   ``cancelled`` conclusion fails the gate. These jobs are unconditional in
   ``ci.yml`` (no ``if:``), so they never legitimately skip on
   ``pull_request``/``merge_group``/``push``; treating a skip as a failure is
   the same fail-closed behavior the old ``== "success"`` condition had.

2. **Skippable aggregate gates.** :data:`SKIPPABLE_GATE_JOBS` must each be
   *present*, *completed*, and conclude ``success`` **or** ``skipped`` — these
   jobs carry a legitimate skip path (e.g. ``migration-integration`` skips on a
   docs-only diff; ``contract-sync-gate`` skips on ``push``), matching the old
   ``success || skipped`` condition.

3. **Default-deny failure sweep.** Any *other* job in the run that is *present*,
   *completed*, and whose conclusion is not ``success``/``skipped`` fails the
   gate — UNLESS it is the poller itself or one of a small, explicit
   :data:`SOFT_ALLOWLIST` of jobs that already exist in ``ci.yml`` as non-gating
   (advisory / warn-only / not in ci-summary's ``needs`` / not a required
   context). This sweep is what makes the poller *stricter* than the old gate:
   the old ``tests-gate`` greens when ``test-parallel`` is ``skipped``, so a
   failure in ``detect-changes`` / ``plugin-env-service-completeness`` /
   ``compose-required-env-coverage`` / ``contract-path-preflight`` (which skip
   ``test-parallel``) used to slip through silently. The sweep catches them.

The strict + skippable gates together are the **completeness anchor**: requiring
them present+good proves the whole substantive matrix actually ran and passed,
which prevents a *false green* before late-created jobs (``detect-changes`` →
``test-parallel`` → ``tests-gate``) have even been instantiated. If a gate is
missing or still running, the verdict is PENDING (poll again). At the caller's
deadline, PENDING is converted to FAILURE (fail-closed): the required context
always reaches a terminal state.

4. **External context assertion (OMN-15496).** Checks 1-3 all read
   ``actions/runs/${RUN_ID}/jobs`` — *this* workflow run's job list. Any check
   produced by a **different workflow file** is structurally invisible to them,
   and ``omnibase_infra``'s ``dev`` requires exactly one context (``CI Summary``,
   ``strict=false``), so such checks were enforced by **neither** layer:
   59 distinct cross-workflow check-run names on a real merged PR head
   (#2567 / ``0fca3b5e``) versus 40 inside this run's suite.

   :data:`EXPECTED_EXTERNAL_CONTEXTS` closes that hole *without* re-fanning 59
   required contexts (which would discard the deliberate single-umbrella design
   of OMN-4497/OMN-14127 — and a context that does not report on every PR shape
   wedges the branch indefinitely). Each named context is resolved from the PR
   head's ``commits/{sha}/check-runs`` and must be **present**, **completed**,
   and conclude ``success``; missing or still-running is PENDING, which the
   caller's deadline converts to FAILURE. This is the presence assertion
   OMN-14456 AC4 asked for.

   *Why this was load-bearing:* PR #2555 merged 2026-07-30T04:25:09Z with
   ``CI Summary`` = **success** (all 53 in-run jobs green) while
   ``deploy-gate / deploy-gate`` = **failure** on the same head SHA. The
   required context was green because the failing check was in another run.

Exit codes: ``0`` success, ``1`` failure, ``2`` pending.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

# The poller's own job — excluded to avoid self-deadlock.
SELF_JOB_NAME = "CI Summary"

# Aggregate/leaf gates that the old needs-based ci-summary required STRICTLY
# (``== "success"``). Each is unconditional in ci.yml (no ``if:``) so it never
# legitimately skips on the gating events; a skip here fails closed. Names are
# the ``name:`` display strings the Actions jobs API returns (verified against
# ci.yml on 2026-07-07).
STRICT_GATE_JOBS: tuple[str, ...] = (
    "occ-preflight / eligibility",  # occ-preflight reusable gate
    "CI Tests Gate",  # tests-gate — aggregator over the split matrix
    "Lint",  # lint
    "ONEX Validators",  # onex-validation
    "Infra Node Handler Ownership",  # infra-node-handler-ownership
    "Migration Freeze Check",  # migration-freeze
    "Fingerprint Check",  # fingerprint-check
    "Demo Loop Gate",  # demo-loop-gate
    "Topic Enum Drift Check",  # topic-enum-drift
    "Topic Naming Lint",  # topic-naming-lint
    "Topic Drift Check",  # topic-drift-check
    "Arch Invariants (OMN-3343)",  # arch-invariants
    "Kafka Schema Handshake (OMN-3411)",  # schema-handshake
    "Writer-Migration Coupling Check",  # migration-required-check
    "Node Migration Declaration Check",  # node-migration-declaration-check (OMN-15717)
    "no-noncanonical-lifecycle-classes",  # OMN-14350 non-canonical lifecycle-class ratchet
    "Effect-Assertion Gate (RT-5)",  # OMN-14467 deploy-trigger fails closed on zero output
    "OCC Companion Merged Gate (OMN-15214)",  # occ-companion-merged — cited OCC evidence must be MERGED before product merge
    # OMN-16774: whole event chains driven through the REAL dispatch seam on the
    # in-memory bus (tests/integration/chains/). THIS LINE IS HALF THE
    # MECHANISM. The default-deny sweep below already fails CI Summary when the
    # job FAILS, but an unregistered job that is `skipped` or absent yields
    # SUCCESS — so without this entry, deleting the job (or letting it be
    # skipped) would silently retire the only per-PR proof that a chain still
    # terminalizes. The job is unconditional in ci.yml (no needs/if), so a skip
    # is anomalous and never a legitimate opt-out. Registered because OMN-16767
    # proved the failure mode is SILENT: the delegation chain was 100% dead for
    # weeks behind green CI, with every request going to the quarantine sink.
    "Event Chain Gate",  # event-chain-gate
    # OMN-16795: static cross-contract check that every declared subscribe_topic
    # has a contract publisher, PLUS allowlist hygiene (expired / malformed /
    # stale entries fail). THIS LINE IS HALF THE MECHANISM, same as the entry
    # above: the default-deny sweep already fails CI Summary when the job FAILS,
    # but an unregistered job that is `skipped` or absent yields SUCCESS — so
    # without this entry, deleting or skipping the job silently restores the
    # advisory-only state that let a 45-entry allowlist drift with lapsed
    # expiries for months (the checker shipped in OMN-7385 and was referenced by
    # NOTHING until this ticket). The job is unconditional in ci.yml (no
    # needs/if), so a skip is anomalous and never a legitimate opt-out.
    # Superseded by OMN-16783's flow-expectation ratchet when that lands.
    "Subscribe Wiring Health",  # subscribe-wiring-health
    # OMN-15378 AC3: scripts/deploy-agent's standalone pytest root. ci.yml's
    # `deploy-agent-tests` job CALLS .github/workflows/deploy-agent-tests.yml,
    # so the inner job surfaces as "<caller display name> / <inner job name>"
    # (same shape as "occ-preflight / eligibility"). Registering it here is what
    # makes those 201 tests GATE merge: while they lived in a separately-
    # triggered workflow this poller could not observe them at all (different
    # run_id), so a RED run left "CI Summary" — the sole required context on
    # dev — green.
    "Deploy Agent Tests (OMN-15378) / deploy-agent-tests",
    # OMN-15484: the Merge Hold Gate, fanned out from OMN-15483 (omnibase_infra
    # carries incident §C, #2560, and had zero coverage). THIS LINE IS THE
    # MECHANISM — not the job's existence in ci.yml. The default-deny sweep
    # below already catches a hold job that FAILS, but an unregistered job that
    # is `skipped` or `absent` yields CI Summary SUCCESS, so a held PR would be
    # required-green and the sweep would land it. Measured against this very
    # evaluator on omnimarket#1973: unregistered, `skipped` -> SUCCESS and
    # `absent` -> SUCCESS; registered, `skipped` -> FAILURE and `absent` ->
    # PENDING. The job is unconditional (no needs/if), so a skip is anomalous,
    # never a legitimate opt-out. Same "<caller display name> / <inner job
    # name>" shape as the two entries above; renaming either half makes this
    # gate permanently PENDING. Pinned by tests/ci/test_merge_hold_gate_omn15484.py.
    "merge-hold-gate / evaluate",
    # OMN-15538: every cross-repo pin must resolve to a commit reachable from a
    # protected branch of the target repo. THIS LINE IS THE MECHANISM, not the
    # job's presence in ci.yml — `CI Summary` is dev's sole required context, so
    # an unregistered job that fails still yields SUCCESS here and the PR lands.
    # The gate it replaces the absence of: on 2026-07-30 a `uses:` pin to a
    # deleted omnimarket branch head made ci.yml startup-fail for ~2.5h
    # (OMN-15536), and a pyproject rev pinned to an unlanded omnibase_core
    # branch head merged past a comment forbidding it — both accepted by
    # SHAPE-only 40-hex validators. The job is unconditional (`if: always()`),
    # so a skip is anomalous and correctly fails closed here.
    "Pin Reachability (OMN-15538)",
    # OMN-15361: one unconditional source+Docker gate executes the classification,
    # schema/RLS, role, adapter, one-database, and topology assertions together
    # with their seeded RED controls. Registering the plain job display name here
    # makes the source contract and rebuilt PostgreSQL 16 proofs part of the sole
    # required CI Summary context rather than a separately-triggered advisory run.
    "Application Database Domain Enforcement (OMN-15361)",
    # OMN-15604: a [tool.uv.sources] git-pinned rev must build the SAME src/
    # tree as the released tag its declared `pkg==X.Y.Z` version names, even
    # on a line carrying a `# raw-override-ok:` escape token (that token only
    # exempts the separate, pre-existing `Dep Provenance Gate` -- the
    # forbid-git-source rule -- never a content-lineage claim). The job is
    # unconditional (`if: always()`), so a skip is anomalous and correctly
    # fails closed here. Registered directly (not via EXPECTED_EXTERNAL_
    # CONTEXTS) because it is a job inside ci.yml's own run, observable
    # without the external-context admission rule's historical measurement.
    "Dep Provenance Lineage Gate (OMN-15604)",
    # OMN-16229: companion to OMN-16228, closing the other half of the
    # 2026-08-18 sqlparse/Trivy incident (OMN-16170) -- an expiring-ignore
    # policy for fix-unavailable CVEs in the Trivy image gate. THIS LINE IS
    # THE MECHANISM, not the job's presence in ci.yml: the job has no
    # job-level `if:` (unconditional, cheap, dependency-free), so it always
    # completes success/failure and a skip/absence here is anomalous --
    # correctly fails closed.
    "Trivyignore Expiry Check (OMN-16229)",
    # OMN-16228: born from the 2026-08-18 sqlparse/Trivy incident (OMN-16170)
    # -- shift lockfile CVE detection left to dependency-pin time (this job)
    # instead of image-build time (the Trivy gate, deep in the deploy
    # pipeline). THIS LINE IS THE MECHANISM, not the job's presence in
    # ci.yml: the job has no job-level `if:` (path relevance is decided
    # internally via step-level `if:` guards, see
    # scripts/ci/check_lockfile_cve.py's module docstring), so it always
    # completes success/failure and a skip/absence here is anomalous --
    # correctly fails closed.
    "Lockfile CVE Scan (OMN-16228)",
    # OMN-16516: structural (tomllib) fail-closed backstop for the
    # 2026-08-23 mirror-leak incident (OMN-16162) -- a committed uv.lock
    # resolving any package from a non-public registry/git/artifact host.
    # THIS LINE IS THE MECHANISM, not the job's presence in ci.yml: the job
    # has no job-level `if:` (unconditional, cheap, dependency-free), so it
    # always completes success/failure and a skip/absence here is anomalous
    # -- correctly fails closed.
    "Lockfile Registry Allowlist (OMN-16516)",
)

# Gates the old ci-summary accepted as ``success`` OR ``skipped``. Each carries
# a legitimate skip path in ci.yml (docs-only diff, or event-scoped ``if:``).
SKIPPABLE_GATE_JOBS: tuple[str, ...] = (
    "Migration Integration Test",  # migration-integration (skips on docs-only)
    "Integration Silent-Skip Guard (OMN-14172)",  # integration-guard (skips on docs-only)
    "Contract Compliance",  # compliance
    "Contract Compliance Check",  # contract-compliance
    "Contract Sync Gate (Wave C) [OMN-8915]",  # contract-sync-gate (skips on push)
)

# --------------------------------------------------------------------------- #
# OMN-16661: the docs-only skip tier.
# --------------------------------------------------------------------------- #
#
# Operator ruling: a Markdown / badge / README PR must not pay for the heavy
# code-verification suite, while the doc gates keep running. Measured on the
# real merged docs-only PR #2906 (head ``0c86fd00``, files = ``docker/README.md``
# + ``docs/**``): 172 check-runs, 157 non-skipped, **350 runner-minutes** — the
# heaviest docs-only PR cost in the registry.
#
# The heavy TEST matrix was already quiet: ``test-parallel`` / ``detect-changes``
# / ``CI Tests Gate`` have gated on ``needs.zone-filter.outputs.docs_only`` for a
# while (observed 0m on #2906). The gap this closes is the *validator* jobs,
# which all ran at full cost on that PR.
#
# WHY A MARKER JOB, not ``needs.zone-filter.outputs.docs_only``
# ------------------------------------------------------------
# omnibase_core's OMN-16625 pilot could gate its ``quality-gate`` aggregator by
# adding ``needs: [zone-filter]`` and reading the output directly. That is not
# available here: ``ci-summary`` is a NO-``needs`` poller *on purpose*
# (OMN-14127 — a ``needs``-gated job gets no check-run until its needs
# terminalize, which is exactly how the old gate went ABSENT under self-hosted
# fleet saturation and wedged PRs BLOCKED with 0 failing / 0 pending), and job
# OUTPUTS do not appear in the ``actions/runs/{run_id}/jobs`` payload this
# module reads. A JOB does appear — hence ``docs-only-marker``, whose own
# ``if:`` is ``always() && needs.zone-filter.outputs.docs_only == 'true'``.
#
# FAIL-CLOSED BY CONSTRUCTION, not by an added guard: the ONLY state that
# relaxes anything is a marker that actually RAN and concluded ``success``.
# Absent, ``in_progress``, ``skipped``, ``cancelled``, ``failure`` — every one
# of them means "not docs-only". On an ordinary code PR the marker's ``if:`` is
# false, so full strictness is the default, not an opt-in. The marker is not a
# caller-supplied flag and cannot be set by hand; its sole authority is the
# reusable zone-filter classifier, which requires EVERY changed path to
# classify ``EnumFileZone.DOCS``.
#
# The relaxation is PER-NAME, never a blanket ``|| skipped`` — the same policy
# ``tests-gate`` already applies per-upstream (OMN-15315). Every gate outside
# this tier must still be exactly ``success`` on a docs-only diff, which is what
# keeps ``Lint``, ``ONEX Validators``, the contract gates, the supply-chain
# gates and ``OCC Companion Merged Gate`` running — the half of the operator
# ruling that is not about saving minutes.
#
# TIER MEMBERSHIP RATIONALE: each entry is a Pydantic-round-trip, DB-schema, or
# effect-shape proof over ``src/`` + Docker. None can change verdict when only
# ``docs/**`` and ``*.md`` moved. Deliberately EXCLUDED despite being expensive:
# ``OCC Companion Merged Gate (OMN-15214)`` (10m) is an evidence-ordering gate,
# orthogonal to code content — a docs PR still cites OCC evidence that must be
# merged first.
DOCS_ONLY_MARKER_JOB = "Docs-Only Marker (OMN-16661)"

#
# NOTE ON MEMBERSHIP vs ci.yml GATING — these are two different sets, on
# purpose. This tuple only ever RELAXES a ``STRICT_GATE_JOBS`` member, so it
# lists exactly the strict gates that are docs_only-gated in ci.yml.
# ``Kafka Boundary Compat (OMN-3256)`` is ALSO docs_only-gated in ci.yml (~5
# min saved) but is deliberately absent here: it is not a ``STRICT_GATE_JOBS``
# entry, so there is nothing to relax — its ``skipped`` conclusion is already
# tolerated by the L3 default-deny sweep (``skipped`` ∈ GOOD_CONCLUSIONS).
# Adding it would be a no-op that misleads the reader into thinking the poller
# enforces it.
DOCS_ONLY_SKIPPABLE_GATE_JOBS: tuple[str, ...] = (
    "Application Database Domain Enforcement (OMN-15361)",  # ~7 min
    "Kafka Schema Handshake (OMN-3411)",  # ~4 min
    "Effect-Assertion Gate (RT-5)",  # ~2 min
    "Demo Loop Gate",  # ~2 min
)

# Every job the completeness anchor must observe present+good for SUCCESS.
GATE_JOBS: tuple[str, ...] = STRICT_GATE_JOBS + SKIPPABLE_GATE_JOBS

# Jobs that do NOT gate merge today (verified against ci.yml ci-summary ``needs``
# + the pass/fail condition, and against dev branch-protection required contexts
# on 2026-07-07). The default-deny sweep ignores these so it never newly-wedges
# a PR on a job that is already non-blocking. Keep this list SMALL and only add
# jobs that genuinely already exist in ci.yml as non-gating.
#
# Matching is prefix-aware (see :func:`_is_allowlisted`) so reusable-workflow
# callers — whose inner jobs surface as ``"<caller name> / <inner job>"`` — are
# covered by their caller entry (``zone-filter``, ``Runtime Boot Smoke
# (compose)``).
SOFT_ALLOWLIST: frozenset[str] = frozenset(
    {
        # In ci-summary ``needs`` but deliberately EXCLUDED from the pass/fail
        # condition (advisory / informational):
        "Test-Failure Ratchet Gate",  # advisory (OMN-13867)
        "Version Pin Compliance",  # in needs, never checked in the condition
        # Not in ci-summary ``needs`` and not a required context:
        "Runtime Boot Smoke (compose)",  # advisory (OMN-9120); reusable caller
        "Cross-Repo Migration Conflicts",  # migration-conflict-check; not required
        "Kafka Boundary Compat (OMN-3256)",  # advisory; carries xfail known-drift
        "AI-Slop Pattern Check (strict, PR diff)",  # aislop-sweep gates the tree
        # Structural path filter — reusable caller, excluded from the condition:
        "zone-filter",  # zone-filter (reusable) inner jobs surface prefixed
    }
)

# ---------------------------------------------------------------------------
# OMN-15496 — cross-workflow ("external") required contexts.
# ---------------------------------------------------------------------------
# Contexts produced by OTHER workflow files on the SAME head SHA. They are not
# in `dev`'s required_status_checks (which is exactly ["CI Summary"]) and are
# invisible to the run-scoped checks above, so before this tuple existed they
# blocked nothing.
#
# ADMISSION RULE — do not add a name here from a workflow file alone.
# A context is admitted only after measuring its *merge-time* report rate over
# the last N merged `dev` PRs: for each PR, the check-runs on its head SHA whose
# `started_at <= mergedAt` (post-merge runs are a retrospective artifact — on the
# first pass they produced three phantom "failures" for contexts that were green
# at merge). A context that does not report on every PR shape MUST NOT be listed:
# a permanently-absent entry burns the poll deadline and then fails closed, i.e.
# it wedges the branch. Every name below was measured 16/16 present over the 16
# `dev` PRs merged 2026-07-29T23:04Z → 2026-07-30T14:54Z (#2546…#2567).
#
# Replaying those 16 PRs' merge-time payloads through this resolver yields 15
# green and exactly one block — #2555, `deploy-gate / deploy-gate` = failure,
# which is the real defect this gate exists to catch. Slowest seeded context
# finished 24.9 min after `CI Summary` started, well inside the caller's 90 min
# poll deadline, so waiting on these cannot time the poller out.
# Fixture + regression: tests/ci/fixtures/omn15496_merge_time_external_check_runs.json.
#
# OMN-15737 (successor to OMN-13873, whose own DoD required this context be
# "required on infra dev/main branch protection" but never followed through):
# `Dep Provenance Gate` (dep-provenance-gate.yml) was separately re-measured
# 16/16 present, 16/16 green over the SAME #2546…#2567 window (job has no
# job-level `if:` — it always executes and reports, even when pyproject.toml is
# unchanged) and folded into the same fixture rows above.
EXPECTED_EXTERNAL_CONTEXTS: tuple[str, ...] = (
    "deploy-gate / deploy-gate",  # 16/16 present, 15/16 green (#2555 red AT MERGE)
    "verify / verify",  # Receipt Gate
    "call-reject-skip-token / scan / reject-skip-gate-token",  # CLAUDE.md rule 10 mechanism
    "main-target-guard",
    "non-dev-base-guard",
    "pr-title / check-title",
    "URL Authority Gate",
    "imperative-contract-guard / Imperative Contract Guard",
    "gate / CodeRabbit Thread Check",
    "Canonical Inference Gate",
    "Type Safety Validation",
    "Omni Standards Gate",
    "Duplication Sweep",
    "Stale TODO Gate",
    "dispatcher-route-coverage",
    "CodeQL",
    "required-check-skip-guard / check-skip-vectors",
    # OMN-15979: the "Integration Test Removal Gate" job (OMN-8732,
    # .github/workflows/integration-test-check.yml, job check-test-removal) hard-
    # blocks a PR that deletes a tests/integration/*.py file without a
    # replacement, and its own header says "No override mechanism" — but before
    # this entry it was invisible to both branch protection (dev requires only
    # `CI Summary`) and this poller, so a red run merged anyway. Live proof: PR
    # #2720 (head ce4e88f8) merged 2026-08-11T04:47:58Z with this context =
    # failure. Measured 16/16 present, 15/16 green over TWO independent 16-PR
    # windows: the original OMN-15496 seed window 2026-07-29T23:04Z ->
    # 2026-07-30T14:54Z (#2546...#2567, backfilled into
    # omn15496_merge_time_external_check_runs.json — 16/16 green there) and the
    # current window 2026-08-09T23:11Z -> 2026-08-11T04:47:58Z (#2705...#2720,
    # omn15979_merge_time_external_check_runs.json — 15/16 green). The one red,
    # #2720, is root-caused, not a repeat-flake pattern: its job (id
    # 93668844708) recorded ZERO steps (no "Set up job"/checkout/script rows,
    # unlike every green run's 6-step shape) — a self-hosted-runner dispatch
    # failure, and PR #2720's diff touched only
    # .github/workflows/build-and-push-runtime.yml (zero tests/integration
    # files), so the gate's own substantive check was never at risk of a
    # legitimate red. Fixture + regression:
    # tests/ci/fixtures/omn15979_merge_time_external_check_runs.json.
    "Integration Test Removal Gate",
    "Dep Provenance Gate",  # OMN-15737: 16/16 present, 16/16 green (#2546-#2567 AND #2646-#2669)
)

# Contexts that were MEASURED and deliberately NOT enforced. Recorded as data —
# not silently omitted — so the exclusion is auditable and has to be re-argued
# with numbers rather than rediscovered. Pinned by test_ci_summary_gate.py.
MEASURED_NOT_ENFORCED_CONTEXTS: dict[str, str] = {
    "Enforce clean + promoted build source": (
        "1/16 present — path-filtered; requiring it would wedge every PR that "
        "does not touch its paths (the exact never-reports failure mode)."
    ),
    "occ-companion-effect / Publish occ-companion-effect command": (
        "16/16 present but only 10/16 green — a flaky publisher EFFECT, not a "
        "validator. The substantive requirement it stands in for is already "
        "enforced in-run by the STRICT gate 'OCC Companion Merged Gate "
        "(OMN-15214)'."
    ),
    "Hostile Review Gate": (
        "16/16 present, 14/16 green — an adversarial-judgment gate. A 12.5% red "
        "rate needs per-red root-cause before it may block merges; admitting it "
        "blind would convert review opinion into a merge outage."
    ),
    "occ-preflight / eligibility": (
        "Already a STRICT_GATE_JOBS entry, and the ONE name observed both inside "
        "and outside this run's check suite (duplicate producers: ci.yml and "
        "hostile-reviewer.yml). Asserting it on both surfaces would double-count "
        "an ambiguous name — see OMN-15112."
    ),
}

# OMN-15532 — contexts whose PRODUCER structurally does not report for a given
# PR author, so "absent" carries no information and must not burn the poll
# deadline. This is an *applicability* rule, not a bypass: the context stays
# fail-closed for every author not named here.
#
# ADMISSION RULE — an entry is justified only by a producer-side condition that
# makes the check-run impossible to create, quoted with the workflow file and
# the live readback that shows it absent. "It was red and I wanted it green" is
# never a reason. Keys must be members of EXPECTED_EXTERNAL_CONTEXTS and actors
# must be concrete logins (no wildcards) — both pinned by tests.
ACTOR_CONDITIONAL_CONTEXTS: dict[str, tuple[str, ...]] = {
    # .github/workflows/cr-thread-gate-caller.yml gates the `gate` job on
    # `(github.event_name == 'pull_request' && github.actor != 'dependabot[bot]')`.
    # The context name is the `caller-job / reusable-job` form, so when the
    # caller job is skipped the reusable's inner job never materialises and NO
    # check-run is created — the context is absent, not `skipped`.
    #
    # Live readback 2026-07-30, infra dev Dependabot batch:
    #   #2522 2cdf352d actor=dependabot[bot] CR-gate run conclusion=skipped -> ABSENT
    #   #2521 841c292f actor=dependabot[bot] CR-gate run conclusion=skipped -> ABSENT
    #   #2520 feb6627b actor=jonahgabriel    -> present, success
    #   #2519 08e356cc actor=jonahgabriel    -> present, success
    #   #2518 e2d38605 actor=jonahgabriel    -> present, success
    # All five are `pull_request`, run_attempt=1: the actor is the discriminator.
    #
    # The OMN-15496 seed measured this context 16/16 present over #2546…#2567 —
    # a window containing NO Dependabot PR, which is exactly how a 16/16 context
    # can still be absent in production.
    #
    # Fixed consumer-side, not producer-side: OMN-10276 removed this actor skip
    # on omnimemory, but omnimemory calls a LOCAL reusable and passes no secrets,
    # whereas this caller invokes omniclaude's reusable with `secrets:
    # CROSS_REPO_PAT`. Dependabot `pull_request` runs do not receive regular repo
    # secrets, so dropping the skip here risks trading an absent-wedge for a
    # red-wedge. See OMN-15532.
    "gate / CodeRabbit Thread Check": ("dependabot[bot]",),
}

# Conclusions that count as "provably passed".
GOOD_CONCLUSIONS: frozenset[str] = frozenset({"success", "skipped"})

# External contexts are held to the STRICT bar: `skipped` fails closed. Every
# name above was measured `success` on all 16 sampled PRs (never skipped), so
# this costs nothing today and closes the skip-vector fail-open that OMN-15057 /
# OMN-14854 exist to prevent.
EXTERNAL_GOOD_CONCLUSIONS: frozenset[str] = frozenset({"success"})

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_PENDING = 2


@dataclass(frozen=True)
class JobState:
    """The latest-attempt state of a single workflow job."""

    name: str
    status: str  # queued | in_progress | completed | waiting | ...
    conclusion: str | None  # success | failure | cancelled | skipped | timed_out | None
    run_attempt: int


def _state_severity(job: JobState) -> int:
    """Rank same-attempt duplicate jobs by the most blocking state."""

    if job.status != "completed":
        return 2
    if job.conclusion not in GOOD_CONCLUSIONS:
        return 3
    return 1


def dedup_latest(
    jobs: list[dict[str, object]],
    *,
    run_attempt: int | None = None,
) -> dict[str, JobState]:
    """Collapse the raw ``/runs/{id}/jobs`` array to one entry per job name.

    When ``run_attempt`` is provided, only rows from that workflow attempt are
    considered. This prevents stale failed/cancelled rows from an earlier
    attempt from becoming authoritative for a current rerun. Within the same
    attempt, duplicate display names keep the most blocking state so a failed
    matrix leg cannot be hidden by a later same-name success.
    """

    latest: dict[str, JobState] = {}
    for raw in jobs:
        name = str(raw.get("name") or "")
        if not name:
            continue
        try:
            attempt = int(str(raw.get("run_attempt") or 1))
        except (TypeError, ValueError):
            attempt = 1
        if run_attempt is not None and attempt != run_attempt:
            continue
        prev = latest.get(name)
        if prev is not None and attempt < prev.run_attempt:
            continue
        conclusion = raw.get("conclusion")
        current = JobState(
            name=name,
            status=str(raw.get("status") or ""),
            conclusion=None if conclusion is None else str(conclusion),
            run_attempt=attempt,
        )
        if (
            prev is not None
            and attempt == prev.run_attempt
            and _state_severity(current) < _state_severity(prev)
        ):
            continue
        latest[name] = current
    return latest


def latest_check_run_by_name(
    check_runs: list[dict[str, object]],
) -> dict[str, JobState]:
    """Collapse ``commits/{sha}/check-runs`` to one entry per context name.

    Resolution is **latest wins** by ``(started_at, id)`` — deliberately the same
    rule GitHub itself applies when deciding a required status check from several
    same-named check-runs on one SHA.

    A stricter "most-blocking across all same-named runs" rule was measured and
    **rejected**: replayed over the 16 sampled merged PRs it blocks 6, of which 5
    are transient-red-then-rerun-green. Because check-runs accumulate on a SHA
    forever, most-blocking makes any transient red permanent and removes re-run
    as a recovery path — it manufactures merge outages instead of catching
    defects. Latest-wins blocks 1/16, and that one is a real red at merge.

    Known bounded residual: when two workflow files emit the same context name, a
    red from the earlier producer followed by a green from the later one resolves
    green. That ANY-vs-ALL ambiguity is tracked in OMN-15112 and is why
    ``occ-preflight / eligibility`` — the one name observed on both sides — is
    excluded here (see :data:`MEASURED_NOT_ENFORCED_CONTEXTS`).
    """

    latest: dict[str, JobState] = {}
    ordering: dict[str, tuple[str, int]] = {}
    for raw in check_runs:
        name = str(raw.get("name") or "")
        if not name:
            continue
        try:
            run_id = int(str(raw.get("id") or 0))
        except (TypeError, ValueError):
            run_id = 0
        key = (str(raw.get("started_at") or ""), run_id)
        if name in ordering and key <= ordering[name]:
            continue
        conclusion = raw.get("conclusion")
        ordering[name] = key
        latest[name] = JobState(
            name=name,
            status=str(raw.get("status") or ""),
            conclusion=None if conclusion is None else str(conclusion),
            run_attempt=1,
        )
    return latest


def applicable_external_contexts(
    expected: tuple[str, ...],
    pr_author: str | None,
) -> tuple[str, ...]:
    """Drop contexts whose producer cannot report for ``pr_author`` (OMN-15532).

    Order preserved. An unknown/empty ``pr_author`` drops NOTHING — the fail-
    closed default — so a missing ``--pr-author`` argument enforces the full set
    rather than silently exempting it.
    """

    if not pr_author:
        return expected
    return tuple(
        context
        for context in expected
        if pr_author not in ACTOR_CONDITIONAL_CONTEXTS.get(context, ())
    )


def evaluate_external_contexts(
    check_runs: list[dict[str, object]] | None,
    expected: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    """Return ``(failures, missing_or_pending)`` for the declared external contexts.

    ``check_runs is None`` means the caller could not fetch the head SHA's
    check-runs. That is treated as **every** expected context being unobserved —
    PENDING, never success — so a transient API failure retries and a permanent
    one fails closed at the deadline. It must never read as green.
    """

    if not expected:
        return [], []
    latest = latest_check_run_by_name(check_runs or [])
    failures: list[str] = []
    unresolved: list[str] = []
    for context in expected:
        state = latest.get(context)
        if state is None or state.status != "completed":
            unresolved.append(context)
        elif state.conclusion not in EXTERNAL_GOOD_CONCLUSIONS:
            failures.append(context)
    return sorted(failures), sorted(unresolved)


def _is_allowlisted(name: str, allowlist: frozenset[str]) -> bool:
    """Prefix-aware allowlist check.

    A reusable-workflow caller's inner jobs surface in the jobs API as
    ``"<caller display name> / <inner job name>"``; matching the caller segment
    lets a single allowlist entry cover all of its inner jobs.
    """

    if name in allowlist:
        return True
    caller = name.split(" / ", 1)[0]
    return caller in allowlist


def evaluate(
    jobs: list[dict[str, object]],
    *,
    run_attempt: int | None = None,
    self_name: str = SELF_JOB_NAME,
    strict_gates: tuple[str, ...] = STRICT_GATE_JOBS,
    skippable_gates: tuple[str, ...] = SKIPPABLE_GATE_JOBS,
    allowlist: frozenset[str] = SOFT_ALLOWLIST,
    check_runs: list[dict[str, object]] | None = None,
    external_contexts: tuple[str, ...] = (),
    pr_author: str | None = None,
    docs_only_marker: str = DOCS_ONLY_MARKER_JOB,
    docs_only_gates: tuple[str, ...] = DOCS_ONLY_SKIPPABLE_GATE_JOBS,
) -> tuple[int, str]:
    """Return ``(exit_code, human_report)`` for the current job snapshot.

    ``external_contexts`` defaults to empty (assert nothing) so non-PR callers —
    ``merge_group`` / ``workflow_dispatch``, where no PR-scoped context set
    exists — are not wedged. The CLI supplies
    :data:`EXPECTED_EXTERNAL_CONTEXTS` and its ``--event-name`` defaults to
    ``pull_request``, so a *forgotten* argument enforces rather than skips.

    ``pr_author`` drops only the contexts that :data:`ACTOR_CONDITIONAL_CONTEXTS`
    marks unreportable for that author (OMN-15532). ``None`` drops nothing.
    """

    external_contexts = applicable_external_contexts(external_contexts, pr_author)
    latest = dedup_latest(jobs, run_attempt=run_attempt)
    gate_names = frozenset(strict_gates) | frozenset(skippable_gates)

    # OMN-16661: derive docs_only from the in-run marker job, never from a
    # caller-supplied argument. ONLY a marker that ran and concluded success
    # relaxes anything — absent / in_progress / skipped / cancelled / failure
    # all leave every gate strict. See the DOCS_ONLY_MARKER_JOB block above for
    # why the bit travels as a job rather than as `needs.<job>.outputs`.
    marker_state = latest.get(docs_only_marker)
    docs_only = (
        marker_state is not None
        and marker_state.status == "completed"
        and marker_state.conclusion == "success"
    )
    # Relaxed ⊆ strict_gates: a name that is not strict cannot be "relaxed" into
    # existence, so a tier entry dropped from STRICT_GATE_JOBS degrades to a
    # no-op here instead of silently becoming permanently skippable.
    relaxed = (
        frozenset(docs_only_gates) & frozenset(strict_gates)
        if docs_only
        else frozenset()
    )

    # (1) Strict aggregate gates: present + completed + conclusion == success.
    #     Members of `relaxed` widen to success/skipped for THIS run only; a
    #     `failure`/`cancelled` conclusion is never admitted, docs-only or not.
    strict_failures = sorted(
        g
        for g in strict_gates
        if (
            (st := latest.get(g)) is not None
            and st.status == "completed"
            and (
                st.conclusion not in GOOD_CONCLUSIONS
                if g in relaxed
                else st.conclusion != "success"
            )
        )
    )

    # (2) Skippable aggregate gates: present + completed + success/skipped.
    skippable_failures = sorted(
        g
        for g in skippable_gates
        if (
            (st := latest.get(g)) is not None
            and st.status == "completed"
            and st.conclusion not in GOOD_CONCLUSIONS
        )
    )

    # (3) Default-deny sweep over every OTHER present+completed job.
    sweep_failures = sorted(
        j.name
        for name, j in latest.items()
        if name != self_name
        and name not in gate_names
        and not _is_allowlisted(name, allowlist)
        and j.status == "completed"
        and j.conclusion not in GOOD_CONCLUSIONS
    )

    # Completeness anchor: every gate must be present AND completed.
    gate_missing_or_pending = [
        g
        for g in (*strict_gates, *skippable_gates)
        if (latest.get(g) is None or latest[g].status != "completed")
    ]

    # (4) OMN-15496 external contexts: cross-workflow checks on the PR head.
    external_failures, external_unresolved = evaluate_external_contexts(
        check_runs, external_contexts
    )

    all_failures = (
        strict_failures + skippable_failures + sweep_failures + external_failures
    )
    all_unresolved = gate_missing_or_pending + external_unresolved

    def _verdict(label: str) -> str:
        return _report(
            label,
            latest,
            strict_gates,
            skippable_gates,
            strict_failures,
            skippable_failures,
            sweep_failures,
            gate_missing_or_pending,
            external_contexts,
            external_failures,
            external_unresolved,
            docs_only=docs_only,
            relaxed=relaxed,
        )

    if all_failures:
        return EXIT_FAILURE, _verdict("FAILURE")
    if all_unresolved:
        return EXIT_PENDING, _verdict("PENDING")
    return EXIT_SUCCESS, _verdict("SUCCESS")


def _report(
    verdict: str,
    latest: dict[str, JobState],
    strict_gates: tuple[str, ...],
    skippable_gates: tuple[str, ...],
    strict_failures: list[str],
    skippable_failures: list[str],
    sweep_failures: list[str],
    gate_missing_or_pending: list[str],
    external_contexts: tuple[str, ...] = (),
    external_failures: list[str] | None = None,
    external_unresolved: list[str] | None = None,
    *,
    docs_only: bool = False,
    relaxed: frozenset[str] = frozenset(),
) -> str:
    lines = [f"CI Summary verdict: {verdict}", f"  jobs observed: {len(latest)}"]
    # OMN-16661: make the relaxation visible in the job summary. A reviewer must
    # be able to read off WHY a strict gate was allowed to skip, and see the
    # marker state that authorised it — silent relaxation is how a skip tier
    # turns into an unnoticed bypass.
    marker_state = latest.get(DOCS_ONLY_MARKER_JOB)
    lines.append(
        "  docs-only marker: "
        + (
            "<absent>"
            if marker_state is None
            else f"{marker_state.status}/{marker_state.conclusion}"
        )
        + f" -> docs_only={str(docs_only).lower()}"
    )
    if relaxed:
        lines.append(
            "  docs-only skip tier ACTIVE (success|skipped accepted for): "
            + ", ".join(sorted(relaxed))
        )
    lines.append("  strict gates:")
    for g in strict_gates:
        st = latest.get(g)
        tier = "  [docs-only tier]" if g in relaxed else ""
        lines.append(
            f"    - {g}: <absent>{tier}"
            if st is None
            else f"    - {g}: {st.status}/{st.conclusion}{tier}"
        )
    lines.append("  skippable gates:")
    for g in skippable_gates:
        st = latest.get(g)
        lines.append(
            f"    - {g}: <absent>"
            if st is None
            else f"    - {g}: {st.status}/{st.conclusion}"
        )
    if strict_failures:
        lines.append(f"  strict-gate failures: {', '.join(strict_failures)}")
    if skippable_failures:
        lines.append(f"  skippable-gate failures: {', '.join(skippable_failures)}")
    if sweep_failures:
        lines.append(f"  default-deny sweep failures: {', '.join(sweep_failures)}")
    if gate_missing_or_pending:
        lines.append(f"  gates missing/pending: {', '.join(gate_missing_or_pending)}")
    if external_contexts:
        lines.append(f"  external contexts asserted: {len(external_contexts)}")
        if external_failures:
            lines.append(f"  external-context failures: {', '.join(external_failures)}")
        if external_unresolved:
            lines.append(
                f"  external contexts missing/pending: {', '.join(external_unresolved)}"
            )
    return "\n".join(lines)


def _load_jobs(path: str | None) -> list[dict[str, object]]:
    if path is None or path == "-":
        raw = sys.stdin.read()
    else:
        with open(path, encoding="utf-8") as handle:
            raw = handle.read()
    data = json.loads(raw)
    # Accept either the raw endpoint object ({"jobs": [...]}) or a bare array.
    if isinstance(data, dict):
        jobs = data.get("jobs", [])
    else:
        jobs = data
    if not isinstance(jobs, list):
        raise ValueError("jobs payload must be a list or an object with a 'jobs' array")
    return jobs


def _load_check_runs(path: str | None) -> list[dict[str, object]] | None:
    """Load ``commits/{sha}/check-runs``; return ``None`` when unavailable.

    ``None`` is the fail-closed signal: :func:`evaluate_external_contexts` reads
    it as "no context observed" → PENDING → FAILURE at the caller's deadline. A
    missing, empty, or malformed payload must never green the gate, so every
    failure path here returns ``None`` rather than an empty list.
    """

    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            raw = handle.read()
    except OSError:
        return None
    if not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        data = data.get("check_runs", [])
    if not isinstance(data, list):
        return None
    return [row for row in data if isinstance(row, dict)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs-file",
        default="-",
        help="Path to the GitHub Actions jobs JSON (default: stdin). Accepts the "
        "raw endpoint object or a bare array of job objects.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print the verdict report and exit 0 regardless (diagnostics only).",
    )
    parser.add_argument(
        "--run-attempt",
        type=int,
        default=None,
        help="Evaluate only rows for this GitHub Actions run_attempt.",
    )
    parser.add_argument(
        "--check-runs-file",
        default=None,
        help="Path to the PR head SHA's commits/{sha}/check-runs JSON, used to "
        "assert EXPECTED_EXTERNAL_CONTEXTS (OMN-15496). A missing/unreadable "
        "file is PENDING, never success.",
    )
    parser.add_argument(
        "--event-name",
        default="pull_request",
        help="GitHub event name. External contexts are asserted on "
        "'pull_request' only — merge_group/workflow_dispatch have no PR-scoped "
        "context set. Defaults to 'pull_request' so a FORGOTTEN argument "
        "enforces rather than silently skips.",
    )
    parser.add_argument(
        "--pr-author",
        default=None,
        help="Login of the PR author. Drops ONLY the ACTOR_CONDITIONAL_CONTEXTS "
        "entries that this author's PRs structurally cannot produce (OMN-15532). "
        "Omitted/empty drops nothing, so a forgotten argument enforces the full "
        "set rather than exempting it.",
    )
    args = parser.parse_args(argv)

    jobs = _load_jobs(args.jobs_file)
    external_contexts = (
        EXPECTED_EXTERNAL_CONTEXTS if args.event_name == "pull_request" else ()
    )
    code, report = evaluate(
        jobs,
        run_attempt=args.run_attempt,
        check_runs=_load_check_runs(args.check_runs_file),
        external_contexts=external_contexts,
        pr_author=args.pr_author,
    )
    print(report)
    if args.report_only:
        return EXIT_SUCCESS
    return code


if __name__ == "__main__":
    raise SystemExit(main())
