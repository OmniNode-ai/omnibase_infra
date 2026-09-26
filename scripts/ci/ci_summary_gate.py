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

5. **Default-deny EXTERNAL sweep (OMN-18960).** Check 4 is a whitelist
   LOOKUP: it walks the expected tuple and asks the head for each name. It
   never walks the head's check-run list the other way, so a check-run whose
   name is not in that tuple was read by nothing — it could conclude
   ``failure`` and this module would not see it. Measured over the 16 dev PRs
   merged 2026-09-20T12:56:06Z → 2026-09-21T00:06:35Z: **40-55 such names per
   head, 70 distinct across the window, against a tuple of 26.**

   This check takes the head's check-runs, subtracts this run's own job names
   (checks 1-3 own those), the expected tuple (check 4 owns those), this
   poller's own context, and rows attributable to a NON-pull-request event,
   and fails on any remainder that concluded a refusal. A name is admitted
   only through :data:`EXTERNAL_SWEEP_EXCLUSIONS`, whose entries each carry a
   reason, an owning ticket, a date and an ABSOLUTE expiry — all four
   validated, a malformed entry failing the gate, and an expired entry
   silently ceasing to exclude so the gate re-arms by itself.

   Its bar is narrower than check 4's and deliberately so: ``skipped`` and
   ``neutral`` are NOT failures here, because 9 of those 70 names are never
   green in the window by design. The strict absent/skipped/neutral bar is
   bought by REGISTERING a name in :data:`EXPECTED_EXTERNAL_CONTEXTS`. This
   check's contract is: **nothing red slipped past unseen.**

Exit codes: ``0`` success, ``1`` failure, ``2`` pending.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime

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
    # OMN-17320: salted-digest denylist of customer identifiers that must never
    # re-enter this PUBLIC repo. THIS LINE IS HALF THE MECHANISM, same as the two
    # entries above: the default-deny sweep already fails CI Summary when the job
    # FAILS, but an unregistered job that is `skipped` or absent yields SUCCESS.
    # Registered because the failure mode is PROVEN, not hypothetical -- OMN-17288
    # scrubbed a live tenant slug from five files here, and three hours later an
    # unrelated lane reintroduced it in omnimarket with every enforced gate green,
    # on the very PR whose acceptance criterion was its absence. The job is
    # unconditional in ci.yml (`if: always()`), so a skip is anomalous and never a
    # legitimate opt-out.
    "Exposed Identifier Gate (OMN-17320)",  # exposed-identifier-gate
    # OMN-18247: every artifact declared in config/ci_evidence_policy.yaml is
    # asserted present and non-empty AT its uploader. THIS LINE IS HALF THE
    # MECHANISM, on the identical reasoning as the entries above: the
    # default-deny sweep fails CI Summary when this job FAILS, but an
    # unregistered job that is `skipped` or ABSENT yields SUCCESS. The failure
    # mode this gate exists for is itself silent -- the lab-load probe produced
    # a zero-byte artifact on 10 of its first 12 runs and never went red -- so a
    # gate that could be silently deleted would reproduce the exact shape it
    # closes. The job is unconditional in ci.yml (no needs/if), so a skip is
    # anomalous and never a legitimate opt-out.
    "CI Evidence Policy (OMN-18247)",  # ci-evidence-policy
    # OMN-18776: the skip-count baseline ratchet (epic OMN-18775). THIS LINE IS
    # HALF THE MECHANISM, on the identical reasoning as the entries above: the
    # default-deny sweep below already fails CI Summary when this job FAILS, but
    # an unregistered job that is `skipped` or ABSENT yields SUCCESS. The failure
    # mode it closes is itself silent and measured -- five per-repo skip counts
    # byte-identical across three consecutive runs each, and 91 PostgreSQL-16
    # cases collected-and-skipped on every full-matrix run for 49 days with no
    # surface registering it -- so a gate that could be silently deleted would
    # reproduce the exact shape it exists to refuse. The job is unconditional in
    # ci.yml (`if: always()`), so a skip is anomalous and never a legitimate
    # opt-out. Pinned by tests/ci/test_skip_count_ratchet_omn18776.py.
    "Skip Count Ratchet (OMN-18776)",  # skip-count-ratchet
    # OMN-18031: the per-run runner routing decision. THIS LINE IS HALF THE
    # MECHANISM, on the identical reasoning as the three entries above: this
    # repo requires exactly one context (`CI Summary`), the default-deny sweep
    # below already fails when a registered job FAILS, but an unregistered job
    # that is `skipped` or ABSENT yields SUCCESS. Without this entry, deleting
    # the `route` job from ci.yml would silently retire per-run routing on a
    # fully green run — and because routing is deliberately INERT while the
    # trusted seam reads '["ubuntu-latest"]' (OMN-16682), nothing about job
    # PLACEMENT would change to reveal it. That is the exact silent-retirement
    # shape this tuple exists for, and it is worse here than elsewhere: the
    # only observable difference between "routing works and chose hosted" and
    # "routing is gone" is a decision artifact nobody is required to read.
    # The job is unconditional in ci.yml (no needs/if), so a skip is anomalous
    # and never a legitimate opt-out. Same "<caller display name> / <inner job
    # name>" shape as the two `uses:` entries below; renaming either half
    # breaks this registration.
    "Runner Route (OMN-18031) / route",
    # OMN-15378 AC3: scripts/deploy-agent's standalone pytest root. ci.yml's
    # `deploy-agent-tests` job CALLS .github/workflows/deploy-agent-tests.yml,
    # so the inner job surfaces as "<caller display name> / <inner job name>"
    # (same shape as "occ-preflight / eligibility"). Registering it here is what
    # makes those 201 tests GATE merge: while they lived in a separately-
    # triggered workflow this poller could not observe them at all (different
    # run_id), so a RED run left "CI Summary" — the sole required context on
    # dev — green.
    "Deploy Agent Tests (OMN-15378) / deploy-agent-tests",
    # OMN-18926: the ONLY leg anywhere that builds `omnidash_analytics` from EMPTY and
    # runs the real scripts/run-forward-migrations.sh over the real corpus. ci.yml's
    # `legacy-rds-fixture-proof` job CALLS
    # .github/workflows/legacy-rds-fixture-proof.yml, so the inner job surfaces as
    # "<caller display name> / <inner job name>" -- same shape as the two entries above.
    # THIS LINE IS THE MECHANISM. While that proof was a separately-triggered workflow
    # it had its own run_id and this poller could not observe it at all, so a RED
    # from-empty build left "CI Summary" -- the sole required context on dev -- green.
    # Measured: the corpus could not build `omnidash_analytics` from empty at all (107
    # migrations applied, then `division by zero` on a precondition probe, because
    # nothing deliverable created the `omninode_internal` schema), and that shipped and
    # stayed shipped with every gate green. The job is unconditional in ci.yml (no
    # needs/if), so a skip is anomalous and never a legitimate opt-out.
    "Sanitized Legacy RDS Fixture Proof / PostgreSQL 16 Fresh + Legacy Fixture",
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
    # OMN-18012: the customer-path boundary chain (tests/integration/
    # customer_path/). THIS LINE IS HALF THE MECHANISM, same as the two
    # entries above -- the default-deny sweep fails CI Summary when the job
    # FAILS, but an unregistered job that is `skipped` or absent yields
    # SUCCESS, and a silently-absent boundary test is exactly the shape of
    # the 2026-09-06 escapes it exists to catch. The job runs the ONLY
    # per-PR proof in this repo that the terminal readback survives a
    # retention-truncated partition and that the bus client can actually
    # authenticate to an auth-required listener; the sharded test matrix
    # excludes it by marker (`-m "not kafka"`).
    "Customer Path Boundary (OMN-18012)",  # customer-path-boundary
    # OMN-19412: the lab probe-window file's drift check. THIS LINE IS THE
    # MECHANISM, the same as the lockfile entries above: the job has no `if:`,
    # so it always completes, and registering it here is what makes a skip or
    # an absence fail CI Summary rather than read green.
    "Lab Probe Windows (OMN-19412)",  # lab-probe-windows
    # OMN-19677: shrink-only debt baselines via the shared omniclaude reusable.
    # A reusable caller reports as '<caller display name> / <inner job>'; these
    # callers are unconditional, so a skip is anomalous.
    "Noncanonical Class Allowlist One-way (OMN-19677) / anti-growth-baseline",
    "Topic Naming Baseline One-way (OMN-19677) / anti-growth-baseline",
    "Validator Requirements Baseline One-way (OMN-19677) / anti-growth-baseline",
    "Runtime Profiles Allowlist One-way (OMN-19677) / anti-growth-baseline",
    "Skip Count Baseline One-way (OMN-19677) / anti-growth-baseline",
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
# + ``docs/**`` — the paths as they existed then; both trees were migrated out of
# this repo by OMN-16607): 172 check-runs, 157 non-skipped, **350 runner-minutes**
# — the heaviest docs-only PR cost in the registry.
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
        # OMN-14909: report-only telemetry needs ci-summary, so it is never
        # completed while the poller runs; allowlisted so a re-run of CI Summary
        # can never read a prior attempt's red telemetry as a failure.
        "CI cascade reason-graph (report-only)",
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
    "Canonical Inference Gate",
    "Type Safety Validation",
    "Omni Standards Gate",
    "Duplication Sweep",
    "Stale TODO Gate",
    "dispatcher-route-coverage",
    # OMN-18938. Registered so its ABSENCE is a failure, not a pass. This gate
    # ran as an unregistered, path-filtered job through the fleet-wide outage of
    # 2026-09-20 that it exists to catch: omnibase_infra#3882 made two dispatch
    # keywords REQUIRED, the deployed consumer passed neither, every delegation
    # on the dev lane terminalised provider_error, and the module reported 3
    # passed 0 failed. The default-deny sweep below would now catch it RED, but
    # a job that never runs is never red -- which is why presence is asserted
    # here and the job moved to an unfiltered trigger in the same change.
    "consumer-kwarg-parity",
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
    # OMN-18796 (epic OMN-18775): the no-new-advisory-job gate, called from
    # .github/workflows/advisory-job-gate.yml against the omniclaude reusable
    # pinned by commit. On THIS repository `dev` requires exactly one context
    # ("CI Summary", the OMN-4497 single-umbrella design), so this tuple is the
    # whole external enforcement surface and an entry here is the only thing
    # that gives the gate merge-blocking force. The caller carries no `paths:`
    # and no `branches:` filter, so it reports on every pull-request shape and
    # cannot be legitimately absent -- the admission condition this tuple takes.
    # The census counted 28 advisory settings and 43 pull-request-reachable
    # verification jobs in this repository, the largest share on the fleet.
    "advisory-job-gate / advisory-job-gate",
    # OMN-16878 (OMN-16876 census items 1-2). Both ran on every infra PR and
    # could not block a merge. That is a sharper failure here than elsewhere:
    # `dev` requires exactly ONE context ("CI Summary", the OMN-4497
    # single-umbrella design), so this tuple IS the whole external enforcement
    # surface — a context missing from it has no second surface to fall back
    # on and no branch-protection signal that it is missing.
    #
    # Both entries close a Done ticket that was not true on live state:
    #   receipt-honesty     — OMN-13328's OCC contract asserted omnibase_infra,
    #                         omniclaude and omnimarket "were already flipped";
    #                         live readback on 2026-08-28 showed none of the
    #                         three was enforced on any surface.
    #   contract-validation — OMN-13326 claimed "REQUIRED across all 6 repos";
    #                         it is genuinely required on omnibase_core,
    #                         omnibase_compat, omnibase_spi, omniintelligence
    #                         and omnidash, and was wired on neither
    #                         omnibase_infra nor omniclaude.
    #
    # Admission: measured over the 16 most recent merged `dev` PR heads,
    # #2955..#2970 (2026-08-28T07:11:14Z -> 2026-08-28T17:18:06Z). Both are
    # 16/16 present and 16/16 green, zero reds — so neither is admitted on a
    # red-rate that would turn a flaky producer into a merge outage, and no
    # freeze-baseline ratchet is indicated.
    #
    # 16/16 green is also the vacuous-pass shape (OMN-16876 finding 5), so each
    # was separately proven able to FAIL on real input before admission
    # (OCC#7433, dod-nonvacuity-negative-tests):
    #   receipt-honesty     — a gamed receipt (verifier == runner, echo probe)
    #                         exits 1; a real committed receipt exits 0.
    #   contract-validation — a schema-invalid contract exits 1;
    #                         contracts/OMN-10041.yaml exits 0.
    #
    # Both producers declare `pull_request` AND `merge_group`, so requiring them
    # cannot wedge a queue SHA, and BOTH jobs carry no `needs:` and no job-level
    # `if:` at all — there is no upstream whose failure could skip them, so the
    # OMN-15057 vector-5 skip-as-pass hazard that forced an `if: always()` fix on
    # omniclaude's three producers does not arise here. Belt and braces anyway:
    # that hazard is a BRANCH-PROTECTION property (a skipped check satisfies it),
    # while this layer's EXTERNAL_GOOD_CONCLUSIONS admits only "success", so a
    # skipped producer fails CI Summary closed rather than satisfying it.
    "receipt-honesty",  # receipt-honesty.yml
    "contract-validation",  # contract-validation.yml
    # OMN-17199 — the reader end of a bus_backed exposure.
    # exposure-reader-coverage.yml, added in the SAME PR as the validator it
    # runs (CLAUDE.md Operating Rule 5: a check that is not a merge condition is
    # advisory and gets ignored). It asserts that every `projection_api`
    # exposure declaring `bus_backed: true` has a declared reader — an omnidash
    # component, a shipped layout entry, or a reasoned `consumers: none`.
    #
    # ADMISSION IS BY CONSTRUCTION, NOT BY MEASUREMENT, and that difference is
    # stated rather than glossed. Every other entry above was admitted on an
    # N-of-16 present/green record over merged `dev` heads. A gate that does not
    # exist yet has no such record and cannot acquire one before it is wired —
    # requiring 16 green merges first would mean the gate is unenforced during
    # exactly the window it was filed to close, which is the OMN-15864-sibling
    # failure this ticket exists to avoid repeating. What replaces the measured
    # record here:
    #   * The producer declares `pull_request` AND `merge_group`, carries no
    #     `needs:` and no job-level `if:`, and has no path filter — so it cannot
    #     be skipped-as-passed and cannot wedge a queue SHA.
    #   * It is proven able to FAIL on real input before admission, which is the
    #     OMN-16876 finding-5 vacuous-pass check: run against the tree at
    #     omnibase_infra 033890c6d / omnimarket 4025105c / omnidash 88c05bd it
    #     exits 1 naming `onex.snapshot.projection.consumer-flow.v1` and
    #     `onex.snapshot.projection.tenant-credentials.v1`.
    #   * It is proven able to PASS: the same run greens once each of those two
    #     exposures has a reader or a reasoned `consumers: none`.
    # Because it has no fixture history, the historical-window regression tests
    # in tests/ci/test_ci_summary_gate.py exclude it by name through
    # POST_FIXTURE_WINDOW_CONTEXTS rather than by having synthetic rows invented
    # for merged PRs that never ran it.
    "exposure-reader-coverage",  # exposure-reader-coverage.yml
    # OMN-17172. The KB doc gate (omniclaude kb-doc-gate-reusable.yml, called
    # from .github/workflows/kb-doc-gate.yml, landed in this same PR under Rule
    # 5) blocks adding or modifying markdown outside the allowed set stated by
    # the 2026-09-01 operator ruling. It is registered here rather than in
    # branch protection for the reason the OMN-16878 note above gives: `dev`
    # requires exactly ONE context, so this tuple IS the external enforcement
    # surface on this repo. Admitted under POST_FIXTURE_WINDOW_CONTEXTS — it
    # postdates both fixture windows by construction.
    "kb-doc-gate / kb-doc-gate",
    # OMN-18096. The CI-bus overlay binding gate
    # (.github/workflows/ci-bus-overlay-binding.yml, job id AND job name
    # `ci-bus-overlay-binding`, so the check-run name is that bare string —
    # readback on #3371's head ac679580 and on the merge cc897440 both show
    # exactly `ci-bus-overlay-binding`). It sparse-checks `omnimarket@dev`'s
    # `config/ci_bus_lanes.yaml` and loads it through `ModelCiBusOverlay` in
    # scripts/trigger_rebuild_on_merge.py — the publisher's own `extra="forbid"`
    # model — so a producer-side key that the consumer cannot model reds HERE,
    # named, instead of on whichever unrelated PR next touches runtime.
    #
    # THIS LINE IS THE MECHANISM, on the identical reasoning as the OMN-16878,
    # OMN-17199 and OMN-17172 notes above: `dev` requires exactly ONE context
    # ("CI Summary", the OMN-4497 single-umbrella design), so this tuple IS the
    # whole external enforcement surface. The gate landed in #3371 (OMN-18060,
    # cc897440) deliberately UNregistered and therefore advisory, and CLAUDE.md
    # Operating Rule 5 is explicit that a detector which is not a merge gate is
    # ignored — the two skews this gate exists to catch (OMN-18012 on
    # 2026-09-07, OMN-18060 on 2026-09-09) each cost a red rebuild-trigger and
    # hours of unrelated PRs paying for someone else's merge.
    #
    # ADMISSION IS BY CONSTRUCTION PLUS ONE MEASURED RUN, and the difference
    # from the N-of-16 entries above is stated rather than glossed. The gate is
    # one day old; a 16-merged-PR window cannot exist yet, and waiting for one
    # leaves it unenforced during exactly the window it was filed to close —
    # the same argument recorded for `exposure-reader-coverage`. What stands in
    # for the measured record:
    #   * The producer declares `pull_request` AND `merge_group` (plus push to
    #     dev, a two-hourly schedule and workflow_dispatch), carries no
    #     `needs:`, no job-level `if:` and no path filter — so it cannot be
    #     skipped-as-passed and cannot wedge a queue SHA.
    #   * It is proven able to FAIL on real input, which is the OMN-16876
    #     finding-5 vacuous-pass check: this repo's model at the pre-#3371 tree
    #     rejects the live omnimarket overlay with
    #     `lanes.dev.projection_readback Extra inputs are not permitted`, the
    #     exact error that reddened ten consecutive rebuild-trigger runs.
    #   * It is proven able to PASS: run 2026-09-09T20:33:53Z on #3371's head
    #     ac679580 concluded `success` against the 172-line live overlay, and
    #     the same job concluded `success` on the merge commit cc897440.
    #   * A silently-empty sparse checkout — the one failure shape that would
    #     turn the job into a green no-op — is refused by the workflow's own
    #     "Assert the overlay checkout produced the file" step (`set -euo
    #     pipefail`, `[[ ! -s ... ]] && exit 1`, no `continue-on-error`, no
    #     `|| true`) before pytest runs. Both repos are public, so the cross-
    #     repo checkout resolves on a fork PR's default `github.token` too.
    # Admitted under POST_FIXTURE_WINDOW_CONTEXTS — it postdates both fixture
    # windows by construction. Pinned by
    # tests/ci/test_omn18096_ci_bus_overlay_gate_wiring.py.
    "ci-bus-overlay-binding",  # ci-bus-overlay-binding.yml
    # OMN-18629: the governed-helper primitive gate. Refuses a bare primitive
    # committed where this repo already ships the governed helper superseding
    # it -- the class behind OMN-18608, OMN-18613 and OMN-18606. Registered
    # here rather than in branch protection for the OMN-16878 reason the
    # kb-doc-gate note above gives: `dev` requires exactly ONE context, so this
    # tuple IS the external enforcement surface on this repo. Admitted under
    # POST_FIXTURE_WINDOW_CONTEXTS, which carries the admission argument.
    #
    # PLACED AT THE TAIL DELIBERATELY. The actor-conditional tests pin
    # EXPECTED_EXTERNAL_CONTEXTS[0] as the synthesised absence and the next
    # entry as the failing control, and both must be present in a 2026-07-30
    # fixture. A context with no fixture history cannot serve as either, so a
    # post-fixture-window admission belongs after every historical name rather
    # than in alphabetical position.
    "Governed helper primitive gate",  # governed-helper-primitive-gate.yml
    # OMN-18865: the pre-merge twin of the OMN-14631 workspace content-parity
    # gate. It proves this repository's built wheel carries byte-for-byte the
    # tracked source tree under src/, plus whatever the repo declares
    # force-included into it -- the property the image build already proves on
    # the lab, moved to the pull request that introduces the change. The job
    # is an ordinary job running a local composite action, so the check-run
    # name is the job's own name, one segment.
    #
    # Registered here rather than in branch protection for the OMN-16878
    # reason the kb-doc-gate note above gives: `dev` requires exactly ONE
    # context ("CI Summary", the OMN-4497 single-umbrella design), so this
    # tuple IS the external enforcement surface on this repo. Admitted under
    # POST_FIXTURE_WINDOW_CONTEXTS, which carries the admission argument, and
    # placed at the tail for the reason the entry above states.
    "wheel-content-parity",
    # OMN-19655: the pre-merge twin of the release workflow's PyPI pin-
    # resolvability step. It builds the pull request's wheel and runs the SAME
    # script the release runs, so a floor raise no published sibling can
    # co-resolve fails before merge instead of failing every release after it
    # (omnimarket#2819, omnimarket#2896). Registered here for the OMN-16878
    # reason the kb-doc-gate note above gives: `dev` requires exactly ONE
    # context, so this tuple IS the external enforcement surface on this repo.
    # Admitted under POST_FIXTURE_WINDOW_CONTEXTS, which carries the admission
    # argument, and placed at the tail for the reason the entry above states.
    "pypi-pin-resolvability",
)

# OMN-17199 — contexts admitted AFTER the last historical measurement window
# closed, and therefore absent from the merge-time check-run fixtures replayed
# in tests/ci/test_ci_summary_gate.py.
#
# This is NOT a bypass and must never become one. Membership changes nothing at
# runtime: `evaluate()` never consults this set, so a context listed here is
# asserted present-completed-success on every live PR exactly like every other
# member of EXPECTED_EXTERNAL_CONTEXTS. It exists so the historical replays keep
# asserting what they were written to assert — that admitting a context does not
# wedge dev, measured over PRs that actually merged — without inventing
# synthetic check-run rows for merged PRs that could not have run a workflow
# which did not exist at the time. Fabricating that history would destroy the
# only evidence those tests carry.
#
# ADMISSION RULE: an entry belongs here only while no measurement window
# covering it exists. It comes OUT the moment a fixture window is captured that
# postdates the gate's first run. An entry that has outlived a re-measure is a
# finding, not a fixture convenience.
POST_FIXTURE_WINDOW_CONTEXTS: frozenset[str] = frozenset(
    {
        # Landed with its validator in the same PR (Rule 5) on 2026-08-30; both
        # fixture windows (#2546…#2567, #2705…#2720) close well before that.
        "exposure-reader-coverage",
        # OMN-17172: the caller workflow lands in this same PR on 2026-09-01,
        # so no merged PR in either fixture window could have produced this
        # check-run. Comes out at the next fixture re-capture.
        "kb-doc-gate / kb-doc-gate",
        # OMN-18938: the parity job moves to its own unfiltered workflow in
        # this same PR, so it reports on every pull request from here on, but
        # no merged PR in either fixture window (#2546...#2567, #2705...#2720)
        # ran it under that trigger. Excluded from the HISTORICAL REPLAY only
        # -- TestPostFixtureWindowContexts proves it still blocks when absent
        # from a live payload, which is the assertion that matters. Comes out
        # at the next fixture re-capture.
        "consumer-kwarg-parity",
        # OMN-18096: the producer workflow landed 2026-09-09 in #3371, so no
        # merged PR in either fixture window could have produced this check-run.
        # Comes out at the next fixture re-capture.
        "ci-bus-overlay-binding",
        # OMN-18629: the producer workflow lands in this same PR on 2026-09-17,
        # so no merged PR in either fixture window could have produced this
        # check-run. Comes out at the next fixture re-capture.
        #
        # ADMISSION IS BY CONSTRUCTION, on the argument recorded for
        # `exposure-reader-coverage` above. What stands in for the measured
        # N-of-16 record:
        #   * The producer declares `pull_request`, `merge_group` and push to
        #     dev/main, carries no `needs:`, no job-level `if:`, no path filter
        #     and no failure-tolerating step key -- so it cannot be
        #     skipped-as-passed and cannot wedge a queue SHA.
        #   * It is proven able to FAIL on real input, which is the OMN-16876
        #     vacuous-pass check: a new bare mkdir lock injected into
        #     scripts/disk-gc.sh, a new unbounded consumer commit injected into
        #     services/post_merge/consumer.py, and a new bare interpreter
        #     injected into scripts/disk-watermark-check.sh each exit 1 naming
        #     the file, the line, the pair and the governed helper. It also
        #     exits 1 when a baselined call site is FIXED without its baseline
        #     entry being deleted.
        #   * It is proven able to PASS: the unmutated tree exits 0, and it did
        #     so on this PR's own head after the OMN-18606 rebase.
        #   * It ALREADY failed this PR for a real reason before admission --
        #     the stale lane-census baseline entries after #3717 landed -- so
        #     its ability to refuse is not a claim, it is in this PR's history.
        "Governed helper primitive gate",
        # OMN-18796: the advisory-job gate's caller lands in this same PR on
        # 2026-09-19, so no merged PR in either fixture window could have
        # produced this check-run. Comes out at the next fixture re-capture.
        #
        # ADMISSION IS BY CONSTRUCTION, on the argument recorded for
        # `exposure-reader-coverage` above. What stands in for the measured
        # N-of-16 record:
        #   * The producer (.github/workflows/advisory-job-gate.yml) declares
        #     `pull_request` with no `types:`, no `branches:` filter, no
        #     `paths:` filter, a single `uses:` job with no `needs:` and no
        #     job-level `if:` -- so it reports on every pull-request shape and
        #     cannot be skipped-as-passed.
        #   * The reusable it calls declares NO inputs, so no caller can soften
        #     a refusal, and it fails CLOSED on an unreadable workflow, an
        #     unresolvable enforcement surface, an unparseable baseline and a
        #     malformed annotation alike.
        #   * It is proven able to FAIL on real input: that is what its own
        #     repository's suite pins (omniclaude
        #     tests/scripts/test_advisory_job_gate.py), and the census it reads
        #     rediscovered all three named findings of the OMN-18775 inventory
        #     independently.
        #   * It is proven able to PASS here: run against this repository's
        #     committed baseline at this branch it exits 0 with 28 advisory
        #     settings and 43 verification jobs grandfathered and zero findings.
        #     No baseline was edited to obtain that.
        "advisory-job-gate / advisory-job-gate",
        # OMN-18865: the wheel content-parity caller lands in this same PR on
        # 2026-09-20, so no merged PR in either fixture window could have
        # produced this check-run. Comes out at the next fixture re-capture.
        #
        # ADMISSION IS BY CONSTRUCTION PLUS MEASURED REPLAYS, on the argument
        # recorded for `exposure-reader-coverage` above. What stands in for
        # the measured N-of-16 record:
        #   * The producer (.github/workflows/wheel-content-parity.yml)
        #     declares `pull_request` with no `types:`, no `branches:` filter
        #     and no `paths:` filter, plus `merge_group`, and its single
        #     `uses:` job carries no `needs:` and no job-level `if:` -- so it
        #     reports on every pull-request shape and on a queue SHA, and
        #     cannot be skipped-as-passed.
        #   * The action takes ONE meaningful input, the package's import
        #     name, which selects WHAT is judged and cannot soften a verdict.
        #     There is no force input, no skip input and no allowlist.
        #   * It is proven able to FAIL on real input, on three independent
        #     historical trees, each reproducing a real incident's finding
        #     byte-for-byte: omnimarket at the #2670 merge commit reds on
        #     ['adapters/codex/skills/merge-sweep/SKILL.md'], the same single
        #     path the image gate refused at 21:19Z on 2026-09-19;
        #     omnibase_core at #1710 reds on ['data/gitignore-baseline.yaml']
        #     under pre-#3846 comparison semantics, the same single path the
        #     image gate refused at 15:37Z; and omnibase_compat before the
        #     OMN-14636 fix reds on four files under env/.
        #   * It is proven able to PASS: the dev heads of omnibase_core,
        #     omnibase_compat, omnimarket and this repository all exit 0, and
        #     omnibase_core at #1710 exits 0 under CURRENT semantics, which is
        #     the control proving the force-include arm is a narrowing rather
        #     than a blanket tolerance.
        #   * Its unresolvable cases exit 2, never 0: a missing source package
        #     directory, a failed wheel build, and a build root the repo's own
        #     ignore patterns match all refuse rather than pass.
        "wheel-content-parity",
        # OMN-19655: the pin-resolvability caller lands in this same PR on
        # 2026-09-25, so no merged PR in either fixture window could have
        # produced this check-run. Comes out at the next fixture re-capture.
        #
        # ADMISSION IS BY CONSTRUCTION PLUS MEASURED REPLAYS:
        #   * The producer (.github/workflows/pin-resolvability-gate.yml)
        #     declares `pull_request` with no `types:`, no `branches:` filter
        #     and no `paths:` filter, plus `merge_group`, and its single job
        #     carries no `needs:` and no job-level `if:`, so it reports on
        #     every pull-request shape. A change touching no declared
        #     dependency is judged not applicable and succeeds by design: it
        #     cannot change what resolves. tests/ci/
        #     test_pin_resolvability_gate_workflow.py pins all of that.
        #   * It runs scripts/ci/verify_pypi_pin_resolvability.py, the script
        #     release.yml runs before publishing, with no force or skip input.
        #   * It is proven able to FAIL on real input: omnimarket at the #2896
        #     merge commit 933d0ca8, with the index held to 2026-09-25T18:00Z,
        #     exits 1 naming omnimarket's omnibase-core>=0.47.23 floor against
        #     omnibase-infra 0.38.57's ==0.47.22 pin, the failure omnimarket's
        #     Release on Merge hit on every dev push that day.
        #   * It is proven able to PASS: omnimarket at 933d0ca8's parent exits
        #     0 under the same index, and this repository's dev head exits 0.
        "pypi-pin-resolvability",
    }
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
ACTOR_CONDITIONAL_CONTEXTS: dict[str, tuple[str, ...]] = {}
# EMPTY BY CONSTRUCTION as of OMN-16933, not by omission.
#
# The registry's only entry was `gate / CodeRabbit Thread Check`
# (OMN-15532): cr-thread-gate-caller.yml gated its `gate` job on
# `github.actor != 'dependabot[bot]'`, and because the context name is the
# `caller-job / reusable-job` form, a skipped caller job produced NO
# check-run at all — absent, not `skipped` — which burned the 90 min
# deadline and then failed closed against the SOLE required context on infra
# dev. CodeRabbit was removed entirely (operator ruling 2026-08-29) and both
# cr-thread-gate*.yml files are deleted, so no live producer is actor-scoped.
#
# The MECHANISM is retained deliberately. The next cross-repo reusable whose
# caller carries an actor `if:` will need it, and re-deriving it costs a
# wedged Dependabot lane. Its tests were re-anchored onto a synthetic
# registry entry over the same real #2522 payload (see
# tests/ci/test_ci_summary_gate.py::TestActorConditionalExternalContexts) so
# the falsification control survives the removal.

# Conclusions that count as "provably passed".
GOOD_CONCLUSIONS: frozenset[str] = frozenset({"success", "skipped"})

# External contexts are held to the STRICT bar: `skipped` fails closed. Every
# name above was measured `success` on all 16 sampled PRs (never skipped), so
# this costs nothing today and closes the skip-vector fail-open that OMN-15057 /
# OMN-14854 exist to prevent.
#
# OMN-18062 narrows that bar by exactly one case, before resolution rather than
# here: a `skipped` row for a name that ALSO carries a real row on the same head
# is a re-trigger artifact and is dropped by `drop_superseded_non_verdicts()`. A
# `skipped` with no such row on that head still reaches this frozenset and still
# fails closed. OMN-18355 extends that drop to `neutral` and adds the bounded
# cancellation grace below; neither admits a conclusion here.
EXTERNAL_GOOD_CONCLUSIONS: frozenset[str] = frozenset({"success"})

# OMN-18355 -- conclusions that are NOT a verdict about the head, and which a
# real row for the same name on the same head therefore supersedes.
#
# `skipped` is the OMN-18062 case: a job whose own `if:` was false for a
# re-trigger run says nothing about the commit.
#
# `neutral` is the code-scanning placeholder, measured live on
# `omnibase_infra#3511` (head `45e9d4d8`, 2026-09-14T01:03-01:09Z): the context
# name `CodeQL` is emitted by TWO producers -- the `github-actions` analysis job
# inside `security-scan.yml`, and the `github-advanced-security` app's results
# check. GHAS writes its row seconds after the head appears, concludes `neutral`
# with the title "1 configuration not found", and only updates that same row to
# `success` once the analysis uploads minutes later. Because the placeholder
# STARTS later than the analysis job, latest-wins resolution picks it over the
# live producer and a `neutral` fails the sole required context on a head whose
# analysis is still running. That is not a defect in either producer: the
# placeholder is a promise of a verdict, not a verdict.
#
# `cancelled` is deliberately NOT here -- see CANCELLED_SUPERSESSION_GRACE_S.
# Dropping it would let an older `success` resolve the context green while the
# replacement run is still in flight, which is a stale-green this module must
# never manufacture. A cancellation is handled by waiting, never by ignoring.
NON_VERDICT_CONCLUSIONS: frozenset[str] = frozenset({"skipped", "neutral"})

# OMN-18355 -- how long a `cancelled` external context is treated as "no verdict
# yet" rather than as a failure.
#
# MECHANISM, measured on `omnibase_infra#3512` (head `689344f8`): the OCC
# autobind stamp edits the PR body, every workflow whose `types:` include
# `edited` starts a fresh run, and GitHub cancels the in-flight one. The
# cancelled run's check-run is written AT ONCE; the replacement run's check-run
# for the same context does not exist yet, so for about a minute the
# cancellation is the ONLY row for that name and latest-wins has nothing else to
# pick. `CI Summary` polled 29 seconds into that window, listed
# `call-reject-skip-token / scan / reject-skip-gate-token` under
# external-context failures and exited FAILURE 55 seconds into a run with 27
# in-run gates still pending; the replacement row appeared at 00:02:00Z, 57
# seconds after the cancellation, and concluded `success`. Only a human
# `gh run rerun --failed` cleared it.
#
# A cancellation means the producer was STOPPED BEFORE IT COULD DECIDE. It is
# the absence of a verdict, and this module already has the right response to an
# absent verdict: keep polling. The grace bounds that wait so a cancellation
# nobody replaces still fails closed rather than burning the poller's whole
# 90-minute deadline -- 10 minutes is an order of magnitude above the 57 seconds
# measured, and short against that deadline.
#
# This RELAXES NOTHING that was ever a verdict: `failure`, `timed_out` and
# `action_required` still fail on the poll that observes them, a cancellation
# older than the grace still fails, and the deadline still converts PENDING to
# FAILURE. The only behaviour removed is the terminal verdict issued inside the
# window where a replacement is demonstrably on its way.
CANCELLED_SUPERSESSION_GRACE_S: int = 600

# OMN-17864 -- how long a `failure` external context is treated as "a verdict a
# re-run is about to replace" rather than as this head's answer.
#
# OMN-18355 (above) closed the `cancelled` half of this and said so explicitly:
# "`failure`, `timed_out` and `action_required` still fail on the poll that
# observes them". That residual is the larger half, and this constant closes it
# for `failure` alone.
#
# MECHANISM, measured on `omnibase_infra#3779` (head `ec6c7636`, captures in
# tests/fixtures/omn17864/): on a ticketed PR the OCC evidence companion is
# minted by AUTOMATION after the PR opens. Until it lands the PR body carries no
# evidence-source line and the Receipt Gate (`verify / verify`) is legitimately
# red. When the companion merges, automation PATCHes the PR body; every workflow
# whose `types:` include `edited` re-fires; the Receipt Gate re-runs and goes
# green ON ITS OWN. `CI Summary` polled inside that window, recorded FAILURE at
# 20:24:26Z on a row that had completed 47 seconds earlier, and exited. The
# replacement row concluded `success` at 20:27:16Z. Only a human `gh run rerun`
# cleared it, and that rerun passed with NO CHANGE TO THE PR -- which is the
# proof that nothing was ever wrong with the head.
#
# THE WINDOW IS MEASURED, NOT CHOSEN. Over the last 30 merged `dev` PRs, 16
# exhibited a red `verify / verify` that later went green on the same head;
# every one of the 16 recovered, the slowest in 6.8 minutes, the median in 1.9
# (tests/fixtures/omn17864/verify-verify-recovery-window.json.captured, replayed
# by test_the_grace_exceeds_every_measured_recovery). 20 minutes is ~3x the
# slowest observed and still under a quarter of the poller's 90-minute deadline.
# Over HALF the merged PRs sampled hit this shape: it is the norm, not an
# outlier, which is why a mechanism is warranted rather than a rerun habit.
#
# THIS COSTS ALMOST NOTHING IN THE COMMON CASE. The poller already runs until
# every IN-RUN gate completes -- typically far longer than this grace -- so a
# genuinely-red external context is usually reported at the same moment it would
# have been anyway. The bounded worst case is a docs-shaped PR whose own gates
# finish first: its FAILURE is recorded up to 20 minutes later than before.
#
# THIS RELAXES NOTHING THAT WAS EVER A STABLE VERDICT: a red older than the
# grace still fails, an absent/unparseable/future `completed_at` still fails,
# `timed_out` and `action_required` are untouched, a missing clock restores the
# strict pre-grace reading, the deadline still converts a sustained PENDING into
# FAILURE, and NOTHING here can resolve a context green -- only a real green
# check-run can. The only behaviour removed is the terminal verdict issued
# inside the window where a replacement is demonstrably on its way.
EXTERNAL_FAILURE_SUPERSESSION_GRACE_S: int = 1200

# ---------------------------------------------------------------------------
# OMN-18960 (parent OMN-18943, epic OMN-18527) — LAYER 5: the default-deny
# external sweep.
#
# Layers 1-3 sweep THIS RUN's job list default-deny. Layer 4 asserts a
# hand-maintained tuple of cross-workflow contexts. Between them sat the hole
# this layer closes: layer 4 is a pure WHITELIST LOOKUP — it iterates
# `for context in expected:` and calls `latest.get(context)`, and never walks
# the head's check-run list the other way. A check-run present on the head
# whose name is not in that tuple was never read at all. It could conclude
# `failure` and this module would not see it, on a repository whose `dev`
# requires exactly one context, so that tuple IS the whole external surface.
#
# MEASURED, 16 dev PRs merged 2026-09-20T12:56:06Z -> 2026-09-21T00:06:35Z,
# scoped to check-runs that had STARTED at or before each merge decision
# (rows created by post-merge triggers are structurally invisible to a
# pre-merge poller and counting them would overstate the population):
#
#   * 40-55 unregistered external check-run names per head, 70 distinct
#     across the window, against an EXPECTED_EXTERNAL_CONTEXTS of 26.
#   * 0 of 16 heads carried a NON-GREEN unregistered context at merge time.
#   * POSITIVE CONTROL for that zero (rule 16): the same sweep over the
#     POST-merge head state finds 8 of 16 heads carrying a red unregistered
#     context, so the pipeline detects a red when one is present. Those reds
#     are `Verify dev lane applied the redeploy` and
#     `Verify the k3s onex-lab overlay applied the merged sha`, emitted by
#     runtime-rebuild-trigger.yml jobs gated on
#     `github.event.pull_request.merged == true`.
#   * Event attribution of the swept population: 693 `pull_request`,
#     48 `dynamic`, 20 non-Actions app rows, 1 `push`. That single push row
#     is why event scoping is part of the mechanism and not an optimisation.
#
# EVERY NAME THAT IS NON-GREEN BY DESIGN IS REGISTERED BELOW, with a reason,
# an owner, a date and an expiry. Ten of the seventy qualify: a fork-only
# verification job, three jobs restricted to the main branch or to a non-pull-
# request event, two manual re-publish entrypoints, three App-written
# placeholder rows, and one adversarial gate that does succeed when it runs
# and skips on some pull-request shapes.
# ---------------------------------------------------------------------------

# Events whose check-runs are NOT a verdict on the pull request being gated.
#
# POLARITY IS DELIBERATE: this is a DENY list of non-PR events, not an ALLOW
# list of PR events. A row whose event cannot be resolved — a non-Actions app
# such as code scanning or the change-control writer, a run absent from the
# supplied index, an unparseable URL, or no index supplied at all — is SWEPT.
# An allow list would silently exempt every one of those.
SWEEP_NON_PR_EVENTS: frozenset[str] = frozenset(
    {
        "push",
        # OMN-18970 adversarial review: a queue run's rows are a verdict about
        # a queue commit, not about this pull request. The sweep only runs on
        # `pull_request` today so this cannot currently fire, and it is listed
        # anyway because the deny list is the place a reader looks to learn
        # which events are not pull-request verdicts.
        "merge_group",
        "schedule",
        "workflow_dispatch",
        "release",
        "deployment",
        "deployment_status",
        "repository_dispatch",
        "create",
        "delete",
        "fork",
        "page_build",
        "public",
        "registry_package",
        "watch",
    }
)

# The STRICT bar, and it is the same one layer 4 holds its own tuple to: the
# ONLY conclusion that passes is `success`.
#
# OMN-18979, operator ruling 2026-09-21, firm, and it REPLACES the
# refusal-only set OMN-18960 shipped here. That set failed on a refusal and
# let `skipped` and `neutral` through, argued from the measurement that nine
# of the seventy names are never green by design, and it shipped with an
# EMPTY registry. The ruling is that the combination is a HIDDEN ALLOWLIST: a
# weaker default beside an empty list tolerates exactly what a list would,
# without writing any of it down, and writing it down is the point. The
# honest form is this bar plus a POPULATED registry where every tolerance
# carries a reason, an owner, a date and an expiry.
#
# The bar changed and the graces did not. A `cancelled` or `skipped` row still
# passes through `verdict_is_provisional` first, so a producer that is
# demonstrably about to re-run is PENDING rather than refused on the poll that
# observes it.
# OMN-18991, OPEN QUESTION, recorded here because the fleet currently
# disagrees with itself and a reader of one repository cannot see the other.
#
# `cancelled` is in the failing set HERE and is NOT in onex_change_control's.
# That sibling argues a cancellation is the ABSENCE of a verdict rather than a
# red, and that an unregistered row carries no presence promise to wait on, so
# it reports a cancelled swept row and carries on. This module instead fails
# it once the OMN-18355 grace closes, which is what the 2026-09-21 instruction
# asked for in as many words: a succeeded-then-cancelled pair must still read
# red.
#
# Both readings are defensible and they cannot both be right for the same
# layer. The divergence is flagged rather than resolved unilaterally, because
# picking one silently is how two gates drift into meaning different things
# under one name.
SWEEP_GOOD_CONCLUSIONS: frozenset[str] = frozenset({"success"})


@dataclass(frozen=True)
class SweepExclusion:
    """One dated, ticketed, EXPIRING admission to the layer-5 sweep.

    All four fields are load-bearing and all four are validated:

    * ``reason`` — why this name may go red without blocking. Free text, but
      it may not be empty; an entry nobody argued for is refused.
    * ``ticket`` — the ``OMN-<number>`` that owns removing it. An exclusion
      with no owner is an allowlist entry wearing a costume.
    * ``added`` — the day it was argued, so its age is readable.
    * ``expires`` — an ABSOLUTE date, never a duration. On and after that date
      the entry stops excluding and the name is swept again. That is what
      makes this list closed-ended: an exclusion nobody renews re-arms the
      gate by itself, rather than outliving the defect it was written for.
    """

    reason: str
    ticket: str
    added: str
    expires: str


# The longest window a single entry may claim. An exclusion that needs longer
# than a quarter is not a temporary exception, it is a decision to stop
# enforcing, and it belongs in a re-argued MEASURED_NOT_ENFORCED_CONTEXTS entry
# where the numbers are recorded, not here.
SWEEP_EXCLUSION_MAX_DAYS: int = 90

# TEN ENTRIES, one per name the measurement found non-green on ANY head over
# the 16-PR window recorded above. OMN-18960 shipped this dict EMPTY beside a
# weaker conclusion set; OMN-18979 replaced that pairing with the strict bar
# and these entries, so every tolerance is now a named, dated, owned decision
# rather than a silent one buried in a frozenset.
#
# They all expire on 2026-12-20, ninety days out, INCLUDING the ones whose
# mechanism looks structural — a fork-only job, a main-branch-only job, a
# manual-dispatch entrypoint. The cap is not a prediction that the mechanism
# will change. It is what forces a premise that has held for a quarter to be
# re-read by a person, which is the whole difference between this list and an
# allowlist.
EXTERNAL_SWEEP_EXCLUSIONS: dict[str, SweepExclusion] = {
    "verify": SweepExclusion(
        reason=(
            "The check run verify was skipped on all sixteen measured heads and never "
            "concluded success. The job configuration includes a condition that "
            "requires the pull request head repository to be a fork. Because the "
            "internal pull requests do not satisfy this fork condition, the job skips "
            "execution without producing a result. Excluding this name prevents the "
            "gate from failing on a check that is logically inapplicable to internal "
            "merges."
        ),
        ticket="OMN-18979",
        added="2026-09-21",
        expires="2026-12-20",
    ),
    "occ-autobind / outcome": SweepExclusion(
        reason=(
            "The check run occ-autobind / outcome was neutral on all sixteen heads "
            "and never concluded success. This status row is written by a GitHub App "
            "rather than by GitHub Actions and serves as a placeholder rather than a "
            "substantive verdict. The context also records that this specific check "
            "prints an incorrect label on its own success path. Without this "
            "exclusion the gate would treat the neutral placeholder as a failure and "
            "block the merge."
        ),
        ticket="OMN-18939",
        added="2026-09-21",
        expires="2026-12-20",
    ),
    "occ-autobind-manual-replay": SweepExclusion(
        reason=(
            "The check run occ-autobind-manual-replay was skipped on all sixteen "
            "heads and never concluded success. The job is gated to the manual "
            "workflow_dispatch event which is not triggered by pull request activity. "
            "Consequently the job skips when the gate evaluates the pull request "
            "head. This exclusion allows the gate to ignore a manual re-publish "
            "entrypoint that does not run in this context."
        ),
        ticket="OMN-18979",
        added="2026-09-21",
        expires="2026-12-20",
    ),
    "occ-companion-effect-manual-replay": SweepExclusion(
        reason=(
            "The check run occ-companion-effect-manual-replay was skipped on all "
            "sixteen heads and never concluded success. This job follows the same "
            "pattern as the previous entry and is restricted to the manual dispatch "
            "event. It functions as a manual re-publish entrypoint that does not "
            "execute on pull request triggers. The gate must exclude this name to "
            "avoid failing on a check that is inactive for the current event type."
        ),
        ticket="OMN-18979",
        added="2026-09-21",
        expires="2026-12-20",
    ),
    "Docker Integration Tests": SweepExclusion(
        reason=(
            "The check run Docker Integration Tests was skipped on the seven heads "
            "where it appeared and never concluded success. The job carries a "
            "condition requiring the event to not be a pull request. Since the gate "
            "judges pull request heads specifically the condition is false and the "
            "job skips. Excluding this entry prevents the gate from treating the "
            "inapplicable integration test suite as a blocking failure."
        ),
        ticket="OMN-18979",
        added="2026-09-21",
        expires="2026-12-20",
    ),
    "Security Scan (Trivy)": SweepExclusion(
        reason=(
            "The check run Security Scan (Trivy) was skipped on the seven heads where "
            "it appeared and never concluded success. The job configuration requires "
            "the git reference to be the main branch for execution. On a pull request "
            "head this reference condition is not met so the job skips. This "
            "exclusion ensures the gate does not block merges due to a security scan "
            "that is restricted to the main branch."
        ),
        ticket="OMN-18979",
        added="2026-09-21",
        expires="2026-12-20",
    ),
    "Image Size Analysis": SweepExclusion(
        reason=(
            "The check run Image Size Analysis was skipped on the seven heads where "
            "it appeared and never concluded success. The job shares the same "
            "mechanism as the security scan and requires the git reference to be the "
            "main branch. It therefore skips on pull request heads where the "
            "reference does not match the main branch. The gate excludes this name to "
            "avoid failing on an analysis job that only runs on the main branch."
        ),
        ticket="OMN-18979",
        added="2026-09-21",
        expires="2026-12-20",
    ),
    "occ-autobind / mint status": SweepExclusion(
        reason=(
            "The check run occ-autobind / mint status was neutral on three of the "
            "sixteen heads and never concluded success. This status row is written by "
            "a GitHub App and its neutral conclusion acts as a placeholder rather "
            "than a verdict. The placeholder status does not reflect a failure of the "
            "head code. Excluding this entry prevents the gate from interpreting the "
            "neutral placeholder as a blocking condition."
        ),
        ticket="OMN-18939",
        added="2026-09-21",
        expires="2026-12-20",
    ),
    "occ-companion-effect / mint status": SweepExclusion(
        reason=(
            "The check run occ-companion-effect / mint status was neutral on one of "
            "the sixteen heads and never concluded success. It belongs to the same "
            "producer family and placeholder mechanism as the autobind mint status. "
            "The neutral conclusion is a placeholder and not a substantive assessment "
            "of the pull request. This exclusion allows the gate to ignore the "
            "placeholder status without blocking the merge."
        ),
        ticket="OMN-18939",
        added="2026-09-21",
        expires="2026-12-20",
    ),
    "Hostile Reviewer (adversarial gate)": SweepExclusion(
        reason=(
            "The check run Hostile Reviewer (adversarial gate) succeeded on thirteen "
            "heads and skipped on three. This entry differs from the others because "
            "the check does reach success when it runs. It skips on some pull request "
            "shapes due to a condition involving the draft state and base branch. The "
            "exclusion covers only the skip case to prevent the gate from failing "
            "when the check is conditionally inactive."
        ),
        ticket="OMN-18979",
        added="2026-09-21",
        expires="2026-12-20",
    ),
    # OMN-19218 — the eleventh name, missed by the 16-PR window because none of
    # those heads touched prod-promotion-lineage.yml's path filter.
    "Enforce clean + promoted build source": SweepExclusion(
        reason=(
            "The check run Enforce clean + promoted build source is the lineage-check "
            "job of prod-promotion-lineage.yml, which carries the condition "
            "github.event_name == 'workflow_call' && inputs.enforce_lineage. The "
            "workflow runs on pull_request for a path filter that includes the "
            "deploy-agent executor and deploy-runtime.sh, so on every such pull "
            "request the job skips without producing a verdict. Measured on "
            "omnibase_infra#3980 at head ba61dc95: every producer green, CI Summary "
            "red on this skipped row alone. The job does its real work only when the "
            "prod image build calls the workflow, which this exclusion does not touch."
        ),
        ticket="OMN-19218",
        added="2026-09-22",
        expires="2026-12-20",
    ),
}


# ---------------------------------------------------------------------------
# OMN-19167 — the CONDITIONAL arm of the layer-5 registry.
#
# EXTERNAL_SWEEP_EXCLUSIONS above admits a NAME unconditionally: once listed,
# that row cannot red the umbrella whatever it concluded. That shape is right
# for a job whose every run on a pull-request head is inapplicable — a
# fork-only job, a main-branch-only job, a manual-dispatch entrypoint.
#
# It is WRONG for `occ-autobind` and `occ-companion-effect`. Those caller jobs
# run, and matter, on an ordinary ticketed pull request; a skip there means the
# change-control mint did not happen and is a real refusal. They skip
# LEGITIMATELY on exactly one shape: a dependency-bot pull request carrying no
# ticket token, which doctrine's own PR-title rule EXEMPTS from carrying one.
# Listing the bare names unconditionally would buy six dependency bumps at the
# price of never again noticing a ticketed PR whose mint silently did not run.
#
# So this registry admits a name only when the PRODUCER'S OWN declared
# eligibility predicate is false for this pull request, evaluated here against
# the same facts the producer's `if:` reads. Every condition must hold; any one
# unresolvable admits nothing. A missing --pr-* argument therefore ENFORCES,
# the same property --pr-author and --workflow-runs-file already have.
#
# MEASURED, 2026-09-22, omnibase_infra#3953 head 4af8a2d5b6, job 106734128710:
#   external sweep failures (red, and named by NOTHING else):
#     occ-autobind (skipped), occ-companion-effect (skipped)
# All six of #3953-#3958 were blocked by this, and so was every other
# ticketless dependency-bot pull request in the repository.
#
# NOT PYDANTIC, and deliberately. ci.yml invokes this module as bare
# `python3 scripts/ci/ci_summary_gate.py` with no dependency install step, so
# the module is stdlib-only by construction (its import block is argparse,
# json, re, sys, dataclasses, datetime). A pydantic model here would import-
# error on the runner and take the repository's sole required context down
# with it. Frozen dataclasses with explicit types are the typed form available.


# The two bot logins the producing workflows name in their own `if:`
# (call-occ-autobind.yml, call-occ-companion-effect.yml). This set is NOT the
# title rule's broader "any login ending in the bot suffix" arm: widening it
# here would admit a skipped mint for any App author, and the narrow
# intersection is what keeps this from becoming an allowlist by degrees.
DEPENDENCY_BOT_AUTHORS: frozenset[str] = frozenset({"dependabot[bot]", "renovate[bot]"})

# The ticket token both the title rule and the producers' `if:` look for.
TICKET_TOKEN_RE = re.compile(r"OMN-\d+")

# A MIRROR, not a second policy. Source of truth, read live on 2026-09-22:
#   OmniNode-ai/onex_change_control
#   .github/workflows/pr-title-check-reusable.yml
#   @babdd13ce68f07df20f989f52ff1c4514d03d896
# which is the exact ref .github/workflows/pr-title-check.yml in THIS repo
# pins, so the mirror and the enforcer cannot be reading different revisions
# without that pin moving. Its shell tests, in order, are:
#   1. PR_AUTHOR ends with the bot suffix                 -> exempt
#   2. lowercased title starts chore(deps | build(deps | "bump "  -> exempt
#   3. lowercased title starts "chore: release" | chore(release) | release:
#   4. title matches OMN-[0-9]+                            -> satisfied
# Arms 1-3 are the EXEMPTIONS; arm 4 is compliance, not exemption, so it is
# not mirrored here. tests/ci/test_ci_summary_gate_bot_skip_omn19167.py pins
# the pin, the arm order and a title table against this comment; an upstream
# edit is a red test rather than silent drift.
_TITLE_EXEMPT_PREFIXES: tuple[str, ...] = (
    "chore(deps",
    "build(deps",
    "bump ",
    "chore: release",
    "chore(release)",
    "release:",
)

_BOT_LOGIN_SUFFIX = "[bot]"


def title_rule_exempts_ticket(*, author: str, title: str) -> bool:
    """Mirror of the pinned PR-title reusable's three exemption arms.

    ``True`` means doctrine does not require this pull request to carry a
    ticket token, so the token's ABSENCE is by design rather than an omission.
    An empty author or title returns ``False``: the upstream refuses an empty
    title outright, and an unresolvable fact admits nothing here.
    """

    if not author or not title:
        return False
    if author.endswith(_BOT_LOGIN_SUFFIX):
        return True
    lowered = title.lower()
    return lowered.startswith(_TITLE_EXEMPT_PREFIXES)


@dataclass(frozen=True)
class PullRequestContext:
    """The pull-request facts the conditional registry decides against.

    Every field is supplied by the caller (``--pr-author``, ``--pr-title``,
    ``--pr-head-ref``, ``--event-actor``), so this carries the SAME honest
    limit ``--pr-author`` already does and it is stated rather than implied:
    nothing here proves the values are the head's real ones. What the
    conditions buy is BLAST RADIUS -- the admission is narrow enough that a
    forged context could only ever excuse a skipped change-control mint on a
    pull request already claiming to be an exempt dependency-bot bump, and the
    mint's own absence is still visible on the head. A caller that supplies
    nothing gets no admission at all, which is the case that actually recurs.
    """

    author: str = ""
    title: str = ""
    head_ref: str = ""
    actor: str = ""

    @property
    def is_resolved(self) -> bool:
        """Whether enough is known to judge. Author and title are required.

        ``head_ref`` may legitimately be empty on a payload that omits it, and
        an empty one simply carries no ticket token -- it cannot manufacture
        an admission, only fail to block one the title already earned. ``actor``
        likewise: empty is not a bot login, so it reads as the stricter half.
        """

        return bool(self.author and self.title)

    @property
    def carries_ticket_token(self) -> bool:
        """The producers' own test: a token in EITHER the title or head ref."""

        return bool(
            TICKET_TOKEN_RE.search(self.title) or TICKET_TOKEN_RE.search(self.head_ref)
        )


def occ_caller_job_is_eligible(ctx: PullRequestContext) -> bool:
    """Mirror of the occ caller jobs' own ``if:`` expression.

    ``.github/workflows/call-occ-autobind.yml`` and
    ``call-occ-companion-effect.yml`` both gate on::

        github.actor != 'dependabot[bot]' &&
        github.actor != 'renovate[bot]' &&
        (contains(title, 'OMN-') || contains(head.ref, 'OMN-'))

    (The companion-effect caller carries one further arm about ``edited``
    events, which only ever makes it skip MORE often; mirroring the weaker of
    the two is the conservative direction, because this predicate is used to
    prove a skip was DECLARED and an over-eager ``True`` here blocks rather
    than admits.)

    ``False`` means the job was declared ineligible and its ``skipped``
    check-run is the workflow's own outcome, not a lost run.
    """

    if ctx.actor in DEPENDENCY_BOT_AUTHORS:
        return False
    return ctx.carries_ticket_token


@dataclass(frozen=True)
class ConditionalSweepExclusion:
    """One dated, ticketed, EXPIRING admission that must ALSO argue its case.

    Carries the same four validated fields as :class:`SweepExclusion` so the
    two registries are read by one validator and age out on one clock, plus:

    * ``conclusions`` -- the ONLY conclusions this entry may admit. Everything
      else on the same name still reds. ``skipped`` is not ``failure``, and an
      entry that admitted both would be the unconditional shape wearing a
      condition.
    * ``condition`` -- the name of the predicate that must also hold, resolved
      through :data:`_SWEEP_CONDITIONS`. An entry naming a predicate that does
      not exist is MALFORMED and fails the gate, rather than quietly admitting
      or quietly refusing.
    """

    reason: str
    ticket: str
    added: str
    expires: str
    conclusions: frozenset[str]
    condition: str


def _declared_ticketless_dependency_bot_skip(ctx: PullRequestContext) -> bool:
    """The one condition, and all four parts of it must hold.

    1. The context resolved at all. An absent ``--pr-title`` admits nothing.
    2. The author is one of the two dependency bots the producers name.
    3. Doctrine's title rule exempts this pull request from carrying a ticket,
       under the mirrored predicate -- so the missing token is BY DESIGN.
    4. The producer's own eligibility predicate is FALSE, so the skip is that
       predicate's outcome rather than a coincidence.

    Parts 3 and 4 are not redundant. Part 4 alone would admit a skip on a
    ticketless HUMAN pull request, where the remedy is to add the ticket. Part
    3 alone would admit a skip on a dependency-bot PR whose job was eligible
    and skipped for some other, unexplained reason.
    """

    if not ctx.is_resolved:
        return False
    if ctx.author not in DEPENDENCY_BOT_AUTHORS:
        return False
    if not title_rule_exempts_ticket(author=ctx.author, title=ctx.title):
        return False
    return not occ_caller_job_is_eligible(ctx)


_SWEEP_CONDITIONS: dict[str, Callable[[PullRequestContext], bool]] = {
    "declared_ticketless_dependency_bot_skip": (
        _declared_ticketless_dependency_bot_skip
    ),
}


CONDITIONAL_SWEEP_EXCLUSIONS: dict[str, ConditionalSweepExclusion] = {
    name: ConditionalSweepExclusion(
        reason=(
            f"The caller job {name} is declared ineligible, by its own `if:` in "
            f".github/workflows/call-{name}.yml, on a pull request that carries no "
            "ticket token -- which doctrine's PR-title rule deliberately EXEMPTS a "
            "dependency-bump title from carrying. GitHub writes the ineligible job "
            "as a check-run concluding `skipped`, and the layer-5 strict bar counts "
            "it red, so every ticketless dependency-bot pull request in this "
            "repository was blocked by construction (measured on #3953-#3958, "
            "2026-09-22). This entry admits ONLY the `skipped` conclusion, ONLY "
            "when the author is one of the two dependency bots those workflows "
            "name, ONLY when the title rule exempts the pull request from carrying "
            "a ticket, and ONLY when the producer's own eligibility predicate "
            "evaluates false. A `failure` on this name, a skip on a ticketed pull "
            "request, a skip on a human-authored ticketless pull request, and an "
            "unresolvable pull-request context all still FAIL."
        ),
        ticket="OMN-19167",
        added="2026-09-22",
        expires="2026-12-20",
        conclusions=frozenset({"skipped"}),
        condition="declared_ticketless_dependency_bot_skip",
    )
    for name in ("occ-autobind", "occ-companion-effect")
}


def conditional_exclusion_admits(
    name: str,
    state: JobState,
    *,
    exclusions: dict[str, ConditionalSweepExclusion],
    context: PullRequestContext | None,
    now: datetime | None,
) -> bool:
    """Whether a conditional entry admits THIS row. Fail-closed throughout.

    Refuses when: the name is unregistered; the entry has expired on the same
    absolute-date clock the unconditional registry uses; the row is not
    completed; the conclusion is outside the entry's declared set; no context
    was supplied; or the named predicate is absent from
    :data:`_SWEEP_CONDITIONS` (a malformed entry, also reported by the
    validator, never a silent pass).
    """

    entry = exclusions.get(name)
    if entry is None:
        return False
    if _exclusion_is_expired(entry.expires, now=now):
        return False
    if state.status != "completed" or state.conclusion not in entry.conclusions:
        return False
    if context is None:
        return False
    predicate = _SWEEP_CONDITIONS.get(entry.condition)
    if predicate is None:
        return False
    return predicate(context)


_SWEEP_TICKET_RE = re.compile(r"^OMN-\d+$")
_SWEEP_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

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
    # Only a check-run carries this; in-run jobs leave it None. It is the clock
    # the OMN-18355 cancellation grace is measured from, so it is read from the
    # row rather than assumed from poll order.
    completed_at: str | None = None


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


def _is_non_verdict_row(raw: dict[str, object]) -> bool:
    """True for a completed check-run carrying no verdict about the head.

    :data:`NON_VERDICT_CONCLUSIONS` -- ``skipped`` (the job's own ``if:`` was
    false for that run) and ``neutral`` (the code-scanning placeholder). A row
    that is still running is NOT a non-verdict row: it is a verdict in progress,
    and treating it as one would let a placeholder suppress PENDING.
    """

    return (
        str(raw.get("status") or "") == "completed"
        and str(raw.get("conclusion") or "") in NON_VERDICT_CONCLUSIONS
    )


def _supersession_partition_key(raw: dict[str, object]) -> tuple[str, str]:
    """Partition key for supersession: ``(context name, head SHA)``.

    The head SHA is load-bearing, not decoration. A ``skipped`` row is only a
    re-trigger artifact when a non-skipped row exists for the same name ON THE
    SAME HEAD; a non-skipped row on a DIFFERENT head is a verdict about a
    different commit and must not clear it. Partitioning by name alone would
    let a ``success`` recorded on an earlier head silently suppress a
    ``skipped`` on the head actually being gated — the exact
    skip-as-pass vector (OMN-15057 / OMN-14854) the strict external bar
    exists for, re-opened through the fix for OMN-18062.

    Rows carrying no ``head_sha`` field all share the ``""`` partition, so a
    payload without head SHAs behaves exactly as it did before this guard.
    Unreachable through the sanctioned caller — it fetches
    ``commits/{sha}/check-runs`` for one head — but the safety of that
    rested on convention, and this makes it a property of the function.
    """

    return (str(raw.get("name") or ""), str(raw.get("head_sha") or ""))


def drop_superseded_non_verdicts(
    check_runs: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Drop non-verdict rows for names that also carry a real row (OMN-18062/OMN-18355).

    MECHANISM this closes, measured on onex_change_control#8709 (2026-09-08): a
    ``gh pr edit`` of the PR body fires a SECOND ``pull_request`` run of a
    workflow whose ``types:`` include ``edited``. A job in that run whose own
    ``if:`` excludes ``edited`` is SKIPPED, and GitHub writes a FRESH check-run
    with conclusion ``skipped`` onto the same, unchanged head SHA where that
    very job reported ``success`` 64 seconds earlier. Latest-wins resolution
    picks the skip, :data:`EXTERNAL_GOOD_CONCLUSIONS` admits only ``success``,
    and ``CI Summary`` fails closed on a head nothing regressed on. Re-running
    ``CI Summary`` cannot clear it — the skip is and stays the newest row for
    that name — so only a new head SHA can, and every lane that edits a PR body
    pays a re-push cycle. The same shape is reachable in this repo:
    ``security-scan.yml``'s ``CodeQL`` job carries a draft/label ``if:`` while
    its workflow retriggers on ``labeled``/``unlabeled``/``ready_for_review``.

    A ``skipped`` row is evidence about a WORKFLOW RUN — a job's ``if:`` was
    false for that run's event — not about the head. When a non-skipped row for
    the same name exists on the same head, that row is the verdict about the
    head and the skip is a re-trigger artifact.

    What this deliberately does NOT relax:

    * ``skipped`` with **no** non-skipped row for that name still stands and
      still fails closed — a producer whose ``if:`` was false for the whole life
      of the head never ran, which is exactly the skip-as-pass vector
      (OMN-15057 / OMN-14854) the strict external bar exists for.
    * A ``failure`` (or ``cancelled``) after a ``success`` still wins on
      recency — a failure IS a verdict about the head.
    * A still-running row is non-skipped, so a later skip can never suppress
      PENDING into a stale green.
    * A skip on a DIFFERENT head SHA. Supersession is partitioned by
      ``(name, head_sha)``, not by name alone — see
      :func:`_supersession_partition_key`.

    OMN-18355 widens the dropped set from ``skipped`` alone to
    :data:`NON_VERDICT_CONCLUSIONS`, which adds the code-scanning ``neutral``
    placeholder. Every clause above holds unchanged for it: a lone ``neutral``
    still fails closed, a real verdict still wins on recency, and a still-
    running row still holds the context at PENDING. It does NOT add
    ``cancelled`` — that row is handled by waiting
    (:func:`cancellation_is_provisional`), because dropping it would let an
    older ``success`` green a context whose replacement run is still in flight.
    """

    verdict_keys = {
        _supersession_partition_key(raw)
        for raw in check_runs
        if str(raw.get("name") or "") and not _is_non_verdict_row(raw)
    }
    return [
        raw
        for raw in check_runs
        if not (
            _is_non_verdict_row(raw)
            and _supersession_partition_key(raw) in verdict_keys
        )
    ]


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

    Resolution runs over the rows that survive
    :func:`drop_superseded_non_verdicts`, so a re-trigger skip or a code-scanning
    placeholder cannot supersede a real conclusion, or a run still in progress,
    already recorded for that name on this head (OMN-18062 / OMN-18355).

    OMN-18979 REFINED THE TIE-BREAK, and it is a refinement rather than a new
    rule: ``started_at`` is second-granular, and a superseding attempt writes
    its cancellation in the same second the replacement starts. When two rows
    for one name share a ``started_at``, the previous order fell through to the
    check-run ``id``, which orders by CREATION and can put a cancellation after
    the success that replaced it. ``completed_at`` now sits between the two, so
    a same-second pair is ordered by when each row actually reached its
    conclusion, and ``id`` still breaks a full tie. Nothing changes when
    ``started_at`` differs, which is every case measured on this repository:
    across 5 heads carrying 8 names with BOTH a cancelled and a successful row,
    latest-wins picked the cancelled row zero times.
    """

    return {
        name: _state_from_check_run(name, raw)
        for name, raw in latest_check_run_rows(check_runs).items()
    }


def _state_from_check_run(name: str, raw: dict[str, object]) -> JobState:
    """Project one raw check-run row onto the shared :class:`JobState`."""

    conclusion = raw.get("conclusion")
    completed_at = raw.get("completed_at")
    return JobState(
        name=name,
        status=str(raw.get("status") or ""),
        conclusion=None if conclusion is None else str(conclusion),
        run_attempt=1,
        completed_at=None if completed_at is None else str(completed_at),
    )


def latest_check_run_rows(
    check_runs: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """The RAW winning row per context name, under latest-wins resolution.

    :func:`latest_check_run_by_name` projects these onto :class:`JobState`,
    which drops the producer URL. The OMN-18960 sweep needs that URL to resolve
    which workflow run — and therefore which EVENT — wrote the row, so the
    ordering lives here and both callers read the same winner. Duplicating the
    ordering would let layer 4 and layer 5 disagree about which row is current,
    which is the one way a red could be judged by neither.
    """

    winners: dict[str, dict[str, object]] = {}
    ordering: dict[str, tuple[str, str, int]] = {}
    for raw in drop_superseded_non_verdicts(check_runs):
        name = str(raw.get("name") or "")
        if not name:
            continue
        try:
            run_id = int(str(raw.get("id") or 0))
        except (TypeError, ValueError):
            run_id = 0
        key = (
            str(raw.get("started_at") or ""),
            str(raw.get("completed_at") or ""),
            run_id,
        )
        if name in ordering and key <= ordering[name]:
            continue
        ordering[name] = key
        winners[name] = raw
    return winners


def _parse_timestamp(raw: str | None) -> datetime | None:
    """Parse a GitHub ISO-8601 ``Z`` timestamp, or ``None`` if unreadable."""

    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def cancellation_is_provisional(state: JobState, now: datetime | None) -> bool:
    """True while a ``cancelled`` external context is still awaiting its replacement.

    OMN-18355. A cancellation is not a verdict: the producer was stopped before
    it could decide, and in the measured shape (a body edit re-triggering a
    workflow whose ``types:`` include ``edited``) the replacement run's check-run
    lands under a minute later. Inside
    :data:`CANCELLED_SUPERSESSION_GRACE_S` of the cancellation, the right answer
    is "no verdict yet, poll again" — not FAILURE.

    FAIL-CLOSED IN EVERY UNCERTAIN CASE, which is the half that keeps this from
    becoming a bypass:

    * ``now is None`` (no clock supplied) → not provisional → fails now. A
      caller that forgets to pass the time enforces the OLD, stricter behaviour.
    * an absent or unparseable ``completed_at`` → not provisional → fails now.
    * a cancellation older than the grace → not provisional → fails now.
    * a ``completed_at`` further in the FUTURE than the grace (a clock so wrong
      the row cannot be reasoned about) → not provisional → fails now, rather
      than waiting forever on a skewed timestamp.

    And the poller's own deadline still converts a sustained PENDING into
    FAILURE, so nothing here can make the required context green or absent.
    """

    if state.conclusion != "cancelled" or now is None:
        return False
    completed = _parse_timestamp(state.completed_at)
    if completed is None:
        return False
    age_s = (now - completed).total_seconds()
    return -CANCELLED_SUPERSESSION_GRACE_S <= age_s <= CANCELLED_SUPERSESSION_GRACE_S


#: Conclusions a re-run of the same producer can replace, and which therefore
#: get the OMN-17864 grace. ``failure`` is the measured companion race. ``skipped``
#: joined it on 2026-09-19 after a THIRD live instance on omnibase_infra#3793:
#: `call-reject-skip-token / scan / reject-skip-gate-token` reported ``skipped``
#: because its job ``needs:`` occ-preflight, which had failed while the evidence
#: companion was unmerged. Once the companion merged, the rerun produced
#: ``success`` 38 SECONDS after `CI Summary` had already recorded FAILURE on the
#: stale ``skipped`` row. A dependency skip is not a verdict about this head.
#:
#: THIS DOES NOT REOPEN THE SKIP-AS-PASS VECTOR (OMN-15057 / OMN-14854). That
#: vector is ``skipped`` read as SUCCESS. Here ``skipped`` is read as NO VERDICT
#: YET: the context is held PENDING, a real verdict may supersede it, and if none
#: arrives it still FAILS at the grace. The strict external bar is unchanged --
#: only ``success`` ever passes.
#:
#: ``cancelled`` is absent deliberately: it has its own, shorter grace
#: (:data:`CANCELLED_SUPERSESSION_GRACE_S`) from OMN-18355. ``timed_out`` and
#: ``action_required`` are absent because neither is produced by a producer that
#: an automatic re-run replaces.
SUPERSEDABLE_CONCLUSIONS: frozenset[str] = frozenset({"failure", "skipped"})


def supersedable_verdict_is_provisional(state: JobState, now: datetime | None) -> bool:
    """True while a supersedable external context is inside its re-run window.

    OMN-17864, the residual OMN-18355 named and left open. A red produced
    before the OCC evidence companion exists is a verdict about the PR's
    METADATA AT THAT MOMENT, not about the head: automation lands the companion,
    PATCHes the body, the producer re-fires on the ``edited`` event and replaces
    its own row with a green one. Inside
    :data:`EXTERNAL_FAILURE_SUPERSESSION_GRACE_S` of the failure, the right
    answer is "a replacement is due, poll again" — not a terminal FAILURE the
    only cure for which is a human rerun that changes nothing.

    A ``skipped`` row reaches the same conclusion by a different route: a
    producer whose job ``needs:`` a gate that failed for the same unmerged
    companion is skipped, not run, so its row is a statement about its
    DEPENDENCY, never about this head. See
    :data:`SUPERSEDABLE_CONCLUSIONS` for the live instance and for why this
    does not reopen the skip-as-pass vector.

    Scoped to :data:`SUPERSEDABLE_CONCLUSIONS` DELIBERATELY. ``timed_out`` and
    ``action_required`` are left terminal: the measured mechanism is an
    automatic re-run of a producer that decided or was prevented from
    deciding, and neither of those is that shape.

    FAIL-CLOSED IN EVERY UNCERTAIN CASE, on exactly the terms
    :func:`cancellation_is_provisional` already sets:

    * ``now is None`` (no clock supplied) → not provisional → fails now, so a
      caller that forgets the time enforces the OLD, stricter behaviour.
    * an absent or unparseable ``completed_at`` → not provisional → fails now.
    * a failure older than the grace → not provisional → fails now.
    * a ``completed_at`` further in the FUTURE than the grace (a clock so wrong
      the row cannot be reasoned about) → not provisional → fails now, rather
      than waiting forever on a skewed timestamp.

    And the poller's own deadline still converts a sustained PENDING into
    FAILURE, so nothing here can make the required context green or absent.
    """

    if state.conclusion not in SUPERSEDABLE_CONCLUSIONS or now is None:
        return False
    completed = _parse_timestamp(state.completed_at)
    if completed is None:
        return False
    age_s = (now - completed).total_seconds()
    return (
        -EXTERNAL_FAILURE_SUPERSESSION_GRACE_S
        <= age_s
        <= EXTERNAL_FAILURE_SUPERSESSION_GRACE_S
    )


def verdict_is_provisional(state: JobState, now: datetime | None) -> bool:
    """True when this row is a verdict an automatic replacement is due to replace.

    The union of the two graces, and the single place the poller's
    "keep waiting" decision is made, so the two cannot drift apart.
    """

    return cancellation_is_provisional(
        state, now
    ) or supersedable_verdict_is_provisional(state, now)


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
    *,
    now: datetime | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(failures, missing_or_pending)`` for the declared external contexts.

    ``check_runs is None`` means the caller could not fetch the head SHA's
    check-runs. That is treated as **every** expected context being unobserved —
    PENDING, never success — so a transient API failure retries and a permanent
    one fails closed at the deadline. It must never read as green.

    ``now`` is the observation time the OMN-18355 cancellation grace is measured
    against. Omitting it is the strict, pre-OMN-18355 behaviour: a ``cancelled``
    context fails on the poll that observes it.
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
        elif state.conclusion in EXTERNAL_GOOD_CONCLUSIONS:
            continue
        elif verdict_is_provisional(state, now):
            unresolved.append(context)
        else:
            failures.append(context)
    return sorted(failures), sorted(unresolved)


def provisional_external_verdicts(
    check_runs: list[dict[str, object]] | None,
    expected: tuple[str, ...],
    now: datetime | None,
) -> list[str]:
    """The subset of ``expected`` held PENDING by a due automatic replacement.

    Reporting only — the union of the OMN-18355 cancellation grace and the
    OMN-17864 failure grace. The poller's log is the diagnostic surface for a
    wedged PR, and "pending because a red is about to be re-run" must not read
    the same as "pending because nothing has started".
    """

    if not expected:
        return []
    latest = latest_check_run_by_name(check_runs or [])
    return sorted(
        context
        for context in expected
        if (state := latest.get(context)) is not None
        and state.status == "completed"
        and verdict_is_provisional(state, now)
    )


def provisional_cancellations(
    check_runs: list[dict[str, object]] | None,
    expected: tuple[str, ...],
    now: datetime | None,
) -> list[str]:
    """The subset of ``expected`` held PENDING by a provisional cancellation.

    Reporting only. The poller's log is the diagnostic surface for a wedged PR,
    and "pending because a superseded run was cancelled 20s ago" and "pending
    because nothing has started" are different situations that must not read the
    same.
    """

    if not expected:
        return []
    latest = latest_check_run_by_name(check_runs or [])
    return sorted(
        context
        for context in expected
        if (state := latest.get(context)) is not None
        and state.status == "completed"
        and cancellation_is_provisional(state, now)
    )


def _parse_exclusion_date(raw: str) -> date | None:
    """Parse a ``YYYY-MM-DD`` exclusion date, or ``None`` if unreadable."""

    if not _SWEEP_DATE_RE.match((raw or "").strip()):
        return None
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        return None


def validate_sweep_exclusions(
    exclusions: dict[str, SweepExclusion],
) -> list[str]:
    """Refusal reasons for malformed :data:`EXTERNAL_SWEEP_EXCLUSIONS` entries.

    A non-empty return FAILS the gate. That is the point: an exclusion nobody
    could have reviewed is worse than no exclusion, because it reads as a
    considered decision. The four fields are checked for PRESENCE and SHAPE
    only — no check here can tell whether a reason is a good one.

    Expiry is deliberately NOT a finding. An entry past its date is not
    malformed, it is spent: :func:`active_sweep_exclusions` drops it and the
    name is swept again, so the gate RE-ARMS rather than breaking. The repo's
    own suite carries the other half, a test that fails the moment a live entry
    expires, so the calendar reaches a person through a red test rather than
    through a wedged pull request.
    """

    findings: list[str] = []
    for name, entry in sorted(exclusions.items()):
        if not isinstance(entry, SweepExclusion):
            findings.append(f"{name}: not a SweepExclusion instance")
            continue
        findings.extend(_validate_exclusion_fields(name, entry))
    return findings


def _validate_exclusion_fields(
    name: str,
    entry: SweepExclusion | ConditionalSweepExclusion,
) -> list[str]:
    """The four field checks BOTH registries are held to.

    Extracted so the conditional registry cannot drift into a weaker bar than
    the unconditional one by being validated somewhere else. Every tolerance
    on either list carries a non-empty reason, an OMN ticket that owns removing
    it, a parseable authoring date and an absolute expiry inside the
    ninety-day cap -- or it fails the gate.
    """

    findings: list[str] = []
    if not entry.reason.strip():
        findings.append(f"{name}: reason is empty")
    if not _SWEEP_TICKET_RE.match(entry.ticket.strip()):
        findings.append(
            f"{name}: ticket {entry.ticket!r} is not an OMN-<number> reference"
        )
    added = _parse_exclusion_date(entry.added)
    expires = _parse_exclusion_date(entry.expires)
    if added is None:
        findings.append(f"{name}: added {entry.added!r} is not a YYYY-MM-DD date")
    if expires is None:
        findings.append(f"{name}: expires {entry.expires!r} is not a YYYY-MM-DD date")
    if added is not None and expires is not None:
        if expires <= added:
            findings.append(
                f"{name}: expires {entry.expires} is not after added {entry.added}"
            )
        elif (expires - added).days > SWEEP_EXCLUSION_MAX_DAYS:
            findings.append(
                f"{name}: window {(expires - added).days}d exceeds the "
                f"{SWEEP_EXCLUSION_MAX_DAYS}d cap"
            )
    return findings


def _exclusion_is_expired(expires: str, *, now: datetime | None) -> bool:
    """One expiry clock for both registries.

    ``now is None`` reads as EXPIRED, which is the enforcing answer: a caller
    with no clock cannot prove an entry is still live, and the fail-closed
    response to that is to sweep the row. Mirrors the rule
    :func:`active_sweep_exclusions` applies to the unconditional registry.
    """

    if now is None:
        return True
    parsed = _parse_exclusion_date(expires)
    return parsed is None or now.date() >= parsed


def validate_conditional_sweep_exclusions(
    exclusions: dict[str, ConditionalSweepExclusion],
) -> list[str]:
    """Refusal reasons for malformed :data:`CONDITIONAL_SWEEP_EXCLUSIONS`.

    Runs the SAME four field checks the unconditional registry gets, plus the
    two a conditional entry adds:

    * ``conclusions`` must be non-empty and must not contain ``success`` --
      a success needs no admission, and listing it would make the entry read
      as covering more than it does.
    * ``condition`` must resolve in :data:`_SWEEP_CONDITIONS`. An entry naming
      a predicate that does not exist fails the gate here rather than being
      silently inert at the call site, which is the failure mode that makes a
      registry stop meaning anything.

    A non-empty return FAILS the gate, exactly as the sibling validator does.
    """

    findings: list[str] = []
    for name, entry in sorted(exclusions.items()):
        if not isinstance(entry, ConditionalSweepExclusion):
            findings.append(f"{name}: not a ConditionalSweepExclusion instance")
            continue
        findings.extend(_validate_exclusion_fields(name, entry))
        if not entry.conclusions:
            findings.append(f"{name}: conclusions is empty")
        if "success" in entry.conclusions:
            findings.append(
                f"{name}: conclusions names 'success', which needs no exclusion"
            )
        if entry.condition not in _SWEEP_CONDITIONS:
            findings.append(
                f"{name}: condition {entry.condition!r} resolves to no predicate"
            )
    return findings


def active_sweep_exclusions(
    exclusions: Mapping[str, SweepExclusion | ConditionalSweepExclusion],
    *,
    now: datetime | None,
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Return ``(names still excluding, names whose entry has expired)``.

    An entry excludes on every day STRICTLY BEFORE its ``expires`` date, and
    stops on that date. ``now is None`` excludes NOTHING — a caller with no
    clock cannot judge an expiry, and the fail-closed answer to that is to
    enforce, which is the same rule :func:`cancellation_is_provisional` applies
    to a missing clock.
    """

    if now is None:
        return frozenset(), tuple(sorted(exclusions))
    today = now.date()
    active: set[str] = set()
    expired: list[str] = []
    for name, entry in exclusions.items():
        expires = _parse_exclusion_date(getattr(entry, "expires", ""))
        if expires is None or today >= expires:
            expired.append(name)
        else:
            active.add(name)
    return frozenset(active), tuple(sorted(expired))


_RUN_ID_RE = re.compile(r"/actions/runs/(\d+)(?:/|$)")


def check_run_event_index(
    workflow_runs: list[dict[str, object]] | None,
) -> dict[int, str]:
    """Map workflow-run id -> triggering event, from ``actions/runs?head_sha=``.

    ``None`` or an empty list yields an empty index, under which
    :func:`resolve_check_run_event` resolves every row to ``None`` and the
    sweep judges all of them. A forgotten argument therefore ENFORCES rather
    than exempting, which is the same property ``--pr-author`` has.
    """

    index: dict[int, str] = {}
    for raw in workflow_runs or []:
        try:
            run_id = int(str(raw.get("id") or 0))
        except (TypeError, ValueError):
            continue
        event = str(raw.get("event") or "")
        if run_id and event:
            index[run_id] = event
    return index


def resolve_check_run_event(
    raw: dict[str, object],
    events: dict[int, str],
) -> str | None:
    """The event that produced this check-run, or ``None`` when unresolvable.

    ``None`` is the fail-closed answer: the caller sweeps the row. Rows written
    by a GitHub App rather than Actions — code scanning, the change-control
    writer — carry no run URL and land here by construction, and they are
    exactly the rows an allow list would have exempted for free.
    """

    for key in ("html_url", "details_url"):
        match = _RUN_ID_RE.search(str(raw.get(key) or ""))
        if match:
            return events.get(int(match.group(1)))
    return None


# OMN-18991 — the reason token a settled cancellation reports under.
#
# It is distinct from every other refusal on purpose. A cancellation that
# outlived its grace is not the producer saying no; it is a replacement that
# never arrived, and the remedy is a re-run rather than a code change. A
# reader who cannot tell those apart re-reads a diff looking for a defect that
# is not there. MEASURED on omnibase_infra#3913, head 236f36698: three gate
# contexts were cancelled at 10:39:56Z by a superseding attempt, the poller's
# final verdict landed at 10:50:36Z, forty seconds after
# CANCELLED_SUPERSESSION_GRACE_S closed, and the successful replacements
# started at 10:53:42Z -- 13 minutes 45 seconds after the cancellation,
# against a grace of ten. The head reads green now. Nothing was wrong with it
# then either, which is why the reason names a re-run rather than a defect.
CANCELLED_WITHOUT_REPLACEMENT: str = "cancelled_without_replacement"


def _sweep_failure_reason(name: str, state: JobState, now: datetime | None) -> str:
    """One refusal line, naming the remedy when the remedy is a re-run.

    Every other conclusion reports as ``<name> (<conclusion>)``. A settled
    cancellation reports its own token plus how long past the grace it is, so
    the line says what to do rather than only what happened.
    """

    if state.conclusion != "cancelled":
        return f"{name} ({state.conclusion})"
    completed = _parse_timestamp(state.completed_at)
    if completed is None or now is None:
        return (
            f"{name} ({CANCELLED_WITHOUT_REPLACEMENT}: no replacement row on this "
            f"head and no readable completion time; re-run the producer)"
        )
    age_s = int((now - completed).total_seconds())
    return (
        f"{name} ({CANCELLED_WITHOUT_REPLACEMENT}: cancelled {age_s}s ago, past the "
        f"{CANCELLED_SUPERSESSION_GRACE_S}s re-run grace, and no replacement row "
        f"exists on this head; re-running the producer clears it)"
    )


def evaluate_external_sweep(
    check_runs: list[dict[str, object]] | None,
    *,
    expected: tuple[str, ...],
    in_run_names: frozenset[str],
    self_name: str,
    exclusions: dict[str, SweepExclusion],
    events: dict[int, str],
    now: datetime | None,
    conditional_exclusions: dict[str, ConditionalSweepExclusion] | None = None,
    pr_context: PullRequestContext | None = None,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """Layer 5 — default-deny over every check-run nothing else accounts for.

    Returns ``(failures, in_flight, swept, excluded, provisional)``:

    * ``failures`` — one line per refusal. These FAIL the umbrella.
    * ``in_flight`` — swept names still running. REPORTING ONLY; see below.
    * ``swept`` — every name this layer judged, so a clean run records what it
      looked at instead of printing nothing (rule 16).
    * ``excluded`` — swept-population names an active registry entry admitted,
      from EITHER registry: the unconditional one, which admits a name whatever
      it concluded, or the OMN-19167 conditional one, which admits a specific
      conclusion only while the producer's own declared eligibility predicate
      is false for this pull request. A conditional entry that does not admit
      leaves the row in the swept population, so it reds exactly as before.
    * ``provisional`` — swept names whose replacement is demonstrably due
      inside the re-run grace. These hold the verdict at PENDING; they are
      NOT a quiet pass, and they red as soon as the grace closes.

    WHY ``in_flight`` DOES NOT HOLD THE VERDICT AT PENDING, stated rather than
    left to be discovered. Every other layer's PENDING is backed by a presence
    PROMISE: a gate job is unconditional in ci.yml, an expected external
    context was measured reporting on 16 of 16 heads. An unregistered row
    carries no such promise, so waiting on one means the poller's 90-minute
    deadline — and therefore a FAILURE on the sole required context — can be
    spent on a job that never terminalizes. This repository has already paid
    for that shape once: the needs-gated CI Summary this module replaced
    wedged pull requests indefinitely under self-hosted fleet saturation.
    RESIDUAL, and it is real: a row that goes red AFTER the poller's last poll
    is not seen. It is bounded by the poller running until every in-run gate
    and every expected external context has completed, which is the long pole
    in practice, and the remedy for a name that matters is to REGISTER it in
    EXPECTED_EXTERNAL_CONTEXTS, where presence is asserted.
    """

    if check_runs is None:
        return [], [], [], [], []
    accounted = frozenset(expected) | in_run_names | {self_name}
    active, _expired = active_sweep_exclusions(exclusions, now=now)
    failures: list[str] = []
    in_flight: list[str] = []
    swept: list[str] = []
    excluded: list[str] = []
    provisional: list[str] = []
    for name, raw in sorted(latest_check_run_rows(check_runs).items()):
        if name in accounted:
            continue
        if resolve_check_run_event(raw, events) in SWEEP_NON_PR_EVENTS:
            continue
        if name in active:
            excluded.append(name)
            continue
        swept.append(name)
        state = _state_from_check_run(name, raw)
        if state.status != "completed":
            in_flight.append(name)
            continue
        if state.conclusion in SWEEP_GOOD_CONCLUSIONS:
            continue
        # OMN-19167 — the conditional arm, consulted only for a row that is
        # ALREADY about to red. It can never turn a red into a pass for a name
        # nothing registered, and it is checked AFTER the strict bar so a
        # `success` never reaches it and never reads as "excluded".
        conditional = (conditional_exclusions or {}).get(name)
        if (
            conditional_exclusion_admits(
                name,
                state,
                exclusions=conditional_exclusions or {},
                context=pr_context,
                now=now,
            )
            and conditional is not None
        ):
            # Reported with its conclusion AND the predicate that admitted it,
            # so the verdict line says which of the two registries acted and on
            # what grounds. A reader should never have to open the source to
            # learn why a red row stopped being red.
            excluded.append(f"{name} ({state.conclusion}; {conditional.condition})")
            continue
        if verdict_is_provisional(state, now):
            # OMN-18991: PENDING, not a quiet pass. A replacement is
            # demonstrably due, so the poller looks again; when the grace
            # closes this same row reds through the branch below. Bounded by
            # the grace, and the caller's deadline still converts a sustained
            # PENDING into FAILURE, so nothing here can hold a head open.
            provisional.append(name)
            continue
        failures.append(_sweep_failure_reason(name, state, now))
    return failures, in_flight, swept, excluded, provisional


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
    now: datetime | None = None,
    sweep_external: bool = True,
    sweep_exclusions: dict[str, SweepExclusion] | None = None,
    conditional_sweep_exclusions: dict[str, ConditionalSweepExclusion] | None = None,
    pr_context: PullRequestContext | None = None,
    workflow_runs: list[dict[str, object]] | None = None,
) -> tuple[int, str]:
    """Return ``(exit_code, human_report)`` for the current job snapshot.

    ``external_contexts`` defaults to empty (assert nothing) so non-PR callers —
    ``merge_group`` / ``workflow_dispatch``, where no PR-scoped context set
    exists — are not wedged. The CLI supplies
    :data:`EXPECTED_EXTERNAL_CONTEXTS` and its ``--event-name`` defaults to
    ``pull_request``, so a *forgotten* argument enforces rather than skips.

    ``pr_author`` drops only the contexts that :data:`ACTOR_CONDITIONAL_CONTEXTS`
    marks unreportable for that author (OMN-15532). ``None`` drops nothing.

    ``now`` is the observation time for the OMN-18355 cancellation grace.
    ``None`` is the strict, pre-OMN-18355 reading: a cancelled external context
    fails on the poll that observes it.

    ``workflow_runs`` is the ``actions/runs?head_sha=`` payload the OMN-18960
    layer-5 sweep resolves each check-run's triggering EVENT from. Omitting it
    resolves every row's event to ``None``, which the sweep judges — a
    forgotten argument enforces rather than exempting.

    ``sweep_external`` defaults to TRUE, so a caller that forgets it ENFORCES
    layer 5 rather than skipping it. It exists for one purpose: a test that
    means to exercise layer 4 in isolation can turn layer 5 off and say so,
    instead of the two layers' verdicts being tangled in one assertion. The
    production caller never passes it.
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
        check_runs, external_contexts, now=now
    )
    external_provisional = provisional_external_verdicts(
        check_runs, external_contexts, now
    )

    # (5) OMN-18960 default-deny external sweep: every check-run on the head
    #     that layers 1-4 do not account for. `latest` is keyed by this run's
    #     own job names, so subtracting it is what keeps layer 5 from
    #     re-judging a job layer 3 already allowlisted.
    if sweep_exclusions is None:
        sweep_exclusions = EXTERNAL_SWEEP_EXCLUSIONS
    if conditional_sweep_exclusions is None:
        conditional_sweep_exclusions = CONDITIONAL_SWEEP_EXCLUSIONS
    exclusion_findings = (
        validate_sweep_exclusions(sweep_exclusions)
        + validate_conditional_sweep_exclusions(conditional_sweep_exclusions)
        if sweep_external
        else []
    )
    _active_exclusions, expired_exclusions = (
        active_sweep_exclusions(sweep_exclusions, now=now)
        if sweep_external
        else (frozenset(), ())
    )
    # OMN-19167: a spent CONDITIONAL entry is reported on the same line as a
    # spent unconditional one. Both re-arm the sweep by themselves; the report
    # is how that reaches a person before a pull request discovers it.
    if sweep_external:
        _, conditional_expired = active_sweep_exclusions(
            conditional_sweep_exclusions, now=now
        )
        expired_exclusions = tuple(sorted({*expired_exclusions, *conditional_expired}))
    (
        ext_sweep_failures,
        ext_sweep_in_flight,
        ext_sweep_names,
        ext_sweep_excluded,
        ext_sweep_provisional,
    ) = (
        evaluate_external_sweep(
            check_runs,
            expected=external_contexts,
            in_run_names=frozenset(latest),
            self_name=self_name,
            exclusions=sweep_exclusions,
            events=check_run_event_index(workflow_runs),
            now=now,
            conditional_exclusions=conditional_sweep_exclusions,
            pr_context=pr_context,
        )
        if sweep_external
        else ([], [], [], [], [])
    )

    all_failures = (
        strict_failures
        + skippable_failures
        + sweep_failures
        + external_failures
        + ext_sweep_failures
        # A malformed exclusion entry fails the gate outright. An unreviewable
        # exception is worse than none: it reads as a considered decision.
        + [f"malformed sweep exclusion: {f}" for f in exclusion_findings]
    )
    # OMN-18991: a swept row inside its re-run grace holds the verdict at
    # PENDING rather than passing quietly. Bounded by the grace, after which
    # the same row reds with a named reason, and the caller's deadline still
    # converts a sustained PENDING into FAILURE.
    all_unresolved = (
        gate_missing_or_pending + external_unresolved + ext_sweep_provisional
    )

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
            external_provisional,
            docs_only=docs_only,
            relaxed=relaxed,
            sweep_names=ext_sweep_names,
            sweep_external_failures=ext_sweep_failures,
            sweep_in_flight=ext_sweep_in_flight,
            sweep_excluded=ext_sweep_excluded,
            sweep_expired=list(expired_exclusions),
            sweep_findings=exclusion_findings,
            sweep_external=sweep_external,
            sweep_provisional=ext_sweep_provisional,
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
    external_provisional: list[str] | None = None,
    *,
    docs_only: bool = False,
    relaxed: frozenset[str] = frozenset(),
    sweep_names: list[str] | None = None,
    sweep_external_failures: list[str] | None = None,
    sweep_in_flight: list[str] | None = None,
    sweep_excluded: list[str] | None = None,
    sweep_expired: list[str] | None = None,
    sweep_findings: list[str] | None = None,
    sweep_external: bool = True,
    sweep_provisional: list[str] | None = None,
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
        if external_provisional:
            # OMN-18355: distinguish "nothing has started" from "a superseded
            # run was cancelled and its replacement has not written a check-run
            # yet". Both are PENDING; only one of them resolves on its own.
            lines.append(
                "  external contexts awaiting an automatic replacement "
                "(cancelled, or failed inside the re-run grace): "
                + ", ".join(external_provisional)
            )
    # OMN-18960 layer 5. The count is printed on EVERY verdict, including a
    # clean one: a sweep that finds nothing and says nothing is
    # indistinguishable from a sweep that did not run (rule 16).
    if not sweep_external:
        return "\n".join(lines)
    lines.append(
        "  external default-deny sweep: "
        f"{len(sweep_names or [])} unregistered context(s) judged"
    )
    if sweep_external_failures:
        lines.append(
            "  external sweep failures (red, and named by NOTHING else): "
            + ", ".join(sweep_external_failures)
        )
    if sweep_excluded:
        lines.append(
            "  external sweep exclusions applied: " + ", ".join(sorted(sweep_excluded))
        )
    if sweep_expired:
        lines.append(
            "  external sweep exclusions EXPIRED (no longer excluding): "
            + ", ".join(sorted(sweep_expired))
        )
    if sweep_provisional:
        lines.append(
            "  external sweep rows awaiting an automatic replacement "
            "(cancelled or failed inside the re-run grace; PENDING, re-polled): "
            + ", ".join(sorted(sweep_provisional))
        )
    if sweep_in_flight:
        lines.append(
            "  external sweep rows still running (reported, not waited on): "
            + ", ".join(sorted(sweep_in_flight))
        )
    if sweep_findings:
        lines.append(
            "  external sweep exclusion registry REFUSED: " + "; ".join(sweep_findings)
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


def _load_workflow_runs(path: str | None) -> list[dict[str, object]] | None:
    """Load ``actions/runs?head_sha=`` rows for the OMN-18960 event scoping.

    ``None`` on a missing or unreadable file, which resolves every row's event
    to ``None`` and therefore SWEEPS every row. Unreadable is the stricter
    reading here, not the looser one, so a failed fetch cannot exempt anything.
    """

    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        runs = payload.get("workflow_runs")
        return runs if isinstance(runs, list) else None
    return payload if isinstance(payload, list) else None


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
        "--workflow-runs-file",
        default=None,
        help="Path to the head SHA's actions/runs JSON, used ONLY to resolve "
        "which EVENT produced each check-run so the OMN-18960 default-deny "
        "sweep can skip non-pull-request rows. A missing/unreadable file "
        "resolves every event as unknown, which SWEEPS every row — the "
        "stricter reading, so a failed fetch cannot exempt anything.",
    )
    parser.add_argument(
        "--pr-author",
        default=None,
        help="Login of the PR author. Drops ONLY the ACTOR_CONDITIONAL_CONTEXTS "
        "entries that this author's PRs structurally cannot produce (OMN-15532). "
        "Omitted/empty drops nothing, so a forgotten argument enforces the full "
        "set rather than exempting it.",
    )
    parser.add_argument(
        "--pr-title",
        default=None,
        help="Title of the PR under evaluation. Read ONLY by the OMN-19167 "
        "conditional sweep registry, to decide whether doctrine's PR-title rule "
        "exempts this PR from carrying a ticket token. Omitted/empty admits "
        "NOTHING, so a forgotten argument enforces.",
    )
    parser.add_argument(
        "--pr-head-ref",
        default=None,
        help="Head branch of the PR under evaluation. Read ONLY by the OMN-19167 "
        "conditional sweep registry, as the second place the occ callers look "
        "for a ticket token. Omitted/empty carries no token, which is the "
        "stricter reading.",
    )
    parser.add_argument(
        "--event-actor",
        default=None,
        help="Login that triggered this run (github.actor), which is NOT always "
        "the PR author -- an update-branch makes it the pushing user. Read ONLY "
        "by the OMN-19167 conditional sweep registry, which mirrors the occ "
        "caller jobs' own actor arm. Omitted/empty is not a bot login, the "
        "stricter reading.",
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
        # OMN-19167. Every field defaults to empty, and an empty context
        # resolves nothing, so the conditional registry admits nothing when
        # these arguments are forgotten -- the same fail-closed posture
        # --pr-author and --workflow-runs-file already carry.
        pr_context=PullRequestContext(
            author=args.pr_author or "",
            title=args.pr_title or "",
            head_ref=args.pr_head_ref or "",
            actor=args.event_actor or "",
        ),
        workflow_runs=_load_workflow_runs(args.workflow_runs_file),
        # The poller runs this module once per poll, so wall-clock IS the
        # observation time for the OMN-18355 cancellation grace. It is not a
        # caller-supplied input: there is no flag for it, so it cannot be
        # backdated to keep a stale cancellation provisional.
        now=datetime.now(UTC),
    )
    print(report)
    if args.report_only:
        return EXIT_SUCCESS
    return code


if __name__ == "__main__":
    raise SystemExit(main())
