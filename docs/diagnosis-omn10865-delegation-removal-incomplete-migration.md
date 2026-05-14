# Diagnosis: OMN-10865 / PR #1569 — delegation removal blocked by incomplete migration

## What is known

PR omnibase_infra#1569 deletes the delegation pipeline from omnibase_infra (node_delegation_orchestrator,
node_delegation_quality_gate_reducer, node_delegation_routing_reducer, adapter_llm_caller_delegation, bifrost
delegation config, routing_tiers.yaml, etc.), on the premise that delegation now lives in omnimarket
(omnimarket#607, merged 2026-05-12, merge commit a73edb04).

The PR-fix work done so far (commits on branch `jonah/omn-10865-remove-delegation-nodes`):
- `0040f7ebd` — repin omnimarket source branch → main; remove stale arch-handler-contract-compliance-allowlist
  entries + stale `_RUNTIME_UTILITY_EXCLUSIONS` / `_INV4_WIRING_EXEMPTIONS` entries for deleted files
- `50457e5a4` — merge `origin/main` (resolved 3 modify/delete conflicts on node_delegation_orchestrator/*,
  removed main's newly-added node_delegation_routing_reducer/dispatchers/ from OMN-10868, reverted
  INFRA_MAX_UNIONS 154→152, kept all non-delegation main changes, removed obsolete
  TestDelegationContractDeclaresPluginManaged from test_plugin_managed_subscription.py)
- `a369dca91` — moved `PluginDelegation` import into the guarded bootstrap() registration block (CodeRabbit
  finding; also fixes a module-load circular import)
- `fb25443e0`, `cb4b951a3` — empty commits to refresh cached PR_EVENT_BODY for receipt/deploy gates after
  the PR body was updated (added `Evidence-Ticket: OMN-10865` + `Evidence-Source: OCC#965`)

CI state on HEAD `cb4b951a3` after multiple reruns:
- PASS: `gate / CodeRabbit Thread Check`, `deploy-gate / deploy-gate`, `Handler Contract Compliance`,
  all the static/contract/lint gates
- FAIL: `verify / verify` (OCC receipt binding — see below)
- FAIL: `CI Tests Gate` / `CI Summary` — 3 of 15 test splits fail with REAL failures (see below);
  earlier reruns also hit flaky cancellations of detect-changes' `needs:` deps (Fingerprint Check,
  Kafka Schema Handshake) which cascaded to skip the test matrix entirely

## What was tried and why it failed

1. `gh run rerun --failed` on the CI workflow (twice) — cleared the flaky detect-changes cascade but then
   exposed 3 genuine test-split failures (below). Reruns do not fix real failures.
2. Updating PR body with `Evidence-Source: OCC#965` + empty-commit pushes — fixed deploy-gate, but
   `verify / verify` still fails because `validator_occ_merge_eligibility` requires a PASS receipt that binds
   to PR #1569 (or one of its commit SHAs); OCC PR #965's receipts bind to pr_number 965 (OCC itself) and
   607 (omnimarket), not 1569.

## Root cause hypotheses

**(A) The migration is incomplete on the omnimarket side — omnimarket main still imports from infra's
deleted delegation orchestrator.**
`omnimarket.nodes.node_output_schema_registry_compute.handlers.handler_output_schema_registry._register()`
does `from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_result import
ModelDelegationResult`. When infra deletes `node_delegation_orchestrator`, this import fails at auto-wiring
time → `test_wire_from_manifest_main_profile_no_crash` and `test_pattern_b_broker_omnimarket.py` fail with
`ModuleNotFoundError: No module named 'omnibase_infra.nodes.node_delegation_orchestrator'`. omnimarket#607
moved the delegation *nodes* but did not update this consumer to import `ModelDelegationResult` from
omnimarket's own delegation orchestrator package.

**(B) `routing_tiers.yaml` deletion is over-broad — it is used by a non-delegation feature.**
`tests/unit/nodes/node_llm_inference_effect/handlers/test_handler_llm_cli_subprocess_claude_opencode.py`
(added to main by OMN-10137 #1429, AFTER #1569 branched) reads
`src/omnibase_infra/configs/routing_tiers.yaml` to verify claude-cli / opencode-cli tier registration —
nothing to do with delegation. #1569 deletes that file. → 3 failures in split 7. The file must be retained
(or this test is also obsolete and should be deleted — needs a call on whether claude-cli/opencode-cli tier
config also moved to omnimarket).

**(C) Entry-points mismatch** — `test_generate_entry_points.py::test_current_repo_entry_points_match_contract_backed_nodes`
fails `assert 1 == 0`; likely a knock-on of (A)/(B) (a node dir partially present, or a contract.yaml without
a matching entry point). Needs investigation once (A)/(B) are resolved.

**(D) `NodeAislopSweep.__init__() missing 1 required positional argument: 'event_bus'`** in
`test_pattern_b_broker_omnimarket.py` — looks like a separate pre-existing/unrelated breakage in
omnimarket-side wiring; verify whether it reproduces on infra `main` without this PR.

**(E) OCC receipt binding** — `verify / verify` needs an onex_change_control receipt that binds to PR #1569.
Per memory `reference_occ_receipt_binding`, this requires a new OCC PR (consumer receipt referencing #1569 +
self-binding receipt + contract update), OR re-running OCC #965's receipt generation against #1569's HEAD.
Cross-repo, structural — needs a decision.

## Proposed fix with rationale

**Do not land #1569 in its current scope.** The delegation migration is not actually complete:
1. **omnimarket first:** ship an omnimarket PR updating `node_output_schema_registry_compute`'s
   `handler_output_schema_registry._register()` to import `ModelDelegationResult` from
   `omnimarket.nodes.node_delegation_orchestrator.models.model_delegation_result` (the location #607 created),
   and audit for any other `from omnibase_infra.nodes.node_delegation*` imports in omnimarket. Merge that.
2. **Decide `routing_tiers.yaml`:** either keep it in omnibase_infra (revert its deletion in #1569) if the
   claude-cli/opencode-cli tier config legitimately stays in infra, or move it to omnimarket and update
   `test_handler_llm_cli_subprocess_claude_opencode.py` accordingly. Then re-evaluate #1569's config-deletion list.
3. **Then rebase #1569** onto current main, re-check entry-points (C) and the NodeAislopSweep wiring (D),
   and run the full suite.
4. **OCC binding (E):** generate/commit an OCC receipt binding to #1569's final HEAD before the receipt gate
   will pass.

## What was done in this branch (preserved)

The PR-fix commits above are sound for what they cover (omnimarket repin, merge resolution, CodeRabbit fix,
stale-allowlist cleanup, PR body) and should be kept when #1569 is reworked — they are not the cause of the
failures. The failures stem from the original PR's deletion scope being premature relative to omnimarket's
state and to post-branch additions on infra main.
