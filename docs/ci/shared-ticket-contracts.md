# Shared ticket contract fragments (OMN-17427)

Status: repository-side preparation only. **Do not switch production PRs to
fragments yet.** Required hosted gates still read the root contract alone. This
change supplies a composer and a regression; it does not resolve the exception
class until the gate owners adopt the format.

## Failure and gate ownership

Two branches appending requirements and evidence to `contracts/OMN-17427.yaml`
edit the same list boundaries. Distinct AC numbers alone do not prevent a text
conflict. Independent files remove those shared edits.

The readers configured by this repository are:

| Caller or reader | Owning implementation | Boundary requiring a change |
| --- | --- | --- |
| `.github/workflows/call-repo-evidence-gate.yml` | `omnibase_core/.github/workflows/receipt-gate.yml` at `4e4f5e0404d364e296c5d37db6aa3ab9e07f8ffa` | Reads the root at PR head, overlays only that root into the control tree, and compares root evidence against the original merge base for carried evidence. |
| That reusable's verifier | Released `omnimarket==0.4.305`, `node_dod_verify` | Reads the explicit contract path as one YAML document. It can consume composed ordinary YAML; it does not discover fragments. |
| `.github/workflows/deploy-gate.yml` | `omniclaude/.github/workflows/deploy-gate-reusable.yml` and `.github/actions/deploy-gate/validate_pr_deploy_required.py`, caller pin `ed4300f8a427ead7f099daf69cedc012907cdbd3` | Sparse-checks out the caller's contracts and tests only `<ticket>.yaml` for deploy evidence. |
| `scripts/ci/check_deploy_scope_dod.py` | This repository | Mirrors both hosted readers; must change in the same adoption to keep parity. |

Other root-only consumers include `contract-validation.yml`, the
`evidence-autoclose-sweep.yml` verifier invocation, the in-progress probe hygiene
consumer, and `scripts/ci/lab_pass_receipt.py`'s exact-commit AC binding. They
also need the composed view before fragment evidence becomes authoritative.
The existing strict-load pre-commit hook scans root YAML files only.

Changing only this repository's caller cannot add preparation steps to an
external reusable job. Replacing the required gate or treating unchanged root
evidence as new evidence would weaken its control. Neither is part of this
preparation.

## Proposed additive format

Keep the legacy root unchanged. Each lane adds one uniquely named
`contracts/<ticket>.d/<scope>.yaml`. A scope is a lowercase, hyphen-separated
branch slug; choose it once and keep it stable after landing. Each fragment has
the same `schema_version` and `ticket_id` as the root and may append only
`requirements`, `evidence_requirements`, and `dod_evidence`.

Requirement IDs start with `req-<scope>-`, evidence IDs with `dod-<scope>-`,
and AC IDs with `AC.<scope>.<positive integer>`, replacing scope hyphens with
dots. Dots survive the verifier's label normalization; hyphens do not. Every
criterion has a statement and is explicitly
bound by the fragment's own non-empty checks. IDs cannot duplicate the root or
another fragment. Fragments cannot overwrite root metadata or bind another
scope's criteria. Existing root declarations are retained verbatim as data,
including historical repeated AC labels; this preparation does not renumber them.

For example, `contracts/OMN-17427.d/alpha.yaml`:

```yaml
schema_version: "1.0.0"
ticket_id: OMN-17427
requirements:
  - id: req-alpha-1
    statement: The lane's behavior holds.
    acceptance:
      - id: AC.alpha.1
        statement: The focused regression passes.
dod_evidence:
  - id: dod-alpha-1
    description: The focused regression checks the lane's behavior.
    source: manual
    binds_ac: [AC.alpha.1]
    checks:
      - check_type: test_passes
        check_value: uv run --frozen pytest tests/ci/test_compose_ticket_contract_omn17427.py -q
        cwd: ${OMNI_HOME}/omnibase_infra
```

Preview with:

```sh
uv run --frozen onex node node_ticket_contract_compose_effect \
  --input tests/ci/fixtures/ticket_contract_compose_request_omn17427.json \
  --backend event_bus=inmemory --state-root .onex_state/ticket-contract-compose
```

The contract routes `HandlerTicketContractCompose` and returns ordinary YAML in
`ModelTicketContractComposeResult.contract_yaml`; local-runtime state records
the typed terminal result. Gate adapters can also use
`omnibase_infra.handlers.handler_ticket_contract_compose.compose_ticket_contract`
for the parsed mapping. The composer never edits the root or runs a
declared check. It rejects duplicate YAML keys, unsupported files, symlinks,
ticket/schema mismatches, duplicate identities, and missing or foreign AC
bindings. File order is lexical and output is deterministic. The evidence
verifier remains responsible for judging and executing checks; structural
composition is not a passing evidence verdict.

## Adoption decision and remaining work

One decision is needed: should `omnibase_core`'s receipt gate and `omniclaude`'s
deploy gate adopt this additive fragment format before lanes start using it?

On adoption, compose the head and original merge base separately. Overlay new
head fragments along with the test side into the control tree, then compose
that tree too. Compare composed evidence when identifying carried checks.
Preserve the refusal when a PR binds no new evidence and the requirement that
each newly bound check fail against the control. Resolve sibling-checkout
declarations from all fragments as well. Do not overwrite a tracked root and
then confuse that generated file with the original merge-base root.

Use the same composition semantics in the deploy gate and its local hook,
update the other consumers above, repin the callers, and only then migrate
authors. The regression models two independent branch trees and uses Git's
text merge for every file in both landing orders. Both contributions validate
and survive, while the root remains unchanged. This proves the storage and
composition approach, not hosted-gate adoption.
