# Database work-ledger reads on the dev lane

`onex ledger query` calls omnimarket's `node_work_ledger_query_effect`, which
reads `omninode_internal.work_ledger_rows`. It returns the node's JSON answer,
including full row text, freshness, parity status and query counts.

The transport declaration is
`src/omnibase_infra/contracts/work_ledger_query_client.contract.yaml`.
The broker address and protocol come from the dev lane declaration. The
credential resolver uses the machine's stored dev identity in `~/.onex`,
then the existing lab/CI credential environment, and refuses an absent or
broken identity. There is no raw broker, principal or password CLI option.

| Principal | Read path |
| --- | --- |
| Issued developer/machine principal (`dev-*`), also used by its lanes | WRITE/DESCRIBE on `onex.cmd.omnimarket.work-ledger-query-requested.v1`; READ/DESCRIBE on `onex.evt.omnimarket.work-ledger-query-completed.v1`; READ/DESCRIBE on the prefixed group `local.omnimarket.work_ledger_query_client.consume.v1.` |
| Read node's existing `omninode_runtime` database binding | Existing SELECT on `work_ledger_rows`; executes the read behind the command topic |
| Parity checker, `work_ledger_ro` | Existing read-only access to `work_ledger_rows` and `work_ledger_state` |
| Unrelated broker principal without those grants | Broker refuses the query; the CLI exits nonzero and prints no answer |

The six developer query grants already landed under OMN-21339. They remain
derived from `deploy/lab/developer-principal-bindings.yaml` by
`HandlerBrokerGrantDerive`; this change connects the client to them. It adds
no wildcard, database privilege, result-publishing permission or permission
to join the runtime's command-consumer group. The issuer's existing
`apply-grants` operation can reconcile an already issued identity without
rotating its credential. It must use the dev declaration and the approved
principal list. A successful live dev read in this lane required no grant
mutation or credential rotation.

## Reader wiring

The existing shared ledger source adapter is
the shared database source for the `onex-ledger` query and catchup read verbs.
Configure its argv prefix in the reader's environment:

```sh
export ONEX_LEDGER_BUS_READ_COMMAND='onex ledger query --bus-lane dev --stdin'
export ONEX_LEDGER_BUS_TIMEOUT_S=60
onex-ledger query --source db --since 2026-10-25T00:00:00Z --ticket OMN-20740
onex-ledger catchup --source db --since 2026-10-25T00:00:00Z
```

Use the installed `onex` executable's absolute path when the reader has a
restricted PATH. The argv prefix consumes one query object on stdin and
prints one answer object on stdout. The caller's correlation UUID and filters
are preserved, including an absent limit: `read_db_lines` refuses a truncated
answer. The same prefix serves ledger-ask's query read and the existing
catchup implementation, retaining their existing rendering and read rules.

For an interactive typed query without the source adapter:

```sh
onex ledger query --bus-lane dev --query rows --ticket OMN-20740 --since 2026-10-25T00:00:00Z
```

`--stdin` rejects filter flags rather than overriding a reader's request.
The reply consumer establishes its latest starting offset before publishing
the request and uses a fresh group under the exact granted prefix. It
matches the envelope and payload correlation, requires freshness and parity,
propagates broker refusal, and closes both clients after success or failure.
A node database error, malformed correlated answer or timeout cannot become
an empty successful result. A refusal diagnostic contains the error class,
not credential-bearing driver text.

This supplies the reader command; it does not change the readers' default
source or flip ledger authority. S7 shadow comparison and the S8 seven-day
parity/claim-token gates still govern those steps. Production is not an
accepted lane for this command.

## Verification

```sh
uv run --frozen pytest tests/unit/cli/test_work_ledger_query_client_omn20740.py tests/unit/cli/test_work_ledger_query_grants_omn20740.py -q
uv run --frozen pytest tests/integration/customer_path/test_work_ledger_query_acl_omn20740.py -q
```

The broker test uses the existing disposable SCRAM Redpanda harness. An
authenticated developer is refused before grants are installed, succeeds
after exactly the six derived dev grants are applied, and an unrelated
authenticated principal is refused. It requires a disposable broker and
refuses an adopted shared lane. Its responder exercises the read node's
wire protocol; the node's database/replay logic remains tested in omnimarket.

The first lab proof additionally ran the real shared `read_db_lines` adapter against
the live dev node as the machine's issued developer identity: 656 rows returned, with correlation
and complete row count verified, freshness present and today's parity
`unmeasured` in the query answer. That status is not an exact daily receipt.
A later attempt to run the existing query/catchup verbs against the live
node timed out at the query read. The command refused the read and emitted
no answer; no fallback was allowed for that probe. The direct source-adapter
success is evidence of client wiring, not a claim of sustained runtime health.

On 2026-10-25, the projection's own parity checker measured the canonical
file snapshot against an export read as `work_ledger_ro`. The initial
00:00–02:35Z window had 561 file rows and 560 projection rows:
missing/extra/state mismatches **1/0/0**, backfilled **0**. The missing row is
the `lab-stack-reaper` STATUS at 02:14:53Z. A fresh export compared through
02:39Z had 589 file rows and 589 projection rows, with **1/1/0**,
unexplained **1**, backfilled **0**, and 128 state entities compared. The
additional projection-only row is `occi-plan-1452-r13-0210`'s STATUS stamped
02:37:44Z (`492ad8e39909…`). Because the file is a frozen lane snapshot,
that extra needs rechecking against the current canonical file; it is not
established as a permanent projection extra. The missing reaper row persists.
The file's state horizon is 02:40Z; three entities advanced with witnessed
projection rows after it and were counted separately by #4365's checker.
Today is not on course for 0/0/0 on this evidence. October 25 is only the first eligible full UTC day after omnimarket#4365
merged on October 24 at 16:23:09Z. No seven-day or authority-flip success is
claimed by this client change.
