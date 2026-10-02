# DelegationReaper model (OMN-19441)

One terminal per command id, keyed on the delivering message id and never on the correlation. Commands `c1` and
`c2` share a correlation; `w1` and `w2` both deliver `c1` (a DLQ replay keeps the command id); `w3` and `w4`
deliver `c2`. The spec header in `DelegationReaper.tla` states the abstraction; `REVIEW.md` records the reviews and
the build conditions.

| Property | Kind | Mutation that must break it |
| -- | -- | -- |
| `AtMostOneTerminal` (one terminal identity per command) | invariant | `mut_live_upsert` (handler record overwrites), `mut_ungated_handlers` (two handlers publish) |
| `WireMatchesRecord` (the wire carries the recorded winner) | invariant | `mut_ungated_publish` (a loser publishes) |
| `LateRealKeptAsEvidence` (a loser is kept) | invariant | `mut_no_evidence` (a loser is dropped) |
| `SlotWriteOnce` (a held terminal stays, a handler `timeout` included) | action | `mut_nonatomic_reap` (check, then write) |
| `NoEarlyReap` (not before the deadline) | action | `mut_early_reap` |
| `ClaimTimeFixed` (deadline from the first claim) | action | `mut_refresh_claim` |
| `EveryClaimedAnswered` (each claimed command, the second on a shared correlation included, ends with a terminal of its own) | temporal | `mut_no_reaper`, `mut_no_heal`, `mut_context_last`, `mut_corr_key` |

`wit_*.cfg` are reachability witnesses: each invariant negates a state that must exist, so TLC must report it
violated.

## Run

`results/` holds the committed TLC output for `Model.cfg`, every `mut_*.cfg` and every `wit_*.cfg`, and
`model.sha256`, the SHA-256 of `DelegationReaper.tla` followed by the `*.cfg` files in byte order. From this
directory, with `tla2tools.jar` at `$TLA2TOOLS_JAR` and either `java` or docker:

```bash
export LC_ALL=C
for cfg in Model.cfg mut_*.cfg wit_*.cfg; do
  docker run --rm -v "$PWD":/m:ro -v "$TLA2TOOLS_JAR":/tla2tools.jar:ro -w /m eclipse-temurin:17-jre-alpine \
    java -cp /tla2tools.jar tlc2.TLC -config "$cfg" -workers 4 -metadir /tmp/tlcmeta DelegationReaper.tla \
    > "results/${cfg%.cfg}.out" 2>&1
done
cat DelegationReaper.tla *.cfg | shasum -a 256 | cut -d' ' -f1 > results/model.sha256
```

`tests/unit/formal/test_delegation_reaper_model.py` checks the digest, that `Model.cfg` passes, that each mutant
fails its named property and that each witness is reached.
