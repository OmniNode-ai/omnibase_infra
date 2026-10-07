# Lab-pass criterion bindings

OMN-18488 adds lab-pass checks to the existing evidence autoclose node. The
artifact remains the receipt source; the closer reuses its validated reader
and passes the checks through the shared criterion pin gate.

Both the candidate boot gate and the compose-dev convergence emitter read the
exact cited commit's `contracts/<ticket>.yaml`. No ticket citation or no contract
leaves the receipt unbound. Several ticket citations are ambiguous and refused.

Declare each lab check explicitly as a `dod_evidence` item with the identity
`lab-pass-<lane>-<check name>`, its `binds_ac` labels, and accepted `ac_bindings`
records. The labels must appear in the contract's requirements acceptance list.
For example, a compose-dev `ready_main` probe uses
`id: lab-pass-compose-dev-ready_main`. Its author decides which criteria that
probe proves; the emitter never derives a mapping from criterion text.

The receipt carries its ticket identity, criterion labels, and the accepted
criterion hashes. Draft records do not bind. A declaration without a readable
pin remains unvalidated at closeout. Changing the live ticket text makes its
old pin stale; re-emitting or re-keying an artifact never refreshes that pin.
Re-keying a receipt onto another commit clears the original criterion claims.

The closer reads receipts for the merged product PR's exact merge commit.
Only a PASS receipt supplies verified checks. A FAIL receipt supplies failed
checks even when some individual probes passed. A receipt for another ticket
or commit supplies no evidence. Missing bindings, missing pins, stale pins and
unreadable artifacts retain the ticket with an explicit gap. The existing
redraw, child, cited-PR, gate and reversal fences continue to apply.

Focused verification:

```sh
uv run --frozen pytest tests/scripts/ci/test_lab_pass_criterion_bindings_omn18488.py -q
```

Live acceptance still requires bound receipts from both workflows and a tally
over four consecutive scheduled closer ticks. Unit tests do not establish the
ticket's four-tick metric.
