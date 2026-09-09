# OMN-18056 — the nine adjudicated tickets, as measured

`<TICKET>.verdict.json` is a real `uv run onex skill dod_verify <TICKET> --dry-run`
receipt, captured 2026-09-08 22:40Z–23:00Z from the `omnibase_infra` canonical
clone, trimmed to the verdict and to each check's `evidence_id` / `status` /
`proof_class` / `binds_ac`. Counters are verbatim. Positive control on the
capture method: OMN-17298 reproduces `total=7 verified=3 failed=0 skipped=1
non_probative=4 behavior_proving=1` — character-for-character the receipt the
DoD closeout sweep of 2026-09-08 recorded for its one (later reverted) flip.

`binds_ac` is `[]` on every check of all nine. That is the measurement, not an
omission: a corpus grep over all 8709 OCC contracts for any acceptance-binding
field returns 0 files, against a positive control of 8708 for `dod_evidence`.

`<TICKET>.ac.md` is the ticket's acceptance-criteria section, read live from
Linear on 2026-09-08 and stored verbatim. It is the SECTION, not the whole
body, and the narrowing is stated because it matters: `_acceptance_criteria_items`
reads the entire body when no recognised heading exists anywhere, so the live
body of a ticket like OMN-15922 parses MORE items than its section does here.
A narrower input can only UNDER-count criteria, and under-counting releases a
flip rather than holding one — so a HOLD asserted on the trimmed section is a
hold a fortiori on the full body. The direction of the approximation is the
safe one, and the tests never assert a FLIP from this corpus.

ONE EXCEPTION, named rather than buried: `OMN-17298.ac.md` is reconstructed
from the sweep report's statement of the unmet criterion (AC6b), not read from
the live ticket body. Its verdict JSON is a real capture; its AC text is not.
