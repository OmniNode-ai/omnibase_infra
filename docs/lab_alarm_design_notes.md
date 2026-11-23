<!-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc. -->
<!-- SPDX-License-Identifier: MIT -->
# The durable lab alarm: design notes

The narrative sections of the module docstring of `scripts/lab_alarm.py`
(OMN-18867), moved here (verbatim, but for one lane name the hygiene gate refuses) when the alarm's ledger access moved onto the
bus (OMN-26593): the canonical-file-shape ratchet counts docstring lines, and a
baselined file may not grow. The last two sections describe the delivery and
ledger paths as they stood before that move; the module docstring and
`config/lab_alarm.json` state the current ones (the posting consent and the
newest-row read come from the work-ledger reader node, the alarm rows are
appended with `onex-ledger append-rows -`, and the alarm opens no ledger file).

## The Defect This Exists To Close

Three lab conditions ran unobserved in one window:

* the savings writer crash-looping on an oversize snapshot, frozen **nine
  days** at lag 498 (OMN-18851);
* ``omninode-runtime`` crash-looping on a projection grant from a stale pin
  (omnibase_infra#3857);
* lab-pass receipts of the lab lane FAILing for hours while a release lane waited.

Every one was found by a person looking, and the first nine days late.
**Detection time is the defect, not the conditions.**

## Why A Launch Agent And Not A Session Tick

Session-scoped scheduling is in-memory and dies with the session; this has to
survive one. ``StartCalendarInterval`` under launchd is the durable local
trigger, installed by ``scripts/launchd/install-lab-alarm.sh``.

The OMN-17173 correction applies and is the reason the installer prints a
readback: **launchd DOES fire on this Mac** -- the earlier ticks had been
bootout'd, not failed -- so before believing this runs, check BOTH a non-zero
PID column in ``launchctl list`` AND a fresh entry in its own log. A
loaded-but-bootout'd job looks installed and does nothing.

## What This Alarm Does Not Claim

**A zero or a green here is not a liveness statement about the node behind
it.** Measured on this lane on 2026-09-20 (OMN-18881): a consumer group can
read Stable with assigned partitions and zero lag while the node behind it
refuses every message, and both dev-lane runtimes were docker-unhealthy on a
single dimension while ``/ready`` and ``/health`` returned 200. So the lag
condition measures LAG AND ITS GROWTH and nothing more, which is exactly the
failure it was built for, and the evidence string on every reading names which
fact was read so a green cannot be quoted as something it is not.

**A container's STATUS is recorded but not graded.** Measured on this lane on
2026-09-21: ``omnimarket-projection-api`` read ``exited`` with a restart count
of 0, which this condition correctly reports as inside its bound, because the
bound is on RESTARTS. Grading status would need its own per-container
declaration -- the migration gate exits on purpose every boot and a writer does
not -- so the status travels in the evidence where a reader can see it, and
alarming on it is a separate condition nobody has declared bounds for yet.

**No dead-letter condition is built here**, deliberately. The boundary writes
to a DERIVED per-class topic rather than the ``dlq_topics`` a contract
declares, so a probe reading the declared topic sees a permanent zero on a
healthy and an unhealthy lane alike -- two lanes concluded "messages are being
lost" from that zero on 2026-09-20. Adding one means reading the derived topic
AND shipping a positive control proving the reader can see a row at all.

## Delivery Is Consented, Cited, And Edge-Triggered

**No agent message is consent.** Not a dispatch brief, not the ticket, not the
plan. The only thing that authorizes a send is a durable row in the rolling
work ledger, resolved at run time by :func:`resolve_posting_consent`, and every
run records the citation it resolved or the fact that it resolved nothing.

The live authorization is an operator ruling of 2026-09-21 permitting this
alarm to post to one named channel using the lab's existing bot token **by
reference**. The token is read out of an env file by NAME at send time, held
for one request, and never logged, never written to the run record, never put
on a command line and never included in an error message -- a failure names
the variable and the file instead.

Delivery is per NEWLY RAISED alarm, so the edge-triggering above is also the
channel's deduplication: a condition that stays bad posts once, not once an
hour. A failed send is recorded as a failure rather than swallowed, because an
alarm that believes it delivered and did not is this ticket's own defect in a
new place.

## Every Raised Alarm Also Reaches The Ledger (Omn-20794)

On 2026-10-09 this alarm raised ``container_restarts`` for
``omninode-runtime-effects`` at 10:25:05Z and delivered it to Slack only. The
rolling ledger carried no ``lab-alarm`` row, so the orchestrator learned of the
crash fifteen minutes later from a stale projection. Each newly raised alarm now
appends ONE FRICTION row through the ledger's own append
(:func:`make_ledger_appender`), naming the condition, the subject and the
evidence the condition read. It does not wait on Slack: a channel with no
consent row, no token or no route still leaves the ledger row. No new ssh or
HTTP read is made for it.

A row that could not be written is kept in the state file and retried on every
tick until it lands, and the run exits as an alarm meanwhile: an alarm that
believes it recorded and did not is the defect again. A subject that clears and
re-raises inside ``ledger_dedupe_window`` appends no second row, so a flapping
condition leaves one row, not one per flap.
