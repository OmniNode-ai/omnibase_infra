Second reader's check of the M0 model (OMN-22478)
================================================

SecondRead.tla is RebuildAdmission.tla with knobs added. Every knob at its default gives the first
reader's module: the reg_* configs reach the same distinct state counts as its Base_* results. The
parent directory is not changed. Nothing here implements or wires anything.

What the configs are
--------------------

reg_*    The first reader's Base scope with the knobs off. Must match its state counts.
p1_* p2_* p3_* p4_*
         One break per property, none the first reader wrote.
         *_survives and *_rows_hold are breaks the first reader's invariants do not notice: exit 0
         there is the finding. The sibling *_strong or the plain config states the invariant that
         does notice.
live_*   Fairness and scope tests of the liveness split.
fold_*   The model with the additions the verdict asks for, against every invariant at once.

Which knob is which addition
----------------------------

ReserveMode "epoch_first"    the epoch is read before any gate fact, and the reservation row is a
                             compare on epoch and occupancy
ConsGap "counted"            a consumed reservation counts until its lane's TERMINAL, whether or not
                             the claim row has landed
StaleKills FALSE, RecheckFresh TRUE
                             staleness has no writer, so consumption re-reads the time-based facts
BreakDropsReading TRUE       a stop voids earlier WORKING readings of its piece
RecordSeparate TRUE          the holder record stays readable with both engines down
SafetyEscOn TRUE             a safety stop escalates like any other stop
KeptBelowStop TRUE           step 3 at plan head 15ddc745
CauseBlocking "join_aware"   step 3 at plan head 15ddc745, with the next-piece case the strict reading
                             wedges on

Rerun, for each cfg, from this directory:
  java -XX:+UseParallelGC -Xmx6g -cp tla2tools.jar tlc2.TLC -config <cfg> -workers 4 \
    -lncheck final -metadir <fresh dir> SecondRead.tla

results/SUMMARY.txt holds each run's exit, verdict, state count and the commit it ran at.
