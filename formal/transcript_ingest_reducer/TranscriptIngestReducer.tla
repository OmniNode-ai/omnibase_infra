------------------------- MODULE TranscriptIngestReducer -------------------------
(* Ingest reducer, terminal outbox and the downstream source-version update   *)
(* (OMN-22258, task T0 of the transcript intent-to-outcome graph plan, tasks  *)
(* T2, T3 and T6).                                                            *)
(*                                                                            *)
(* Two ingest runs r1 and r2 read overlapping manifests. A manifest lists     *)
(* required tuples <<key, revision, digest>>, a key being one (session id,    *)
(* stream id). Each tuple is published as an event; the reducer receives the  *)
(* events in any order (reversed delivery) and one reducer crash can leave an *)
(* applied event un-acknowledged, so it is delivered again.                   *)
(*                                                                            *)
(* Reducer: a session row holds (revision, digest) per key. Compare-and-      *)
(* upsert: an event with a higher revision replaces the row; with an equal    *)
(* revision it leaves the row and, when the digest differs, flags a           *)
(* conflict; with a lower revision it changes nothing.                        *)
(*                                                                            *)
(* Close and terminal outbox: a run's manifest is satisfied when every        *)
(* required tuple is stored at a revision above the required one, or at the   *)
(* required revision with the same digest. The close transaction sets the     *)
(* run closed and writes the outbox row `pending` in ONE step. A publisher    *)
(* publishes the terminal for a pending row and then marks the row `sent`; it *)
(* can die between publish and mark, and so publish the terminal again. The   *)
(* consumer of the terminal deduplicates on the manifest id (the run id).     *)
(*                                                                            *)
(* Downstream: the generic source-version conditional update of every         *)
(* downstream reducer. A result carries the source version it was computed    *)
(* from (1 or 2); results arrive in any order and are redelivered after a     *)
(* crash. The stored result is replaced only by a higher source version.      *)
(*                                                                            *)
(* Property                  Kind       Config that must break it             *)
(* RowNeverDecreases         invariant  mut_accept_any_decreases (the reducer *)
(*                                      stores whatever revision arrives)     *)
(* RowIsHighestApplied       invariant  mut_accept_any_highest (same control)  *)
(* TerminalOnlyWhenSatisfied invariant  mut_term_on_publish (the run closes   *)
(*                                      when its events are published, not   *)
(*                                      when they are stored),                *)
(* EmittedOnlyWhenSatisfied  invariant  mut_term_on_publish_emitted (the same *)
(*                                      control, caught at the publish, not   *)
(*                                      at the close),                        *)
(*                                      mut_no_digest (an equal revision      *)
(*                                      counts whatever its digest)           *)
(* CloseCarriesOutbox        invariant  mut_two_step (completion and the      *)
(*                                      outbox row are two steps; a crash     *)
(*                                      between them loses the terminal)      *)
(* SentWasPublished          invariant  mut_mark_before_publish (the row is   *)
(*                                      marked sent first; a crash before     *)
(*                                      publish loses the terminal)           *)
(* EffectAtMostOnce          invariant  mut_no_dedup (the consumer does not   *)
(*                                      deduplicate on the manifest id)       *)
(* DerivedNeverOlder         invariant  mut_apply_by_arrival (a downstream    *)
(*                                      result replaces by arrival order)     *)
(* DerivedIsNewest           invariant  mut_apply_by_arrival                  *)
(* EventuallyApplied         property   mut_no_fairness (SpecNoFair)          *)
(* ClosedEventuallyEffected  property   mut_two_step (the lost terminal is     *)
(*                                      never effected), mut_no_fairness     *)
(* Witnesses (invariants the cfgs wit_* must violate):                        *)
(*   BothEffectedWitness   wit_both_effected (NoConflict.cfg values: both    *)
(*                         runs close and take effect)                        *)
(*   ConflictWitness       wit_conflict (a digest conflict is flagged)        *)
(*   DuplicateWitness      wit_duplicate_terminal (a terminal published twice *)
(*                         and consumed twice takes effect once)              *)
(*   IgnoredOlderWitness   wit_ignored_older (a lower revision arrives after  *)
(*                         a higher one and is ignored)                       *)
(*   RedeliveryWitness     wit_redelivery (an applied event is delivered     *)
(*                         again)                                             *)
(*                                                                            *)
(* Bounds: 2 runs, 2 keys, revisions 1..2, digests 1..2, 2 crashes in all    *)
(* (a reducer redelivery, a reducer between the two steps of a close, a       *)
(* publisher between publish and mark; a terminal can be published up to      *)
(* three times). results/ holds TLC output for every cfg and model.sha256     *)
(* (sha256 of this file then the cfg files in sorted order).                  *)
(*                                                                            *)
(* Not modelled: the transaction as the database sees it (a step is atomic),  *)
(* more than two runs, a manifest with more than two tuples, a consumer       *)
(* crash (its dedup record and its effect are one step), and time.            *)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS ConflictDig,   \* digest of r2's tuple for k2 (1: same as r1's, 2: a conflict)
          MaxCrash,
          MUT            \* see MutOK

Runs == {"r1", "r2"}
Keys == {"k1", "k2"}
MaxCopies == MaxCrash + 1

\* The manifests: a set of <<key, revision, digest>> per run. They overlap on k1
\* (r1 requires revision 2, r2 revision 1) and on k2 (revision 1).
Man == [r \in Runs |-> IF r = "r1" THEN {<<"k1", 2, 1>>, <<"k2", 1, 1>>}
                                   ELSE {<<"k1", 1, 1>>, <<"k2", 1, ConflictDig>>}]

Ev == UNION {{<<r, t>> : t \in Man[r]} : r \in Runs}

VARIABLES pubd,      \* events emitted by the runs
          pend,      \* events emitted and not yet acknowledged by the reducer
          row,       \* [Keys -> [rev, dig]]; rev 0 is an empty row
          hiRow,     \* ghost: highest revision each row ever held
          seen,      \* ghost: highest revision of any event the reducer has applied, per key
          conflict,  \* ghost: a digest conflict was flagged
          ignored,   \* ghost: a lower revision was ignored
          redeliv,   \* ghost: an applied event was applied again
          applied,   \* ghost: events the reducer has applied at least once
          closed,    \* [Runs -> BOOLEAN]
          mid,       \* [Runs -> BOOLEAN] the two-step close between its steps
          outbox,    \* [Runs -> "none" | "pending" | "sent"]
          pubSt,     \* [Runs -> "idle" | "published" | "marked"] the publisher
          pubTotal,  \* [Runs -> terminals ever published]
          bus,       \* [Runs -> terminals published and not yet consumed]
          dedupIds,  \* the consumer's record of manifest ids it has acted on
          effects,   \* [Runs -> times the terminal took effect]
          crashes,
          dpubd, dpend, dres, dhi, dseen    \* downstream sub-machine

vars == <<pubd, pend, row, hiRow, seen, conflict, ignored, redeliv, applied, closed, mid,
          outbox, pubSt, pubTotal, bus, dedupIds, effects, crashes,
          dpubd, dpend, dres, dhi, dseen>>

Max2(a, b) == IF a > b THEN a ELSE b

Init == /\ pubd = {}
        /\ pend = {}
        /\ row = [k \in Keys |-> [rev |-> 0, dig |-> 0]]
        /\ hiRow = [k \in Keys |-> 0]
        /\ seen = [k \in Keys |-> 0]
        /\ conflict = FALSE
        /\ ignored = FALSE
        /\ redeliv = FALSE
        /\ applied = {}
        /\ closed = [r \in Runs |-> FALSE]
        /\ mid = [r \in Runs |-> FALSE]
        /\ outbox = [r \in Runs |-> "none"]
        /\ pubSt = [r \in Runs |-> "idle"]
        /\ pubTotal = [r \in Runs |-> 0]
        /\ bus = [r \in Runs |-> 0]
        /\ dedupIds = {}
        /\ effects = [r \in Runs |-> 0]
        /\ crashes = 0
        /\ dpubd = {}
        /\ dpend = {}
        /\ dres = 0
        /\ dhi = 0
        /\ dseen = 0

\* ---- ingest ---------------------------------------------------------------

Emit(e) ==
  /\ e \in Ev \ pubd
  /\ pubd' = pubd \cup {e}
  /\ pend' = pend \cup {e}
  /\ UNCHANGED <<row, hiRow, seen, conflict, ignored, redeliv, applied, closed, mid,
                 outbox, pubSt, pubTotal, bus, dedupIds, effects, crashes,
                 dpubd, dpend, dres, dhi, dseen>>

\* The reducer's effect of one event on the row of its key.
ApplyRow(e) ==
  LET t == e[2]
      k == t[1]
      rev == t[2]
      dig == t[3]
      cur == row[k]
      newRow == IF MUT = "accept_any" THEN [rev |-> rev, dig |-> dig]
                ELSE IF rev > cur.rev THEN [rev |-> rev, dig |-> dig]
                ELSE cur
  IN /\ row' = [row EXCEPT ![k] = newRow]
     /\ hiRow' = [hiRow EXCEPT ![k] = Max2(@, newRow.rev)]
     /\ seen' = [seen EXCEPT ![k] = Max2(@, rev)]
     /\ conflict' = (conflict \/ (rev = cur.rev /\ dig # cur.dig))
     /\ ignored' = (ignored \/ (rev < cur.rev))
     /\ redeliv' = (redeliv \/ e \in applied)
     /\ applied' = applied \cup {e}

\* Apply and acknowledge in one transaction.
Apply(e) ==
  /\ e \in pend
  /\ ApplyRow(e)
  /\ pend' = pend \ {e}
  /\ UNCHANGED <<pubd, closed, mid, outbox, pubSt, pubTotal, bus, dedupIds, effects, crashes,
                 dpubd, dpend, dres, dhi, dseen>>

\* The reducer dies after the row write and before the acknowledgement: the
\* event stays pending and is delivered again.
ApplyCrash(e) ==
  /\ crashes < MaxCrash
  /\ e \in pend
  /\ ApplyRow(e)
  /\ crashes' = crashes + 1
  /\ UNCHANGED <<pubd, pend, closed, mid, outbox, pubSt, pubTotal, bus, dedupIds, effects,
                 dpubd, dpend, dres, dhi, dseen>>

\* ---- close and outbox -----------------------------------------------------

\* What the stored row says of one required tuple (the true meaning).
SatTuple(t) == LET c == row[t[1]]
               IN c.rev > t[2] \/ (c.rev = t[2] /\ c.dig = t[3])

SatTrue(r) == \A t \in Man[r] : SatTuple(t)

\* What the reducer reads: the same, except in mut_no_digest.
SatReducer(r) == \A t \in Man[r] :
    IF MUT = "no_digest" THEN row[t[1]].rev >= t[2] ELSE SatTuple(t)

\* Control: the run's events are all published.
AllPublished(r) == \A t \in Man[r] : <<r, t>> \in pubd

CloseOK(r) == IF MUT = "term_on_publish" THEN AllPublished(r) ELSE SatReducer(r)

Close(r) ==
  /\ ~closed[r]
  /\ CloseOK(r)
  /\ closed' = [closed EXCEPT ![r] = TRUE]
  /\ IF MUT = "two_step"
        THEN /\ mid' = [mid EXCEPT ![r] = TRUE]
             /\ UNCHANGED outbox
        ELSE /\ outbox' = [outbox EXCEPT ![r] = "pending"]
             /\ UNCHANGED mid
  /\ UNCHANGED <<pubd, pend, row, hiRow, seen, conflict, ignored, redeliv, applied,
                 pubSt, pubTotal, bus, dedupIds, effects, crashes,
                 dpubd, dpend, dres, dhi, dseen>>

\* mut_two_step: the second step, the outbox row.
CloseOutbox(r) ==
  /\ MUT = "two_step"
  /\ mid[r]
  /\ mid' = [mid EXCEPT ![r] = FALSE]
  /\ outbox' = [outbox EXCEPT ![r] = "pending"]
  /\ UNCHANGED <<pubd, pend, row, hiRow, seen, conflict, ignored, redeliv, applied, closed,
                 pubSt, pubTotal, bus, dedupIds, effects, crashes,
                 dpubd, dpend, dres, dhi, dseen>>

\* The reducer dies between the two steps of mut_two_step; the first is durable.
ReducerCrash ==
  /\ crashes < MaxCrash
  /\ \E r \in Runs : mid[r]
  /\ mid' = [r \in Runs |-> FALSE]
  /\ crashes' = crashes + 1
  /\ UNCHANGED <<pubd, pend, row, hiRow, seen, conflict, ignored, redeliv, applied, closed,
                 outbox, pubSt, pubTotal, bus, dedupIds, effects,
                 dpubd, dpend, dres, dhi, dseen>>

\* ---- publisher ------------------------------------------------------------

DoPublish(r) ==
  /\ bus' = [bus EXCEPT ![r] = @ + 1]
  /\ pubTotal' = [pubTotal EXCEPT ![r] = @ + 1]

\* Publish the terminal of a pending row, then (next step) mark the row sent.
Publish(r) ==
  /\ MUT # "mark_before_publish"
  /\ outbox[r] = "pending"
  /\ pubSt[r] = "idle"
  /\ bus[r] < MaxCopies
  /\ DoPublish(r)
  /\ pubSt' = [pubSt EXCEPT ![r] = "published"]
  /\ UNCHANGED <<outbox, pubd, pend, row, hiRow, seen, conflict, ignored, redeliv, applied,
                 closed, mid, dedupIds, effects, crashes,
                 dpubd, dpend, dres, dhi, dseen>>

Mark(r) ==
  /\ MUT # "mark_before_publish"
  /\ pubSt[r] = "published"
  /\ outbox' = [outbox EXCEPT ![r] = "sent"]
  /\ pubSt' = [pubSt EXCEPT ![r] = "idle"]
  /\ UNCHANGED <<pubd, pend, row, hiRow, seen, conflict, ignored, redeliv, applied,
                 closed, mid, pubTotal, bus, dedupIds, effects, crashes,
                 dpubd, dpend, dres, dhi, dseen>>

\* Control: mark the row sent first, publish in a later step.
MarkFirst(r) ==
  /\ MUT = "mark_before_publish"
  /\ outbox[r] = "pending"
  /\ pubSt[r] = "idle"
  /\ outbox' = [outbox EXCEPT ![r] = "sent"]
  /\ pubSt' = [pubSt EXCEPT ![r] = "marked"]
  /\ UNCHANGED <<pubd, pend, row, hiRow, seen, conflict, ignored, redeliv, applied,
                 closed, mid, pubTotal, bus, dedupIds, effects, crashes,
                 dpubd, dpend, dres, dhi, dseen>>

PublishMarked(r) ==
  /\ MUT = "mark_before_publish"
  /\ pubSt[r] = "marked"
  /\ DoPublish(r)
  /\ pubSt' = [pubSt EXCEPT ![r] = "idle"]
  /\ UNCHANGED <<outbox, pubd, pend, row, hiRow, seen, conflict, ignored, redeliv, applied,
                 closed, mid, dedupIds, effects, crashes,
                 dpubd, dpend, dres, dhi, dseen>>

\* The publisher dies mid-run: whatever it holds in memory is gone, the outbox
\* row stays as it was. A death with nothing in flight changes nothing and is
\* the same as a death between close and publish.
PublisherCrash ==
  /\ crashes < MaxCrash
  /\ \E r \in Runs : pubSt[r] # "idle"
  /\ pubSt' = [r \in Runs |-> "idle"]
  /\ crashes' = crashes + 1
  /\ UNCHANGED <<pubd, pend, row, hiRow, seen, conflict, ignored, redeliv, applied,
                 closed, mid, outbox, pubTotal, bus, dedupIds, effects,
                 dpubd, dpend, dres, dhi, dseen>>

\* ---- consumer of the terminal --------------------------------------------

Consume(r) ==
  /\ bus[r] > 0
  /\ bus' = [bus EXCEPT ![r] = @ - 1]
  /\ IF MUT # "no_dedup" /\ r \in dedupIds
        THEN UNCHANGED <<dedupIds, effects>>
        ELSE /\ dedupIds' = dedupIds \cup {r}
             /\ effects' = [effects EXCEPT ![r] = @ + 1]
  /\ UNCHANGED <<pubd, pend, row, hiRow, seen, conflict, ignored, redeliv, applied, closed,
                 mid, outbox, pubSt, pubTotal, crashes,
                 dpubd, dpend, dres, dhi, dseen>>

\* ---- downstream: generic source-version conditional update ---------------

DEmit(v) ==
  /\ v \in {1, 2} \ dpubd
  /\ dpubd' = dpubd \cup {v}
  /\ dpend' = dpend \cup {v}
  /\ UNCHANGED <<pubd, pend, row, hiRow, seen, conflict, ignored, redeliv, applied, closed,
                 mid, outbox, pubSt, pubTotal, bus, dedupIds, effects, crashes,
                 dres, dhi, dseen>>

DApplyBody(v) ==
  LET new == IF MUT = "apply_by_arrival" THEN v ELSE IF v > dres THEN v ELSE dres
  IN /\ dres' = new
     /\ dhi' = Max2(dhi, new)
     /\ dseen' = Max2(dseen, v)

DApply(v) ==
  /\ v \in dpend
  /\ DApplyBody(v)
  /\ dpend' = dpend \ {v}
  /\ UNCHANGED <<pubd, pend, row, hiRow, seen, conflict, ignored, redeliv, applied, closed,
                 mid, outbox, pubSt, pubTotal, bus, dedupIds, effects, crashes, dpubd>>

DApplyCrash(v) ==
  /\ crashes < MaxCrash
  /\ v \in dpend
  /\ DApplyBody(v)
  /\ crashes' = crashes + 1
  /\ UNCHANGED <<pubd, pend, row, hiRow, seen, conflict, ignored, redeliv, applied, closed,
                 mid, outbox, pubSt, pubTotal, bus, dedupIds, effects, dpubd, dpend>>

Next == \/ \E e \in Ev : Emit(e)
        \/ \E e \in pend : Apply(e) \/ ApplyCrash(e)
        \/ \E r \in Runs : Close(r) \/ CloseOutbox(r) \/ Publish(r) \/ Mark(r)
                           \/ MarkFirst(r) \/ PublishMarked(r) \/ Consume(r)
        \/ ReducerCrash \/ PublisherCrash
        \/ \E v \in {1, 2} : DEmit(v) \/ DApply(v) \/ DApplyCrash(v)

Spec       == Init /\ [][Next]_vars /\ WF_vars(Next)
SpecNoFair == Init /\ [][Next]_vars

\* ---- properties -----------------------------------------------------------

TypeOK == /\ pubd \subseteq Ev
          /\ pend \subseteq Ev
          /\ row \in [Keys -> [rev : 0..2, dig : 0..2]]
          /\ hiRow \in [Keys -> 0..2]
          /\ seen \in [Keys -> 0..2]
          /\ {conflict, ignored, redeliv} \subseteq BOOLEAN
          /\ applied \subseteq Ev
          /\ closed \in [Runs -> BOOLEAN]
          /\ mid \in [Runs -> BOOLEAN]
          /\ outbox \in [Runs -> {"none", "pending", "sent"}]
          /\ pubSt \in [Runs -> {"idle", "published", "marked"}]
          /\ pubTotal \in [Runs -> 0..MaxCopies]
          /\ bus \in [Runs -> 0..MaxCopies]
          /\ dedupIds \subseteq Runs
          /\ effects \in [Runs -> 0..MaxCopies]
          /\ crashes \in 0..MaxCrash
          /\ {dpubd, dpend} \subseteq SUBSET {1, 2}
          /\ {dres, dhi, dseen} \subseteq 0..2

\* A session row's revision never decreases.
RowNeverDecreases == \A k \in Keys : row[k].rev >= hiRow[k]

\* A session row equals the highest revision applied so far.
RowIsHighestApplied == \A k \in Keys : row[k].rev = seen[k]

\* A terminal is emitted only when every required tuple is stored at a higher
\* revision, or at the required revision with equal content: the run is closed
\* only then, and a terminal is published only then.
TerminalOnlyWhenSatisfied == \A r \in Runs : closed[r] => SatTrue(r)
EmittedOnlyWhenSatisfied == \A r \in Runs : (pubTotal[r] >= 1 \/ effects[r] >= 1) => SatTrue(r)

\* Completion and the outbox row are one step: a closed run has its row, except
\* while a two-step close is between its steps.
CloseCarriesOutbox == \A r \in Runs : closed[r] => (outbox[r] # "none" \/ mid[r])

\* A row marked sent has been published (or is between the mark and the publish
\* of a control that marks first).
SentWasPublished == \A r \in Runs : outbox[r] = "sent" => (pubTotal[r] >= 1 \/ pubSt[r] = "marked")

\* The terminal takes effect at most once per manifest id.
EffectAtMostOnce == \A r \in Runs : effects[r] <= 1

\* A downstream result computed from older source versions never replaces one
\* computed from newer, and it is the one of the newest version applied.
DerivedNeverOlder == dres >= dhi
DerivedIsNewest == dres = dseen

\* Under fair delivery every published revision is applied or superseded.
EventuallyApplied == \A e \in Ev :
    (e \in pubd) ~> (e \in applied /\ row[e[2][1]].rev >= e[2][2])

\* A closed run's terminal takes effect.
ClosedEventuallyEffected == \A r \in Runs : closed[r] ~> effects[r] >= 1

\* Witnesses: each is an invariant that the model must violate.
BothEffectedWitness == ~(effects["r1"] = 1 /\ effects["r2"] = 1)
ConflictWitness == ~conflict
DuplicateWitness == ~(\E r \in Runs : pubTotal[r] = 2 /\ effects[r] = 1 /\ bus[r] = 0)
IgnoredOlderWitness == ~ignored
RedeliveryWitness == ~redeliv

=============================================================================
