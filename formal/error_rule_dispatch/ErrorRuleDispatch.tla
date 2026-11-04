--------------------------- MODULE ErrorRuleDispatch ---------------------------
(* The orchestrated error-rule workflow (OMN-19998): event, class, row, action,
   read-back, escalation. One key is one dispatch key
   cause=<class key>@<subject>@<head>; Subjects are the subjects that share one
   class and one head, so each subject is its own key with its own budget.

   Writers. Two deciders (the tick and the foreground) take events off the
   adapter. A claim is a unique-key insert keyed by (key, gen): the winner
   proceeds, a loser is a duplicate. The effect re-reads the table row before
   every send and every retransmission and proceeds only when the row's
   revision is the one the claim holds and its mode is still dispatch;
   otherwise it cancels and writes a STATUS. The receiver ignores a repeat of
   an effect_id (its idempotent_by key), and a retransmission first reads the
   receiver's state and sends only when it finds nothing. An operator-question
   row keeps an asked marker per key on the claim row, so a question is asked
   once per key. The lane a sent effect starts may fix the failure or not, and
   its report (carrying its gen) may be delayed or lost. The decide node drops
   a report older than the key's gen (the fence). A lease that runs out
   abandons the generation; the reaper reaps the abandoned run before the
   read-back. The read-back reads the live failure: the key is cleared only if
   the failure is gone, otherwise the next gen is claimed while budget is left,
   otherwise the key escalates once. A cleared key whose failure returns on
   the same head escalates once.

   Mapping to the schema. Mode classes: dispatch (claims) and shadow (records
   and never claims; hint and wait behave as shadow here: they never claim).
   Action classes: RowAction = "action" (landing-intent, fill-backlog-item,
   install-branch: natural key is the effect_id) or "question"
   (operator-question: natural key is the dispatch key). status[k] is the
   projection row's state; claimCount[k][g] counts CLAIM rows for (k, g);
   execs[k][g] counts receiver executions of effect_id sha256(k, g).

   Fairness. Every system step (decide, claim, send, retransmit, lane finish,
   report consume, lease expiry, reap, read-back) is weakly fair. The
   environment (event arrival, acknowledgement delivery, demotion, recurrence)
   is not: an acknowledgement or a report can be lost forever.

   Bounds. Budget gens per key (hard stop at Budget + 1 for the no-budget
   mutation), MaxEvents events per key, one retransmission per gen, one
   demotion of the row, one recurrence per key. Time is abstract: a lease may
   run out at any point while its generation is open.

   Not covered. One owner per pull request (the controller's lease, task 2.4);
   the Limits gates; the content of any action; real-time lease lengths and
   the receiver's own window; crash of the claim store itself.

   Properties:
     AtMostOneClaimPerGen   one CLAIM row per (key, gen)
     GenWithinBudget        a key never passes its budget
     EffectOnce             each effect_id executes at most once at its receiver
     AskOnce                an operator question is asked at most once per key
     NoShadowDispatch       nothing is claimed or sent while the row is not dispatch
     EscalateOnce           a key escalates at most once
     StaleIgnored           a report older than the key's gen changes nothing (action)
     ClearOnlyWhenGone      a key becomes cleared only while its failure is gone (action)
     EveryKeyEnds           every key ends cleared, escalated, cancelled or shadow
     RecurrenceEscalates    a cleared failure that returns escalates
   Mutations (cfg constants) that must break them:
     AtomicClaim=FALSE   check then insert in two steps     -> AtMostOneClaimPerGen
     Fence=FALSE         a late lane's report is applied     -> StaleIgnored
     LaneClears=TRUE     the lane's report clears the key    -> ClearOnlyWhenGone
     BudgetOn=FALSE      no budget                           -> GenWithinBudget
     ModeGate=FALSE      claim and send ignore the mode      -> NoShadowDispatch
     AskedMarker=FALSE   no asked marker on the claim row    -> AskOnce
     IdemCheck=FALSE     a retransmission ignores the receiver -> EffectOnce, AskOnce
     ExpiryOn=FALSE      no lease expiry                     -> EveryKeyEnds *)
EXTENDS Integers, FiniteSets
CONSTANTS Subjects, RowAction, Budget, MaxEvents,
          AtomicClaim, Fence, LaneClears, BudgetOn, ModeGate, AskedMarker,
          IdemCheck, ExpiryOn, DemoteOn

Keys == Subjects
Actors == {"tick", "fg"}
Gens == 1..(Budget + 1)
Done == {"cleared", "escalated", "cancelled", "shadow"}
NoKey == "none"

VARIABLES pending, arrived, apc, status, gen, claimCount, claimRev, eff,
          retried, execs, asked, questions, lane, inbox, delivered, failing,
          recurred, mode, rev, escalations, shadowAct, viaExpiry, staleDropped
vars == <<pending, arrived, apc, status, gen, claimCount, claimRev, eff,
          retried, execs, asked, questions, lane, inbox, delivered, failing,
          recurred, mode, rev, escalations, shadowAct, viaExpiry, staleDropped>>

Init ==
  /\ pending = [k \in Keys |-> 1]
  /\ arrived = [k \in Keys |-> 1]
  /\ apc = [a \in Actors |-> NoKey]
  /\ status = [k \in Keys |-> "new"]
  /\ gen = [k \in Keys |-> 0]
  /\ claimCount = [k \in Keys |-> [g \in Gens |-> 0]]
  /\ claimRev = [k \in Keys |-> 0]
  /\ eff = [k \in Keys |-> "none"]
  /\ retried = [k \in Keys |-> FALSE]
  /\ execs = [k \in Keys |-> [g \in Gens |-> 0]]
  /\ asked = [k \in Keys |-> FALSE]
  /\ questions = [k \in Keys |-> 0]
  /\ lane = [k \in Keys |-> 0]
  /\ inbox = {}
  /\ delivered = {}
  /\ failing = [k \in Keys |-> TRUE]
  /\ recurred = [k \in Keys |-> FALSE]
  /\ mode = "dispatch"
  /\ rev = 1
  /\ escalations = [k \in Keys |-> 0]
  /\ shadowAct = FALSE
  /\ viaExpiry = [k \in Keys |-> FALSE]
  /\ staleDropped = FALSE

RowLive(k) == rev = claimRev[k] /\ mode = "dispatch"

(* The claim: a CLAIM row for (k, g), the lease, effect_id sha256(k, g). *)
ClaimVars(k, g) ==
  /\ gen' = [gen EXCEPT ![k] = g]
  /\ claimCount' = [claimCount EXCEPT ![k][g] = @ + 1]
  /\ claimRev' = [claimRev EXCEPT ![k] = rev]
  /\ status' = [status EXCEPT ![k] = "dispatched"]
  /\ eff' = [eff EXCEPT ![k] = "claimed"]
  /\ retried' = [retried EXCEPT ![k] = FALSE]
  /\ viaExpiry' = [viaExpiry EXCEPT ![k] = FALSE]
  /\ shadowAct' = (shadowAct \/ mode # "dispatch")

EscalateVars(k) ==
  /\ status' = [status EXCEPT ![k] = "escalated"]
  /\ escalations' = [escalations EXCEPT ![k] = @ + 1]

(* Decide on one event: record-shadow, dispatch (claim), escalate a recurrence,
   or ignore-duplicate. *)
Decide(a, k) ==
  /\ apc[a] = NoKey
  /\ pending[k] > 0
  /\ pending' = [pending EXCEPT ![k] = @ - 1]
  /\ IF status[k] = "new"
       THEN IF ModeGate /\ mode # "dispatch"
              THEN /\ status' = [status EXCEPT ![k] = "shadow"]
                   /\ UNCHANGED <<apc, gen, claimCount, claimRev, eff, retried,
                                  viaExpiry, shadowAct, escalations>>
              ELSE IF AtomicClaim
                     THEN /\ ClaimVars(k, 1)
                          /\ UNCHANGED <<apc, escalations>>
                     ELSE /\ apc' = [apc EXCEPT ![a] = k]
                          /\ UNCHANGED <<status, gen, claimCount, claimRev, eff,
                                         retried, viaExpiry, shadowAct, escalations>>
       ELSE IF status[k] = "cleared" /\ failing[k]
              THEN /\ EscalateVars(k)
                   /\ UNCHANGED <<apc, gen, claimCount, claimRev, eff, retried,
                                  viaExpiry, shadowAct>>
              ELSE UNCHANGED <<apc, status, gen, claimCount, claimRev, eff,
                               retried, viaExpiry, shadowAct, escalations>>
  /\ UNCHANGED <<arrived, execs, asked, questions, lane, inbox, delivered,
                 failing, recurred, mode, rev, staleDropped>>

(* Mutation only: the insert after a separate read saw the key free. *)
ClaimStep2(a) ==
  /\ apc[a] # NoKey
  /\ ClaimVars(apc[a], 1)
  /\ apc' = [apc EXCEPT ![a] = NoKey]
  /\ UNCHANGED <<pending, arrived, execs, asked, questions, lane, inbox,
                 delivered, failing, recurred, mode, rev, escalations,
                 staleDropped>>

(* The receiver for effect_id (k, g). A question row with its asked marker set
   asks nothing. *)
Exec(k, g, useMarker) ==
  IF RowAction = "question" /\ useMarker /\ AskedMarker /\ asked[k]
    THEN UNCHANGED <<execs, questions, lane, asked>>
    ELSE /\ execs' = [execs EXCEPT ![k][g] = @ + 1]
         /\ lane' = [lane EXCEPT ![k] = g]
         /\ asked' = [asked EXCEPT ![k] = TRUE]
         /\ questions' = IF RowAction = "question"
                           THEN [questions EXCEPT ![k] = @ + 1]
                           ELSE questions

CancelVars(k) ==
  /\ eff' = [eff EXCEPT ![k] = "cancelled"]
  /\ status' = [status EXCEPT ![k] = "cancelled"]

Send(k) ==
  /\ status[k] = "dispatched"
  /\ eff[k] = "claimed"
  /\ IF ModeGate /\ ~RowLive(k)
       THEN /\ CancelVars(k)
            /\ UNCHANGED <<execs, questions, lane, asked, shadowAct>>
       ELSE /\ Exec(k, gen[k], TRUE)
            /\ eff' = [eff EXCEPT ![k] = "sent"]
            /\ shadowAct' = (shadowAct \/ mode # "dispatch")
            /\ UNCHANGED status
  /\ UNCHANGED <<pending, arrived, apc, gen, claimCount, claimRev, retried,
                 inbox, delivered, failing, recurred, mode, rev, escalations,
                 viaExpiry, staleDropped>>

(* Environment: the acknowledgement reaches the projection (or never does). *)
Ack(k) ==
  /\ eff[k] = "sent"
  /\ eff' = [eff EXCEPT ![k] = "acked"]
  /\ UNCHANGED <<pending, arrived, apc, status, gen, claimCount, claimRev,
                 retried, execs, asked, questions, lane, inbox, delivered,
                 failing, recurred, mode, rev, escalations, shadowAct,
                 viaExpiry, staleDropped>>

ReceiverHas(k) ==
  IF RowAction = "question" THEN questions[k] > 0 ELSE execs[k][gen[k]] > 0

(* A retransmission of the same attempt, same effect_id. *)
Retry(k) ==
  /\ status[k] = "dispatched"
  /\ eff[k] = "sent"
  /\ ~retried[k]
  /\ retried' = [retried EXCEPT ![k] = TRUE]
  /\ IF ModeGate /\ ~RowLive(k)
       THEN /\ CancelVars(k)
            /\ UNCHANGED <<execs, questions, lane, asked, shadowAct>>
       ELSE /\ IF IdemCheck
                 THEN IF ReceiverHas(k)
                        THEN /\ eff' = [eff EXCEPT ![k] = "acked"]
                             /\ UNCHANGED <<execs, questions, lane, asked>>
                        ELSE /\ Exec(k, gen[k], TRUE)
                             /\ UNCHANGED eff
                 ELSE /\ Exec(k, gen[k], FALSE)
                      /\ UNCHANGED eff
            /\ shadowAct' = (shadowAct \/ mode # "dispatch")
            /\ UNCHANGED status
  /\ UNCHANGED <<pending, arrived, apc, gen, claimCount, claimRev, inbox,
                 delivered, failing, recurred, mode, rev, escalations,
                 viaExpiry, staleDropped>>

(* The lane ends: it may fix the failure, and its report may be lost. *)
LaneFinish(k) ==
  /\ lane[k] # 0
  /\ \E fixed \in BOOLEAN, deliver \in BOOLEAN :
       /\ failing' = [failing EXCEPT ![k] = @ /\ ~fixed]
       /\ inbox' = IF deliver THEN inbox \cup {<<k, lane[k]>>} ELSE inbox
       /\ delivered' = IF deliver THEN delivered \cup {<<k, lane[k]>>} ELSE delivered
       /\ status' = IF LaneClears /\ status[k] = "dispatched" /\ gen[k] = lane[k]
                      THEN [status EXCEPT ![k] = "cleared"]
                      ELSE status
  /\ lane' = [lane EXCEPT ![k] = 0]
  /\ UNCHANGED <<pending, arrived, apc, gen, claimCount, claimRev, eff, retried,
                 execs, asked, questions, recurred, mode, rev, escalations,
                 shadowAct, viaExpiry, staleDropped>>

(* The decide node takes a completion; the fence drops an older gen. *)
Consume(k, g) ==
  /\ <<k, g>> \in inbox
  /\ inbox' = inbox \ {<<k, g>>}
  /\ IF Fence /\ g < gen[k]
       THEN /\ staleDropped' = TRUE
            /\ UNCHANGED status
       ELSE /\ status' = IF status[k] = "dispatched" /\ (g = gen[k] \/ ~Fence)
                           THEN [status EXCEPT ![k] = "readback"]
                           ELSE status
            /\ UNCHANGED staleDropped
  /\ UNCHANGED <<pending, arrived, apc, gen, claimCount, claimRev, eff, retried,
                 execs, asked, questions, lane, delivered, failing, recurred,
                 mode, rev, escalations, shadowAct, viaExpiry>>

Expire(k) ==
  /\ ExpiryOn
  /\ status[k] = "dispatched"
  /\ status' = [status EXCEPT ![k] = "expired"]
  /\ UNCHANGED <<pending, arrived, apc, gen, claimCount, claimRev, eff, retried,
                 execs, asked, questions, lane, inbox, delivered, failing,
                 recurred, mode, rev, escalations, shadowAct, viaExpiry,
                 staleDropped>>

(* The runner's lane reaper: the abandoned run is reaped (or confirmed never
   started) before the read-back and before any next generation. *)
Reap(k) ==
  /\ status[k] = "expired"
  /\ lane' = IF lane[k] = gen[k] THEN [lane EXCEPT ![k] = 0] ELSE lane
  /\ eff' = IF eff[k] \in {"claimed", "sent"}
              THEN [eff EXCEPT ![k] = "abandoned"] ELSE eff
  /\ status' = [status EXCEPT ![k] = "readback"]
  /\ viaExpiry' = [viaExpiry EXCEPT ![k] = TRUE]
  /\ UNCHANGED <<pending, arrived, apc, gen, claimCount, claimRev, retried,
                 execs, asked, questions, inbox, delivered, failing, recurred,
                 mode, rev, escalations, shadowAct, staleDropped>>

(* The read-back reads the live failure. *)
Readback(k) ==
  /\ status[k] = "readback"
  /\ IF ~failing[k]
       THEN /\ status' = [status EXCEPT ![k] = "cleared"]
            /\ UNCHANGED <<gen, claimCount, claimRev, eff, retried, viaExpiry,
                           shadowAct, escalations>>
       ELSE IF (~BudgetOn \/ gen[k] < Budget) /\ gen[k] < Budget + 1
              THEN IF ModeGate /\ mode # "dispatch"
                     THEN /\ status' = [status EXCEPT ![k] = "cancelled"]
                          /\ UNCHANGED <<gen, claimCount, claimRev, eff, retried,
                                         viaExpiry, shadowAct, escalations>>
                     ELSE /\ ClaimVars(k, gen[k] + 1)
                          /\ UNCHANGED escalations
              ELSE /\ EscalateVars(k)
                   /\ UNCHANGED <<gen, claimCount, claimRev, eff, retried,
                                  viaExpiry, shadowAct>>
  /\ UNCHANGED <<pending, arrived, apc, execs, asked, questions, lane, inbox,
                 delivered, failing, recurred, mode, rev, staleDropped>>

(* Environment. *)
Arrive(k) ==
  /\ arrived[k] < MaxEvents
  /\ arrived' = [arrived EXCEPT ![k] = @ + 1]
  /\ pending' = [pending EXCEPT ![k] = @ + 1]
  /\ UNCHANGED <<apc, status, gen, claimCount, claimRev, eff, retried, execs,
                 asked, questions, lane, inbox, delivered, failing, recurred,
                 mode, rev, escalations, shadowAct, viaExpiry, staleDropped>>

Demote ==
  /\ DemoteOn
  /\ mode = "dispatch"
  /\ mode' = "shadow"
  /\ rev' = rev + 1
  /\ UNCHANGED <<pending, arrived, apc, status, gen, claimCount, claimRev, eff,
                 retried, execs, asked, questions, lane, inbox, delivered,
                 failing, recurred, escalations, shadowAct, viaExpiry,
                 staleDropped>>

(* The cleared failure returns on the same head, with its event. *)
Recur(k) ==
  /\ status[k] = "cleared"
  /\ ~recurred[k]
  /\ failing' = [failing EXCEPT ![k] = TRUE]
  /\ recurred' = [recurred EXCEPT ![k] = TRUE]
  /\ pending' = [pending EXCEPT ![k] = @ + 1]
  /\ UNCHANGED <<arrived, apc, status, gen, claimCount, claimRev, eff, retried,
                 execs, asked, questions, lane, inbox, delivered, mode, rev,
                 escalations, shadowAct, viaExpiry, staleDropped>>

SysActs(k) ==
  \/ \E a \in Actors : Decide(a, k)
  \/ Send(k) \/ Retry(k) \/ LaneFinish(k)
  \/ \E g \in Gens : Consume(k, g)
  \/ Expire(k) \/ Reap(k) \/ Readback(k)

Next ==
  \/ \E k \in Keys : SysActs(k) \/ Ack(k) \/ Arrive(k) \/ Recur(k)
  \/ \E a \in Actors : ClaimStep2(a)
  \/ Demote

Spec == Init /\ [][Next]_vars
        /\ \A k \in Keys : WF_vars(SysActs(k))
        /\ \A a \in Actors : WF_vars(ClaimStep2(a))

TypeOK ==
  /\ status \in [Keys -> {"new", "shadow", "dispatched", "expired", "readback",
                          "cleared", "escalated", "cancelled"}]
  /\ eff \in [Keys -> {"none", "claimed", "sent", "acked", "cancelled",
                       "abandoned"}]
  /\ gen \in [Keys -> 0..(Budget + 1)]

(* Safety. *)
AtMostOneClaimPerGen == \A k \in Keys, g \in Gens : claimCount[k][g] <= 1
GenWithinBudget == \A k \in Keys : gen[k] <= Budget
EffectOnce == \A k \in Keys, g \in Gens : execs[k][g] <= 1
AskOnce == \A k \in Keys : questions[k] <= 1
NoShadowDispatch == ~shadowAct
EscalateOnce == \A k \in Keys : escalations[k] <= 1

StaleIgnored ==
  [][\A k \in Keys, g \in Gens :
       (<<k, g>> \in inbox /\ <<k, g>> \notin inbox' /\ g < gen[k])
         => (status'[k] = status[k] /\ gen'[k] = gen[k])]_vars

ClearOnlyWhenGone ==
  [][\A k \in Keys :
       (status'[k] = "cleared" /\ status[k] # "cleared") => ~failing'[k]]_vars

(* Liveness. *)
EveryKeyEnds == \A k \in Keys : <>[](status[k] \in Done)
RecurrenceEscalates == \A k \in Keys : [](recurred[k] => <>(status[k] = "escalated"))

(* Reachability witnesses: each is violated when the named path exists. *)
WitExpiredBeforeEffect == \A k \in Keys : ~(status[k] = "expired" /\ eff[k] = "claimed")
WitExpiredAfterLostAck == \A k \in Keys : ~(status[k] = "expired" /\ eff[k] = "sent")
WitSingleGenClearThroughExpiry ==
  \A k \in Keys : ~(/\ status[k] = "cleared" /\ gen[k] = 1 /\ viaExpiry[k]
                    /\ eff[k] = "abandoned" /\ execs[k][1] = 1
                    /\ <<k, 1>> \notin delivered)
WitStaleDropped == ~staleDropped
WitSecondGen == \A k \in Keys : gen[k] < 2
WitBudgetEscalated == \A k \in Keys : ~(status[k] = "escalated" /\ ~recurred[k])
WitRecurrenceEscalated == \A k \in Keys : ~(status[k] = "escalated" /\ recurred[k])
WitDemotedCancel == \A k \in Keys : status[k] # "cancelled"
WitEverySubjectDispatched == ~(\A k \in Keys : claimCount[k][1] = 1)
=============================================================================
