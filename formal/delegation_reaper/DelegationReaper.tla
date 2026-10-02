---------------------------- MODULE DelegationReaper ----------------------------
(* The delegation reaper (OMN-19441): a delegate-skill command that holds a
   claim and has no terminal by its budget plus the reaper's grace gets exactly
   one typed terminal, cause no_terminal. The model is keyed per COMMAND (the
   delivering message id), not per correlation.

   Commands c1 and c2 share one correlation (a new command reusing a
   correlation). Workers w1 and w2 both deliver c1: w2 is a DLQ replay or a
   redelivery that KEEPS the command id. w3 and w4 deliver c2 the same way. A
   worker may hang or crash at any point before it hands its terminal to the
   wire, and a worker that finishes after the reaper has acted is the late real
   terminal.

   Shared state per key (the rows of the claim table):
     reapT     the reap-context row: the first claim time, insert-only, written
               BEFORE the claim row (a replay never moves it)
     claimRow  the claim row exists; its terminal copy is `copied`
     slot      the one terminal the command holds: NoSlot or <<kind, writer>>,
               written once by compare-and-set. Writer is the worker that wrote
               it or "reaper"; two real terminals from two workers are two
               different terminals
     copied    the winner's terminal has been handed to the wire
     evid      terminals that lost the slot, kept as attempt evidence
   pub is the set of <<command, kind, writer>> terminals on the wire. A
   republish of the same terminal has the same identity and adds nothing:
   terminal replay is not a second outcome. The model counts distinct terminal
   identities, not envelopes: two envelopes carrying the same terminal (the
   reaper healing a winner that is merely slow) are one terminal here.

   The correlation is only a stress input. Nothing reads it except the
   KeyByCorrelation mutation, which keys the rows on it and loses the second
   command's terminal.

   Properties:
     AtMostOneTerminal      at most one distinct terminal per command id
     WireMatchesRecord      every terminal on the wire is the recorded one
     SlotWriteOnce          a held terminal is never replaced (the reaper never
                            reaps a command that holds one, a handler timeout
                            terminal included)
     NoEarlyReap            no_terminal is written only at or past the deadline
     ClaimTimeFixed         the deadline is measured from the FIRST claim (an
                            implementation choice: a replay that restarted the
                            clock could only postpone the reaper)
     LateRealKeptAsEvidence a terminal that lost the slot is kept as evidence
     EveryClaimedAnswered   every claimed command, the second command on a
                            shared correlation included, ends with a terminal
                            of its own on the wire (liveness)
   Mutations (cfg constants) that must break them:
     AtomicReap=FALSE         reaper checks then writes in two steps -> SlotWriteOnce
     SlotCas=FALSE            handler record is a live upsert         -> AtMostOneTerminal
     GateOnWin=FALSE          a loser publishes its own terminal      -> WireMatchesRecord
     GateOnWin=FALSE, ReaperOn=FALSE  two handlers both publish       -> AtMostOneTerminal
     KeepEvidence=FALSE       a loser drops its terminal              -> LateRealKeptAsEvidence
     DeadlineGuard=FALSE      reaper ignores the deadline             -> NoEarlyReap
     FirstClaimDeadline=FALSE a replay claim restarts the deadline    -> ClaimTimeFixed
     ReaperOn=FALSE           no reaper                               -> EveryClaimedAnswered
     HealOn=FALSE             an orphaned handler slot is not healed  -> EveryClaimedAnswered
     ContextFirst=FALSE       claim row before the reap row           -> EveryClaimedAnswered
     KeyByCorrelation=TRUE    claim and slot keyed on the correlation -> EveryClaimedAnswered *)
EXTENDS Integers, FiniteSets
CONSTANTS Deadline, ClaimWindow, Horizon,
          AtomicReap, SlotCas, GateOnWin, KeepEvidence, DeadlineGuard,
          FirstClaimDeadline, ReaperOn, HealOn, ContextFirst, KeyByCorrelation

Cmds == {"c1", "c2"}
Workers == {"w1", "w2", "w3", "w4"}
Writers == Workers \cup {"reaper"}
CmdOf(w) == IF w \in {"w3", "w4"} THEN "c2" ELSE "c1"
Corr(c) == "k1"
Kinds == {"real", "timeout", "no_terminal"}
NoSlot == <<"none", "none">>
Key(c) == IF KeyByCorrelation THEN Corr(c) ELSE c
Keys == IF KeyByCorrelation THEN {"k1"} ELSE Cmds

VARIABLES now, pc, kindOf, reapT, claimRow, first, slot, copied, evid, pub, reapSeen
vars == <<now, pc, kindOf, reapT, claimRow, first, slot, copied, evid, pub, reapSeen>>

Init ==
  /\ now = 0
  /\ pc = [w \in Workers |-> "idle"]
  /\ kindOf = [w \in Workers |-> "none"]
  /\ reapT = [k \in Keys |-> -1]
  /\ claimRow = [k \in Keys |-> FALSE]
  /\ first = [k \in Keys |-> "none"]
  /\ slot = [k \in Keys |-> NoSlot]
  /\ copied = [k \in Keys |-> FALSE]
  /\ evid = [k \in Keys |-> {}]
  /\ pub = {}
  /\ reapSeen = [k \in Keys |-> FALSE]

Tick ==
  /\ now < Horizon
  /\ now' = now + 1
  /\ UNCHANGED <<pc, kindOf, reapT, claimRow, first, slot, copied, evid, pub, reapSeen>>

\* The two writes of a claim, in the order the handler performs them. The reap
\* row (insert-only: only the first writer sets the claim time) goes first, so a
\* worker that dies between the writes leaves a reap row and no claim row, which
\* nothing scans. A fresh command may only be claimed while the clock still has
\* room to reach its deadline.
Fresh(k) == reapT[k] = -1 /\ ~claimRow[k]
WriteReap(k) ==
  reapT' = IF reapT[k] = -1 \/ ~FirstClaimDeadline THEN [reapT EXCEPT ![k] = now] ELSE reapT
WriteClaim(k, w) ==
  /\ claimRow' = [claimRow EXCEPT ![k] = TRUE]
  /\ first' = IF first[k] = "none" THEN [first EXCEPT ![k] = CmdOf(w)] ELSE first

ClaimA(w) ==
  LET k == Key(CmdOf(w)) IN
  /\ pc[w] = "idle"
  /\ (~Fresh(k) \/ now <= ClaimWindow)
  /\ IF ContextFirst
       THEN /\ WriteReap(k) /\ UNCHANGED <<claimRow, first>>
       ELSE /\ WriteClaim(k, w) /\ UNCHANGED reapT
  /\ pc' = [pc EXCEPT ![w] = "mid"]
  /\ UNCHANGED <<now, kindOf, slot, copied, evid, pub, reapSeen>>

\* A delivery that finds a terminal already held is answered from it.
ClaimB(w) ==
  LET k == Key(CmdOf(w)) IN
  /\ pc[w] = "mid"
  /\ IF ContextFirst
       THEN /\ WriteClaim(k, w) /\ UNCHANGED reapT
       ELSE /\ WriteReap(k) /\ UNCHANGED <<claimRow, first>>
  /\ pc' = [pc EXCEPT ![w] = IF slot[k] # NoSlot THEN "served" ELSE "running"]
  /\ UNCHANGED <<now, kindOf, slot, copied, evid, pub, reapSeen>>

\* Answered from the held terminal: the same terminal, the same identity.
Serve(w) ==
  LET k == Key(CmdOf(w)) IN
  /\ pc[w] = "served"
  /\ pub' = pub \cup {<<first[k], slot[k][1], slot[k][2]>>}
  /\ copied' = [copied EXCEPT ![k] = TRUE]
  /\ pc' = [pc EXCEPT ![w] = "done"]
  /\ UNCHANGED <<now, kindOf, reapT, claimRow, first, slot, evid, reapSeen>>

\* The handler ends: its own budget cancel (timeout) or a real result.
Finish(w, kd) ==
  /\ pc[w] = "running"
  /\ pc' = [pc EXCEPT ![w] = "finished"]
  /\ kindOf' = [kindOf EXCEPT ![w] = kd]
  /\ UNCHANGED <<now, reapT, claimRow, first, slot, copied, evid, pub, reapSeen>>

\* The worker dies (crash, eviction, rebalance) or hangs; a hang is just never
\* taking another step. A crash may land between the claim writes, mid-run, or
\* between the slot write and the hand-off.
Crash(w) ==
  /\ pc[w] \in {"mid", "running", "finished", "recorded"}
  /\ pc' = [pc EXCEPT ![w] = "dead"]
  /\ UNCHANGED <<now, kindOf, reapT, claimRow, first, slot, copied, evid, pub, reapSeen>>

\* Compare-and-set of the terminal slot. The loser keeps its terminal as attempt
\* evidence and publishes nothing.
Record(w) ==
  LET k == Key(CmdOf(w)) IN
  /\ pc[w] = "finished"
  /\ IF SlotCas /\ slot[k] # NoSlot
       THEN /\ pc' = [pc EXCEPT ![w] = "lost"]
            /\ evid' = IF KeepEvidence
                         THEN [evid EXCEPT ![k] = @ \cup {<<kindOf[w], w>>}] ELSE evid
            /\ pub' = IF GateOnWin THEN pub ELSE pub \cup {<<first[k], kindOf[w], w>>}
            /\ UNCHANGED slot
       ELSE /\ slot' = [slot EXCEPT ![k] = <<kindOf[w], w>>]
            /\ pc' = [pc EXCEPT ![w] = "recorded"]
            /\ UNCHANGED <<evid, pub>>
  /\ UNCHANGED <<now, kindOf, reapT, claimRow, first, copied, reapSeen>>

\* The winner hands the recorded terminal to the wire and marks the claim.
Handoff(w) ==
  LET k == Key(CmdOf(w)) IN
  /\ pc[w] = "recorded"
  /\ pub' = pub \cup {<<first[k], slot[k][1], slot[k][2]>>}
  /\ copied' = [copied EXCEPT ![k] = TRUE]
  /\ pc' = [pc EXCEPT ![w] = "done"]
  /\ UNCHANGED <<now, kindOf, reapT, claimRow, first, slot, evid, reapSeen>>

\* Only a claim row with a reap row is scanned and only past the deadline: a
\* claim written without a reap row (a legacy claim) is never reaped.
Overdue(k) ==
  /\ claimRow[k] /\ reapT[k] # -1
  /\ (~DeadlineGuard \/ now - reapT[k] >= Deadline)

\* The reaper writes no_terminal only into an empty slot, atomically.
ReapCas(k) ==
  /\ ReaperOn /\ AtomicReap /\ Overdue(k) /\ slot[k] = NoSlot
  /\ slot' = [slot EXCEPT ![k] = <<"no_terminal", "reaper">>]
  /\ UNCHANGED <<now, pc, kindOf, reapT, claimRow, first, copied, evid, pub, reapSeen>>

\* Mutation: check in one step, write in a later one.
ReapCheck(k) ==
  /\ ReaperOn /\ ~AtomicReap /\ Overdue(k) /\ slot[k] = NoSlot /\ ~reapSeen[k]
  /\ reapSeen' = [reapSeen EXCEPT ![k] = TRUE]
  /\ UNCHANGED <<now, pc, kindOf, reapT, claimRow, first, slot, copied, evid, pub>>
ReapWrite(k) ==
  /\ ReaperOn /\ ~AtomicReap /\ reapSeen[k]
  /\ slot' = [slot EXCEPT ![k] = <<"no_terminal", "reaper">>]
  /\ reapSeen' = [reapSeen EXCEPT ![k] = FALSE]
  /\ UNCHANGED <<now, pc, kindOf, reapT, claimRow, first, copied, evid, pub>>

\* A held terminal that was never handed to the wire (its writer died between
\* the slot write and the hand-off) is published by the reaper. Same identity as
\* the writer's own hand-off, so a double publish is not a second terminal. The
\* reaper's own terminal is always published (HealOn only gates the heal of a
\* terminal a handler wrote).
ReapHeal(k) ==
  /\ ReaperOn /\ (HealOn \/ slot[k][2] = "reaper") /\ Overdue(k)
  /\ slot[k] # NoSlot /\ ~copied[k]
  /\ pub' = pub \cup {<<first[k], slot[k][1], slot[k][2]>>}
  /\ copied' = [copied EXCEPT ![k] = TRUE]
  /\ UNCHANGED <<now, pc, kindOf, reapT, claimRow, first, slot, evid, reapSeen>>

ReapActs(k) == ReapCas(k) \/ ReapCheck(k) \/ ReapWrite(k) \/ ReapHeal(k)

Next ==
  \/ Tick
  \/ \E w \in Workers :
       ClaimA(w) \/ ClaimB(w) \/ Serve(w) \/ Crash(w) \/ Record(w) \/ Handoff(w)
  \/ \E w \in Workers, kd \in {"real", "timeout"} : Finish(w, kd)
  \/ \E k \in Keys : ReapActs(k)

Spec == Init /\ [][Next]_vars /\ WF_vars(Tick)
        /\ \A k \in Keys : WF_vars(ReapActs(k))

Claimed(c) == claimRow[Key(c)] /\ \E w \in Workers : CmdOf(w) = c /\ pc[w] # "idle"
Terminals(c) == {t \in Kinds \X Writers : <<c, t[1], t[2]>> \in pub}
Answered(c) == Terminals(c) # {}

AtMostOneTerminal == \A c \in Cmds : Cardinality(Terminals(c)) <= 1
WireMatchesRecord ==
  \A p \in pub : \E k \in Keys : first[k] = p[1] /\ slot[k] = <<p[2], p[3]>>
LateRealKeptAsEvidence ==
  \A w \in Workers : pc[w] = "lost" => <<kindOf[w], w>> \in evid[Key(CmdOf(w))]
SlotWriteOnce ==
  [][\A k \in Keys : slot[k] # NoSlot => slot'[k] = slot[k]]_vars
NoEarlyReap ==
  [][\A k \in Keys : (slot[k] = NoSlot /\ slot'[k][1] = "no_terminal")
                       => now - reapT[k] >= Deadline]_vars
ClaimTimeFixed ==
  [][\A k \in Keys : reapT[k] # -1 => reapT'[k] = reapT[k]]_vars
EveryClaimedAnswered == \A c \in Cmds : Claimed(c) ~> Answered(c)

\* Reachability witnesses: each must be VIOLATED, which proves the state exists.
WitReaped == ~(<<"c1", "no_terminal", "reaper">> \in pub)
WitLateRealEvidence ==
  ~(slot["c1"] = <<"no_terminal", "reaper">> /\ \E w \in Workers : <<"real", w>> \in evid["c1"])
WitTimeoutHeldPastDeadline ==
  ~(\E w \in Workers : slot["c1"] = <<"timeout", w>> /\ now - reapT["c1"] >= Deadline)
WitReplayServed == ~(pc["w2"] = "served")
WitSharedCorrelationBothAnswered == ~(Answered("c1") /\ Answered("c2"))
\* Only the reaper can have published this: w1 died holding the slot, and the
\* replay w2 never delivered.
WitHealedOrphan ==
  ~(pc["w1"] = "dead" /\ slot["c1"] = <<"real", "w1">>
    /\ <<"c1", "real", "w1">> \in pub /\ pc["w2"] = "idle")
\* Two real terminals from two different workers of one command: the case a
\* kind-only identity could not tell apart.
WitTwoRealTerminals == ~(\E w \in Workers : pc[w] = "lost" /\ kindOf[w] = "real" /\ slot["c1"][1] = "real" /\ slot["c1"] # <<"real", w>>)
================================================================================
