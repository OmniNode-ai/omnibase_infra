-------------------------- MODULE RebuildAdmission --------------------------
(***************************************************************************)
(* M0 of the subsystem rebuild plan (OMN-22478, plan section 12): the stop *)
(* gate, WIP admission and the holder, as one state machine. Sections 7.1  *)
(* (the admission table) and 7.5 (the holder and its fallback) plus W28    *)
(* (the single-use start reservation) are what it states. It implements    *)
(* and wires nothing. README.txt gives the writer inventory that fixes the *)
(* bounds, the property table and the mutation per property.               *)
(*                                                                         *)
(* A design knob is a CONSTANT. Base.cfg sets every knob to the protocol   *)
(* that the four properties need; PlanAsWritten_*.cfg set one knob each to *)
(* what the plan's text says today; mut_*.cfg break one thing each.        *)
(*                                                                         *)
(* Abstractions, stated:                                                   *)
(* - There is no epoch number. An advance turns every live reservation     *)
(*   into a dead one ("rold"), which is what comparing numbers decides.    *)
(* - No history counters: stops, safety failures, crashes and lanes may    *)
(*   recur without bound, because the state space stays finite.            *)
(* - WORKING for 24 hours is one published reading.                        *)
(* - Time: a stale table, an aged stop and an expired reservation are      *)
(*   actions; weak fairness says they happen.                              *)
(***************************************************************************)
EXTENDS Integers, FiniteSets, TLC

CONSTANTS
  o1, o2, o3, o4, o5, o6, o7, o8, o9,   \* see Ords
  Ords,           \* model values: the concurrent ordinary tasks of piece 1 (a symmetry set)
  Others,         \* the other start requests by name, each carried by one start path
  SafetyIds,      \* safety failures that can be open together
  Limit,          \* the order overlay's WIP limit per piece
  Seven,          \* the piece that step 2 calls piece 7
  MaxPiece,       \* pieces are 1..MaxPiece, bottom up
  NeedTyped,      \* the stop's release also needs a typed record (friction)
  CrashOn,        \* holder engines crash, recover and hand over
  TrackCleared,   \* the WitStopCleared witness only: remember that a stop cleared
  BreakPieces,    \* pieces a stop can open on
  PubDieOn,       \* the reading writer can die
  \* design knobs
  CountResv,      \* the WIP check counts live reservations as well as claim rows
  GateBumps,      \* every write to the gate's state advances the epoch, not only an advance
  EpochGuard,     \* consuming a reservation checks the epoch (W28)
  SingleUse,      \* a consumed reservation cannot be consumed again (W28)
  TableOrder,     \* "design", "stale_first", "stop_first", "cont_in_safety", "no_repair_bound"
  SafetyRepair,   \* a stale table during a safety failure admits the bounded repair lane
  ContFence,      \* each step of a continuation lane checks for an open safety failure
  SuspendOn,      \* running continuation lanes are suspended on a safety failure
  ClearGuard,     \* "all": rows and reading; "reading": no rows; "rows": no fresh reading
  LeaseResetsSee, \* a lease started after the go-and-see row voids the row (6.2)
  EscalateOn,     \* an aged stop writes one decision row to the operator
  KeyedSteps,     \* steps are keyed run_key/step (7.5)
  SharedKeys,     \* holder and fallback read one key store
  Fence,          \* a write by an engine that lost the order is refused
  HandoverBumps,  \* a handover advances the epoch
  RestartOn,      \* a supervisor restarts a crashed engine
  JoinCount       \* "same": gate and reading count a join in one piece; "split": they differ

Tasks == Ords \cup Others
Procs == {"H", "F"}             \* H: the engine holding the order; F: the fallback
Pieces == 1..MaxPiece
NSteps == MaxPiece - 1          \* step i advances the order from piece i to i + 1

\* The start requests and their kinds. A piece of 0 means none.
KindOf(t) ==
  CASE t \in Ords -> [k |-> "ord", p |-> 1, s |-> "-"]
    [] t = "j"  -> [k |-> "join", p |-> 2, s |-> "-"]
    [] t = "j3" -> [k |-> "join", p |-> 3, s |-> "-"]
    [] t = "u"  -> [k |-> "ord", p |-> 2, s |-> "-"]
    [] t = "c1" -> [k |-> "cause", p |-> 1, s |-> "-"]
    [] t = "c2" -> [k |-> "cause", p |-> 2, s |-> "-"]
    [] t = "c3" -> [k |-> "cause", p |-> MaxPiece, s |-> "-"]
    [] t = "kf" -> [k |-> "kept", p |-> MaxPiece, s |-> "-"]
    [] t = "g"  -> [k |-> "cont", p |-> 0, s |-> "-"]
    [] t \in {"r1", "r2"} -> [k |-> "repair", p |-> 0, s |-> "-"]
    [] t = "s1" -> [k |-> "scause", p |-> MaxPiece, s |-> "s1"]
    [] t = "s2" -> [k |-> "scause", p |-> MaxPiece, s |-> "s2"]

\* Every kind of the table, including those no reachable behaviour depends on: the
\* decision itself is checked for each of them in every reachable state.
AllKinds ==
  [k : {"ord", "join", "cause", "kept"}, p : Pieces, s : {"-"}]
  \cup [k : {"ro", "cont", "repair", "b0", "other"}, p : {0}, s : {"-"}]
  \cup [k : {"scause"}, p : {MaxPiece}, s : SafetyIds]

VARIABLES
  pc,        \* [Tasks -> idle, resv, rold, run, susp]
  consumed,  \* [Tasks -> 0..2]: consumptions of the task's current reservation
  safety,    \* open safety failures
  sFixed,    \* those whose cause lane has done its work
  fresh,     \* the reading table is inside 30 minutes
  pubUp,     \* the reading writer is alive
  broken,    \* pieces truly not working
  working,   \* pieces the table last published as WORKING
  stop,      \* [Pieces -> none, open, esc]: esc is an open stop with its decision row
  rows,      \* [Pieces -> SUBSET {"see", "typed"}]: evidence rows
  cur,       \* the lowest unfinished piece
  joined,    \* pieces whose join test has passed
  up,        \* [Procs -> BOOLEAN]
  handed,    \* the fallback holds the order
  busy,      \* [Procs -> 0..NSteps]: step whose row is written and not yet checkpointed
  ckpt,      \* [Procs -> 0..NSteps]: the engine's own checkpoint
  keys,      \* [Procs -> SUBSET 1..NSteps]: step keys, H's copy is the one when shared
  effects,   \* [1..NSteps -> 0..2]: times each step's effect ran
  cleared,   \* ghost: some stop has cleared
  seeAfter,  \* ghost: [Pieces -> a go-and-see row was written after the last lease]
  oldResv,   \* ghost: reservations that outlived a handover (empty unless a mutation)
  badStart, staleCons, contStep, badClear, clash   \* ghosts: the safety properties' witnesses

gateVars == <<safety, sFixed, fresh, pubUp, broken, working, stop, rows, cur, joined>>
engVars  == <<up, handed, busy, ckpt, keys, effects>>
ghost    == <<cleared, seeAfter, oldResv, badStart, staleCons, contStep, badClear, clash>>
vars     == <<pc, consumed, gateVars, engVars, ghost>>

K(t) == KindOf(t)
\* A cause lane is class rebuild (7.1). A safety cause lane may sit above piece 7: step 1 must
\* take it before step 2 would refuse it, which is the M0 row's stale-table case.
Rebuild(k) == k.k \in {"ord", "join", "cause", "scause"}

Stops == {p \in Pieces : stop[p] # "none"}
Min(S) == CHOOSE x \in S : \A y \in S : x <= y
Low == Min(Stops)

Owner == IF handed THEN "F" ELSE "H"
HolderOK == up[Owner]
\* The engines whose order writes the record would accept.
\* H keeps believing it holds the order after a handover it has not seen; only the fence refuses it.
Accepts(x) == IF Fence THEN x = Owner ELSE (x = "H" \/ handed)

Working(p) == fresh /\ p \in working
Pred(p) == p = 1 \/ Working(p - 1)
Joined(p) == p = 1 \/ p \in joined

AdmitPiece(k) == k.p
ReadPiece(k) == IF k.k = "join" /\ JoinCount = "split" THEN k.p - 1 ELSE k.p
Wip(k) == k.k \in {"ord", "join"}
Claims(p) == Cardinality({t \in Tasks : Wip(K(t)) /\ ReadPiece(K(t)) = p /\ pc[t] \in {"run", "susp"}})
GateCount(p) ==
  Cardinality({t \in Tasks : Wip(K(t)) /\ AdmitPiece(K(t)) = p
                 /\ pc[t] \in (IF CountResv THEN {"resv", "run", "susp"} ELSE {"run", "susp"})})
Room(p) == GateCount(p) < Limit

\* One reserved cause slot per piece, one bounded repair lane (W5, step 2).
Slot(k) ==
  \A t \in Tasks : (K(t).k = k.k /\ K(t).p = k.p /\ K(t).s = k.s) => pc[t] \notin {"resv", "run", "susp"}
RepairFree ==
  TableOrder = "no_repair_bound" \/ \A t \in Tasks : K(t).k = "repair" => pc[t] \notin {"resv", "run"}

\* ---- the table of 7.1, each step as its own decision ("next" passes on) ----
Step1(k) ==
  IF safety = {} THEN "next"
  ELSE IF k.k = "scause" /\ k.s \in safety THEN "admit"
  ELSE IF SafetyRepair /\ k.k = "repair" /\ ~fresh /\ RepairFree THEN "admit"
  ELSE "refuse"
Step2(k) ==
  IF fresh THEN "next"
  ELSE IF k.k = "repair" THEN (IF RepairFree THEN "admit" ELSE "refuse")
  ELSE IF k.k = "kept" \/ (Rebuild(k) /\ k.p > Seven) THEN "refuse"
  ELSE "next"
OrdOK(k) == k.p = cur /\ Pred(k.p) /\ Joined(k.p) /\ Room(k.p)
JoinOK(k) == k.p = cur + 1 /\ k.p \notin joined /\ Pred(k.p) /\ Room(k.p)
Step3(k) ==
  IF Stops = {} THEN "next"
  ELSE CASE k.k = "cause" -> IF k.p = Low /\ Slot(k) THEN "admit" ELSE "refuse"
         [] k.k = "ord"   -> IF k.p < Low /\ OrdOK(k) THEN "admit" ELSE "refuse"
         [] k.k = "join"  -> IF k.p < Low /\ JoinOK(k) THEN "admit" ELSE "refuse"
         [] k.k \in {"b0", "cont"} -> "admit"
         [] OTHER -> "refuse"
StepOrder(k) ==
  CASE k.k = "ord"  -> IF OrdOK(k) THEN "admit" ELSE "refuse"
    [] k.k = "join" -> IF JoinOK(k) THEN "admit" ELSE "refuse"
    [] k.k \in {"b0", "kept", "cont"} -> "admit"
    [] OTHER -> "refuse"
First(a, b, c, d) ==
  IF a # "next" THEN a ELSE IF b # "next" THEN b ELSE IF c # "next" THEN c ELSE d
Decide(k) ==
  IF k.k = "ro" THEN "admit"
  ELSE CASE TableOrder \in {"design", "no_repair_bound"} -> First(Step1(k), Step2(k), Step3(k), StepOrder(k))
         [] TableOrder = "stale_first" -> First(Step2(k), Step1(k), Step3(k), StepOrder(k))
         [] TableOrder = "stop_first"  -> First(Step1(k), Step3(k), Step2(k), StepOrder(k))
         [] TableOrder = "cont_in_safety" ->
              IF k.k = "cont" THEN "admit" ELSE First(Step1(k), Step2(k), Step3(k), StepOrder(k))

\* ---- what the order forbids, written from the property and not from the table ----
\* WIP room is not here: property 2 owns it.
Against(k) ==
  IF k.k = "ro" THEN FALSE
  ELSE IF safety # {} THEN
       ~((k.k = "scause" /\ k.s \in safety) \/ (SafetyRepair /\ k.k = "repair" /\ ~fresh))
  ELSE IF ~fresh /\ (k.k = "kept" \/ (Rebuild(k) /\ k.p > Seven)) THEN TRUE
  ELSE IF ~fresh /\ k.k = "repair" THEN FALSE
  ELSE IF Stops # {} THEN
       \/ k.k \in {"kept", "other", "repair", "scause"}
       \/ (k.k = "cause" /\ k.p # Low)
       \/ (k.k \in {"ord", "join"} /\ k.p >= Low)
       \/ (k.k = "ord" /\ k.p # cur)
  ELSE \/ k.k \in {"cause", "other", "repair", "scause"}
       \/ (k.k = "ord" /\ k.p # cur)

\* ---- the epoch ----
\* With the guard on, a dead reservation can only be refused and released, so it goes
\* straight back; the rold state exists only for the mutation that consumes one.
Kill == [t \in Tasks |-> IF pc[t] = "resv" THEN (IF EpochGuard THEN "idle" ELSE "rold") ELSE pc[t]]
GateKill == IF GateBumps THEN Kill ELSE pc

Init ==
  /\ pc = [t \in Tasks |-> "idle"] /\ consumed = [t \in Tasks |-> 0]
  /\ safety = {} /\ sFixed = {} /\ fresh = TRUE /\ pubUp = TRUE
  /\ broken = {} /\ working = Pieces
  /\ stop = [p \in Pieces |-> "none"] /\ rows = [p \in Pieces |-> {}]
  /\ cur = 1 /\ joined = {}
  /\ up = [x \in Procs |-> TRUE] /\ handed = FALSE
  /\ busy = [x \in Procs |-> 0] /\ ckpt = [x \in Procs |-> 0]
  /\ keys = [x \in Procs |-> {}] /\ effects = [i \in 1..NSteps |-> 0]
  /\ cleared = FALSE /\ seeAfter = [p \in Pieces |-> FALSE] /\ oldResv = {}
  /\ badStart = FALSE /\ staleCons = FALSE /\ contStep = FALSE /\ badClear = FALSE /\ clash = FALSE

\* ---- the admission path: reserve, then consume at the last step before an engine call ----
Reserve(t) ==
  /\ pc[t] = "idle"
  /\ HolderOK
  /\ Decide(K(t)) = "admit"
  /\ pc' = [pc EXCEPT ![t] = "resv"]
  /\ UNCHANGED <<consumed, gateVars, engVars, ghost>>

Release(t) ==   \* an expired, dead or refused reservation goes back; the path may ask again
  /\ pc[t] \in {"resv", "rold"}
  /\ pc' = [pc EXCEPT ![t] = "idle"]
  /\ oldResv' = oldResv \ {t}
  /\ UNCHANGED <<consumed, gateVars, engVars, cleared, seeAfter, badStart, staleCons, contStep, badClear, clash>>

Consume(t) ==
  /\ pc[t] \in (IF EpochGuard THEN {"resv"} ELSE {"resv", "rold"})
  /\ HolderOK
  /\ pc' = [pc EXCEPT ![t] = "run"]
  /\ consumed' = [consumed EXCEPT ![t] = 1]
  /\ staleCons' = (staleCons \/ pc[t] = "rold" \/ t \in oldResv)
  /\ oldResv' = oldResv \ {t}
  /\ badStart' = (badStart \/ Against(K(t)))
  /\ rows' = IF LeaseResetsSee /\ K(t).k = "cause"
             THEN [rows EXCEPT ![K(t).p] = @ \ {"see"}] ELSE rows
  /\ seeAfter' = IF K(t).k = "cause" THEN [seeAfter EXCEPT ![K(t).p] = FALSE] ELSE seeAfter
  /\ UNCHANGED <<safety, sFixed, fresh, pubUp, broken, working, stop, cur, joined,
                 engVars, cleared, contStep, badClear, clash>>

DupConsume(t) ==   \* the bus redelivers the start, or a restarted path asks again
  /\ ~SingleUse /\ pc[t] = "run" /\ consumed[t] = 1
  /\ consumed' = [consumed EXCEPT ![t] = 2]
  /\ UNCHANGED <<pc, gateVars, engVars, ghost>>

Finish(t) ==       \* the lane's TERMINAL row; a repair or a join lane leaves its result
  /\ pc[t] = "run"
  /\ pc' = [pc EXCEPT ![t] = "idle"]
  /\ consumed' = [consumed EXCEPT ![t] = 0]
  /\ pubUp' = (pubUp \/ K(t).k = "repair")
  /\ joined' = IF K(t).k = "join" THEN joined \cup {K(t).p} ELSE joined
  /\ UNCHANGED <<safety, sFixed, fresh, broken, working, stop, rows, cur,
                 engVars, ghost>>

\* ---- a running lane's work ----
Fix(t) ==          \* nothing assumes a cause lane ever succeeds
  /\ pc[t] = "run" /\ K(t).k = "cause" /\ K(t).p \in broken
  /\ broken' = broken \ {K(t).p}
  /\ UNCHANGED <<pc, consumed, safety, sFixed, fresh, pubUp, working, stop, rows, cur, joined,
                 engVars, ghost>>
SafetyFix(t) ==
  /\ pc[t] = "run" /\ K(t).k = "scause" /\ K(t).s \in safety /\ K(t).s \notin sFixed
  /\ sFixed' = sFixed \cup {K(t).s}
  /\ UNCHANGED <<pc, consumed, safety, fresh, pubUp, broken, working, stop, rows, cur, joined,
                 engVars, ghost>>
ContStep(t) ==     \* one step of a running continuation lane
  /\ pc[t] = "run" /\ K(t).k = "cont"
  /\ (ContFence => safety = {})
  /\ safety # {}   \* a step with no failure open changes nothing the model reads
  /\ contStep' = TRUE
  /\ UNCHANGED <<pc, consumed, gateVars, engVars, cleared, seeAfter, oldResv, badStart, staleCons, badClear, clash>>
Suspend(t) ==
  /\ SuspendOn /\ pc[t] = "run" /\ K(t).k = "cont" /\ safety # {}
  /\ pc' = [pc EXCEPT ![t] = "susp"]
  /\ UNCHANGED <<consumed, gateVars, engVars, ghost>>
Resume(t) ==
  /\ pc[t] = "susp" /\ safety = {}
  /\ pc' = [pc EXCEPT ![t] = "run"]
  /\ UNCHANGED <<consumed, gateVars, engVars, ghost>>

\* ---- safety failures: written from the sources, independently of the table ----
SafetyOpens(s) ==
  /\ s \notin safety
  /\ safety' = safety \cup {s} /\ sFixed' = sFixed \ {s}
  /\ pc' = GateKill
  /\ UNCHANGED <<consumed, fresh, pubUp, broken, working, stop, rows, cur, joined, engVars, ghost>>
SafetyCloses(s) ==   \* "until its cause lane reads WORKING": a stale table reads nothing
  /\ s \in safety /\ s \in sFixed /\ fresh
  /\ safety' = safety \ {s} /\ sFixed' = sFixed \ {s}
  /\ pc' = GateKill
  /\ UNCHANGED <<consumed, fresh, pubUp, broken, working, stop, rows, cur, joined, engVars, ghost>>

\* ---- the reading writer and the table ----
PubDies ==
  /\ PubDieOn /\ pubUp /\ pubUp' = FALSE
  /\ UNCHANGED <<pc, consumed, safety, sFixed, fresh, broken, working, stop, rows, cur, joined, engVars, ghost>>
GoesStale ==
  /\ fresh /\ ~pubUp /\ fresh' = FALSE
  /\ pc' = GateKill
  /\ UNCHANGED <<consumed, safety, sFixed, pubUp, broken, working, stop, rows, cur, joined, engVars, ghost>>
Publish ==
  /\ pubUp
  /\ working' = Pieces \ broken /\ fresh' = TRUE
  /\ (fresh' # fresh \/ working' # working)
  /\ pc' = GateKill
  /\ UNCHANGED <<consumed, safety, sFixed, pubUp, broken, stop, rows, cur, joined, engVars, ghost>>

\* ---- the stop ----
Breaks(p) ==   \* two failing probes, a stale read, an andon trigger, a return
  /\ p \in BreakPieces /\ stop[p] = "none"
  /\ stop' = [stop EXCEPT ![p] = "open"] /\ broken' = broken \cup {p}
  /\ rows' = [rows EXCEPT ![p] = {}]
  /\ seeAfter' = [seeAfter EXCEPT ![p] = FALSE]
  /\ pc' = GateKill
  /\ UNCHANGED <<consumed, safety, sFixed, fresh, pubUp, working, cur, joined, engVars,
                 cleared, oldResv, badStart, staleCons, contStep, badClear, clash>>
EvidenceRow(p, r) ==   \* go-and-see is a read-only lane: it needs no admission
  /\ stop[p] # "none" /\ r \notin rows[p]
  /\ (r = "typed" => NeedTyped)
  /\ rows' = [rows EXCEPT ![p] = @ \cup {r}]
  /\ seeAfter' = IF r = "see" THEN [seeAfter EXCEPT ![p] = TRUE] ELSE seeAfter
  /\ UNCHANGED <<pc, consumed, safety, sFixed, fresh, pubUp, broken, working, stop, cur, joined, engVars,
                 cleared, oldResv, badStart, staleCons, contStep, badClear, clash>>
Required == IF NeedTyped THEN {"see", "typed"} ELSE {"see"}
Evidence(p) == Required \subseteq rows[p]
Clears(p) ==
  /\ stop[p] # "none"
  /\ CASE ClearGuard = "all"     -> Evidence(p) /\ Working(p)
       [] ClearGuard = "reading" -> Working(p)
       [] ClearGuard = "rows"    -> Evidence(p)
  /\ stop' = [stop EXCEPT ![p] = "none"]
  /\ rows' = [rows EXCEPT ![p] = {}]
  /\ cleared' = TrackCleared
  /\ badClear' = (badClear \/ ~(Evidence(p) /\ Working(p) /\ seeAfter[p]))
  /\ seeAfter' = [seeAfter EXCEPT ![p] = FALSE]
  /\ pc' = GateKill
  /\ UNCHANGED <<consumed, safety, sFixed, fresh, pubUp, broken, working, cur, joined, engVars,
                 oldResv, badStart, staleCons, contStep, clash>>
Escalates(p) ==
  /\ EscalateOn /\ stop[p] = "open"
  /\ stop' = [stop EXCEPT ![p] = "esc"]
  /\ UNCHANGED <<pc, consumed, safety, sFixed, fresh, pubUp, broken, working, rows, cur, joined, engVars, ghost>>

\* ---- the engines: holder, fallback, restart and the keyed steps ----
KeyStore(x) == IF SharedKeys THEN "H" ELSE x
Ready(i) == cur = i /\ Working(i) /\ (i + 1) \in joined /\ stop[i] = "none" /\ safety = {}
StepWrite(x) ==   \* writes the step's row; the effect of step i is the order's advance
  LET i == ckpt[x] + 1 IN
  /\ up[x] /\ busy[x] = 0 /\ i <= NSteps /\ Accepts(x)
  /\ (cur > i \/ Ready(i))
  /\ busy' = [busy EXCEPT ![x] = i]
  /\ clash' = (clash \/ x # Owner)
  /\ IF KeyedSteps /\ i \in keys[KeyStore(x)]
     THEN UNCHANGED <<effects, keys, cur, pc>>
     ELSE /\ effects' = [effects EXCEPT ![i] = IF @ < 2 THEN @ + 1 ELSE @]
          /\ keys' = [keys EXCEPT ![KeyStore(x)] = @ \cup {i}]
          /\ cur' = IF cur = i THEN i + 1 ELSE cur
          /\ pc' = IF cur = i THEN Kill ELSE pc
  /\ UNCHANGED <<consumed, safety, sFixed, fresh, pubUp, broken, working, stop, rows, joined,
                 up, handed, ckpt, cleared, seeAfter, oldResv, badStart, staleCons, contStep, badClear>>
Checkpoint(x) ==
  /\ up[x] /\ busy[x] # 0
  /\ ckpt' = [ckpt EXCEPT ![x] = busy[x]] /\ busy' = [busy EXCEPT ![x] = 0]
  /\ UNCHANGED <<pc, consumed, gateVars, up, handed, keys, effects, ghost>>
Crash(x) ==   \* a step in hand is lost; the row it wrote is not
  /\ CrashOn /\ up[x]
  /\ up' = [up EXCEPT ![x] = FALSE] /\ busy' = [busy EXCEPT ![x] = 0]
  /\ UNCHANGED <<pc, consumed, gateVars, handed, ckpt, keys, effects, ghost>>
Recover(x) ==
  /\ CrashOn /\ RestartOn /\ ~up[x]
  /\ up' = [up EXCEPT ![x] = TRUE]
  /\ UNCHANGED <<pc, consumed, gateVars, handed, busy, ckpt, keys, effects, ghost>>
Handover ==   \* stage A fails, or no read at the stop date: one way, fenced; H may not know
  /\ CrashOn /\ ~handed
  /\ handed' = TRUE
  /\ pc' = IF HandoverBumps THEN Kill ELSE pc
  /\ oldResv' = IF HandoverBumps THEN {} ELSE {t \in Tasks : pc[t] = "resv"}
  /\ UNCHANGED <<consumed, gateVars, up, busy, ckpt, keys, effects,
                 cleared, seeAfter, badStart, staleCons, contStep, badClear, clash>>

Next ==
  \/ \E t \in Tasks : Reserve(t) \/ Release(t) \/ Consume(t) \/ DupConsume(t) \/ Finish(t)
                      \/ Fix(t) \/ SafetyFix(t) \/ ContStep(t) \/ Suspend(t) \/ Resume(t)
  \/ \E s \in SafetyIds : SafetyOpens(s) \/ SafetyCloses(s)
  \/ PubDies \/ GoesStale \/ Publish
  \/ \E p \in Pieces : Breaks(p) \/ Clears(p) \/ Escalates(p)
                       \/ \E r \in {"see", "typed"} : EvidenceRow(p, r)
  \/ \E x \in Procs : StepWrite(x) \/ Checkpoint(x) \/ Crash(x) \/ Recover(x)
  \/ Handover

\* Fairness, stated, used only by liveness. Time passes (a reservation expires, a dead
\* publisher's table goes stale, an aged stop writes its decision row, a suspension sweep
\* runs, a crashed engine is restarted by its supervisor). A reading writer that keeps coming
\* back publishes at some point (strong fairness: one that dies before every publish forever
\* is not assumed away by the plan; it is a stop that escalates). A bounded repair lane that is admitted finishes, and is retried while a gate write
\* kills its reservation; a close of a safety failure whose cause reads WORKING happens if it
\* is possible infinitely often. Nothing is assumed about a cause lane fixing a stop, about any
\* other admission, or about an operator.
Fairness ==
  /\ \A t \in Tasks : WF_vars(Release(t)) /\ WF_vars(Suspend(t)) /\ WF_vars(Resume(t))
  /\ \A t \in Tasks : WF_vars(Reserve(t)) /\ WF_vars(Consume(t)) /\ WF_vars(SafetyFix(t))
  /\ \A p \in Pieces : WF_vars(Escalates(p))
  /\ \A s \in SafetyIds : SF_vars(SafetyCloses(s))
  /\ \A t \in Tasks : K(t).k = "repair" => (SF_vars(Reserve(t)) /\ SF_vars(Consume(t)) /\ WF_vars(Finish(t)))
  /\ SF_vars(Publish) /\ WF_vars(GoesStale)
  /\ \A x \in Procs : WF_vars(Recover(x))

Spec == Init /\ [][Next]_vars
LiveSpec == Spec /\ Fairness

(***************************************************************************)
(* Property 1: admission follows the order of 7.1.                         *)
(***************************************************************************)
\* The table never admits what the order forbids, for every kind, in every reachable state.
DecideFollowsOrder == \A k \in AllKinds : Decide(k) = "admit" => ~Against(k)
\* No send starts against the order as it stands when the start happens.
NoStartAgainstOrder == ~badStart
\* A stale table with a safety-failure cause lane: the lane is admitted, on the sources alone.
SafetyCauseAdmitted == \A k \in AllKinds : (k.k = "scause" /\ k.s \in safety) => Decide(k) = "admit"
\* Read-only measurement passes in every state (step 0).
ReadOnlyAdmitted == Decide([k |-> "ro", p |-> 0, s |-> "-"]) = "admit"
\* No continuation runs during a safety failure: admission and each step of a running lane.
NoContinuationAdmittedInSafety == safety # {} => Decide([k |-> "cont", p |-> 0, s |-> "-"]) = "refuse"
NoContinuationStepInSafety == ~contStep
\* One bounded repair lane.
OneRepairLane == Cardinality({t \in Tasks : K(t).k = "repair" /\ pc[t] \in {"resv", "run"}}) <= 1

(***************************************************************************)
(* Property 2: claims per piece stay within the limit under concurrent     *)
(* admission. Claims(p) is what the independent reading counts.            *)
(***************************************************************************)
ClaimsWithinLimit == \A p \in Pieces : Claims(p) <= Limit

(***************************************************************************)
(* Property 3: a stop clears only with its evidence rows, and eventually   *)
(* clears or escalates.                                                    *)
(***************************************************************************)
StopClearsOnlyWithEvidence == ~badClear
StopClearsOrEscalates == \A p \in Pieces : (stop[p] = "open") ~> (stop[p] # "open")
RunningContinuationSuspended ==
  \A t \in Tasks : (pc[t] = "run" /\ K(t).k = "cont" /\ safety # {}) ~> (pc[t] # "run" \/ safety = {})

(***************************************************************************)
(* Property 4: a restart repeats no step; exactly one of holder and        *)
(* fallback owns the order; a reservation is consumed at most once and     *)
(* never after the epoch advanced.                                         *)
(***************************************************************************)
RestartRepeatsNoStep == \A i \in 1..NSteps : effects[i] <= 1
ExactlyOneOwner == Cardinality({x \in Procs : Accepts(x)}) = 1
NoWriteByLostOwner == ~clash
ConsumedAtMostOnce == \A t \in Tasks : consumed[t] <= 1
NeverConsumedAfterEpoch == ~staleCons
OwnerComesBack == (~up[Owner]) ~> up[Owner]

(***************************************************************************)
(* Not one of the four: a safety failure whose cause is fixed closes.      *)
(* Its counterexample is finding F4 in README.txt.                         *)
(***************************************************************************)
FixedSafetyCloses == \A s \in SafetyIds : (s \in sFixed) ~> (s \notin safety)

(***************************************************************************)
(* Reachability witnesses: each is expected to be violated, so a pass is   *)
(* not a vacuous one.                                                      *)
(***************************************************************************)
Has(t) == t \in Tasks
WitOrderReachesLastPiece == cur < MaxPiece
WitLimitReached == Claims(1) < Limit
WitSafetyCauseStartsOnStaleTable == ~(Has("s1") /\ ~fresh /\ safety # {} /\ pc["s1"] = "run")
WitRepairDuringSafety == ~(Has("r1") /\ ~fresh /\ safety # {} /\ pc["r1"] = "run")
WitLowerPieceRunsUnderStop == ~(\E t \in Ords : pc[t] = "run" /\ Stops # {} /\ Low > 1)
WitTwoStopsCauseOfLowest == ~(Has("c2") /\ Has("c3") /\ stop[2] # "none" /\ stop[3] # "none" /\ pc["c2"] = "run")
WitContinuationSuspended == ~(Has("g") /\ pc["g"] = "susp")
WitStopCleared == ~(\E p \in Pieces : stop[p] = "none" /\ p \in working /\ rows[p] = {} /\ seeAfter[p] = FALSE /\ broken = {} /\ cleared)
WitEscalated == \A p \in Pieces : stop[p] # "esc"
WitHandoverStepByFallback == ~(handed /\ cur > 1 /\ ckpt["F"] > 0)

OrdSymmetry == Permutations(Ords)

TypeOK ==
  /\ pc \in [Tasks -> {"idle", "resv", "rold", "run", "susp"}]
  /\ cur \in 1..(MaxPiece + 1)
  /\ safety \subseteq SafetyIds
  /\ stop \in [Pieces -> {"none", "open", "esc"}]
=============================================================================
