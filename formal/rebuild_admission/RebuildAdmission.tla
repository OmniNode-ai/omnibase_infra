-------------------------- MODULE RebuildAdmission --------------------------
(***************************************************************************)
(* M0 of the subsystem rebuild plan (OMN-22478, plan section 12), round 2: *)
(* the stop gate, WIP admission and the holder as one state machine,       *)
(* sections 6.2, 7.1, 7.3 and 7.5 and W28. It implements and wires nothing.*)
(*                                                                         *)
(* One protocol. There are no design knobs: Mut = {} is the protocol, and  *)
(* each name in Mut breaks one thing. Scope constants bound the            *)
(* environment, never the protocol.                                        *)
(*                                                                         *)
(* What is one action and why. The gate store is one serializable store    *)
(* (7.5), so a transaction on it is one action: Reserve, Consume, Begin,   *)
(* SafetyOpens, Publish, Breaks, Clears, StepPut. Everything outside it is *)
(* its own action, and those races are in the protocol's own runs: a       *)
(* source going bad before it is recorded, time passing with no write, a   *)
(* process dying before anyone reads it, an in-flight step acting after a  *)
(* failure opened, an engine acting on what it read before a handover, a   *)
(* probe read before the store came back. A split inside a transaction     *)
(* stays a mutation, since the store's isolation rules it out.             *)
(*                                                                         *)
(* Abstractions, stated:                                                   *)
(* - No epoch number: an advance kills every live reservation, which is    *)
(*   what comparing numbers decides.                                       *)
(* - WORKING for 24 hours is one published bit (3.5 keeps two predicates). *)
(* - No wall clock: a stale table and an expired reservation are actions.  *)
(* - A debit taken afterwards by an exempt lane is not counted.            *)
(***************************************************************************)
EXTENDS Integers, FiniteSets, TLC

CONSTANTS
  o1, o2, o3, o4, o5, o6, o7, o8, o9,   \* see Ords
  Ords,           \* model values: concurrent ordinary tasks of piece 1 (a symmetry set)
  Others,         \* the other start requests by name
  SafetyIds,      \* safety failures that can be open together
  Limit,          \* the order overlay's WIP limit per piece
  Seven,          \* the piece step 2 calls piece 7
  KeptPieces,     \* the pieces step 3 calls 8 and 9
  MaxPiece,       \* pieces are 1..MaxPiece, bottom up
  NSteps,         \* order steps the engines may take; step i moves piece i to i + 1
  BreakPieces,    \* pieces a stop can open on
  NeedTyped,      \* the release also needs a typed row (scope: a second evidence row)
  PubDieOn,       \* the reading writer can die
  SrcOn,          \* safety sources can go bad
  StoreDownOn,    \* the gate store and broker can go down
  CrashOn,        \* engines crash, restart and hand over
  EngRace,        \* an engine's read is its own action, before its put
  EscOn,          \* the escalation timers run
  EscDieOn,       \* the escalation timer's own process can die
  CreditOn,       \* credits are consumed and freed
  SweepOn,        \* the sweep parks lanes
  FlightKinds,    \* kinds whose outward step is two actions, Begin and Act
  TrackOverrun,   \* count outward actions after a park
  DeadTasks,      \* tasks whose process exit and the read of it are two actions
  DupTasks,       \* tasks whose start the bus can redeliver later
  LiveTask,       \* the task the admission liveness property is about
  SFTasks,        \* tasks whose admission retries until started (strong fairness)
  WFTasks,        \* the same with weak fairness only: a control
  Mut,            \* named breaks; {} is the protocol
  Drop            \* fairness clauses a control removes

M(x) == x \in Mut
Dropped(x) == x \in Drop

Tasks == Ords \cup Others
Procs == {"H", "F"}
Pieces == 1..MaxPiece
Steps == 1..NSteps

KindOf(t) ==
  CASE t \in Ords -> [k |-> "ord", p |-> 1, s |-> "-", r |-> "pub"]
    [] t = "j"  -> [k |-> "join", p |-> 2, s |-> "-", r |-> "pub"]
    [] t = "j3" -> [k |-> "join", p |-> 3, s |-> "-", r |-> "pub"]
    [] t = "u"  -> [k |-> "ord", p |-> 2, s |-> "-", r |-> "pub"]
    [] t \in {"c1", "c1b"} -> [k |-> "cause", p |-> 1, s |-> "-", r |-> "pub"]
    [] t \in {"c2", "c2b"} -> [k |-> "cause", p |-> 2, s |-> "-", r |-> "pub"]
    [] t = "c3" -> [k |-> "cause", p |-> MaxPiece, s |-> "-", r |-> "pub"]
    [] t = "kf" -> [k |-> "kept", p |-> MaxPiece, s |-> "-", r |-> "pub"]
    [] t = "g"  -> [k |-> "cont", p |-> 0, s |-> "-", r |-> "pub"]
    [] t \in {"r1", "r2"} -> [k |-> "repair", p |-> 0, s |-> "-", r |-> "pub"]
    [] t = "s1" -> [k |-> "scause", p |-> MaxPiece, s |-> "s1", r |-> "pub"]
    [] t = "s2" -> [k |-> "scause", p |-> MaxPiece, s |-> "s2", r |-> "pub"]
    [] t \in {"e1", "e2"} -> [k |-> "em", p |-> 0, s |-> "-", r |-> "pub"]

\* Every kind of the table, checked in every reachable state. r "priv" is a task in a private
\* repository outside step 0b's list; a safety cause lane there is the case 0b must pass.
AllKinds ==
  [k : {"ord", "join", "cause", "kept"}, p : Pieces, s : {"-"}, r : {"pub", "priv"}]
  \cup [k : {"ro", "cont", "repair", "b0", "other"}, p : {0}, s : {"-"}, r : {"pub", "priv"}]
  \cup [k : {"scause"}, p : {MaxPiece}, s : SafetyIds, r : {"pub", "priv"}]

VARIABLES
  pc,        \* [Tasks -> idle, resv, run, dead]; dec, rold, cons, zombie only under a mutation
  parked,    \* lanes the sweep has parked; each keeps its place
  inflight,  \* lanes whose begun step has not acted yet
  overrun,   \* [FlightTasks -> outward actions since the lane last began under no park]
  dup,       \* starts the bus still holds a redelivery of
  src,       \* [SafetyIds -> BOOLEAN]: the source itself reads bad
  safety,    \* failures recorded in the gate store: this is what "open" means
  sEsc,      \* open failures with their escalation row
  fresh,     \* the reading table is inside 30 minutes
  pubUp,     \* the reading writer is alive
  broken,    \* pieces truly not working
  working,   \* pieces the table in the store reads as WORKING
  stop,      \* [Pieces -> none, open, esc]
  rows,      \* [Pieces -> SUBSET {"see", "typed"}], bound to the open stop
  cur,       \* the lowest unfinished piece
  joined,    \* pieces whose join has passed
  started,   \* pieces whose join has been consumed
  credit,    \* credits at hand
  storeUp,   \* the gate store and broker answer
  probeDown, \* the local probe's last read says down; may be behind the truth
  emCount,   \* emergency starts this outage
  up,        \* [Procs -> BOOLEAN]
  handed,    \* the record names the fallback as owner
  phase,     \* [Procs -> Steps + 0]: step the engine has read and not yet put
  snap,      \* [Procs -> what the read saw]: read only under a mutation
  keys,      \* [Procs -> SUBSET Steps]: the one key store is keys["H"]
  effects,   \* [Steps -> 0..2]: times each step's effect ran
  escUp,     \* the escalation timer's process
  ghSee,     \* ghost: [Pieces -> go-and-see written after the stop and the last lease]
  ghRead,    \* ghost: [Pieces -> WORKING probed after the stop and the last lease]
  bad,       \* ghost: names of the safety properties that broke; stays {} in the protocol
  tok, checked, oldResv, pend, pendKey   \* state of mutations only

laneV  == <<pc, parked, inflight, overrun, dup>>
gateV  == <<src, safety, sEsc, fresh, pubUp, broken, working, stop, rows, cur, joined, started, credit>>
storeV == <<storeUp, probeDown, emCount>>
engV   == <<up, handed, phase, snap, keys, effects, escUp>>
ghV    == <<ghSee, ghRead, bad>>
mutV   == <<tok, checked, oldResv, pend, pendKey>>
vars   == <<laneV, gateV, storeV, engV, ghV, mutV>>

K(t) == KindOf(t)
Rebuild(k) == k.k \in {"ord", "join", "cause", "scause"}
AnyRun(k) == \E t \in Tasks : K(t).k = k /\ pc[t] = "run"
Stops == {p \in Pieces : stop[p] # "none"}
Min(S) == CHOOSE x \in S : \A y \in S : x <= y
Low == Min(Stops)
Owner == IF handed THEN "F" ELSE "H"
FlightTasks == {t \in Tasks : K(t).k \in FlightKinds}

Working(p) == fresh /\ p \in working
Pred(p) == p = 1 \/ Working(p - 1)
Joined(p) == p = 1 \/ p \in joined

\* ---- places: what the gate counts, in the reservation's own transaction ----
ReadPiece(k) == IF k.k = "join" /\ M("join_split") THEN k.p - 1 ELSE k.p
Wip(k) == k.k \in {"ord", "join"}
Holding ==
  (IF M("rows_only") THEN {"run", "dead"} ELSE {"resv", "run", "dead"})
GateCount(p) ==
  Cardinality({t \in Tasks : Wip(K(t)) /\ K(t).p = p /\ pc[t] \in Holding
                             /\ ~(M("parked_frees_place") /\ t \in parked)})
Room(p) == GateCount(p) < Limit
\* The lanes alive, whatever any row says: the independent reading's truth.
Alive(p) == Cardinality({t \in Tasks : Wip(K(t)) /\ ReadPiece(K(t)) = p /\ pc[t] \in {"cons", "run", "zombie"}})

\* One reserved cause slot per piece, one bounded repair lane; a dead lane holds it until read gone.
Slot(k) ==
  M("no_slot") \/ \A t \in Tasks : (K(t).k = k.k /\ K(t).p = k.p /\ K(t).s = k.s) => pc[t] = "idle"
RepairFree == M("repair_unbounded") \/ \A t \in Tasks : K(t).k = "repair" => pc[t] = "idle"

\* The freed-place guard: a safety cause lane, a repair lane and a stop's cause lane debit afterwards.
NeedsCredit(k) ==
  /\ k.k \notin {"ro", "scause", "repair", "em", "b0"}
  /\ (k.k = "cause" => M("plan_cause_needs_credit"))
CreditOK(k) == (NeedsCredit(k) \/ (M("credit_refuses_safety") /\ k.k = "scause")) => credit > 0

RepairCredit == ~M("credit_refuses_repair") \/ credit > 0

\* ---- the table of 7.1 ("next" passes on) ----
Step0b(k) ==
  IF k.r = "priv" /\ (k.k # "scause" \/ M("scause_filtered")) THEN "refuse" ELSE "next"
Step1(k) ==
  IF safety = {} THEN "next"
  ELSE IF k.k = "scause" /\ k.s \in safety THEN (IF CreditOK(k) THEN "admit" ELSE "refuse")
  ELSE IF k.k = "repair" /\ ~fresh /\ RepairFree /\ RepairCredit THEN "admit"
  ELSE "refuse"
Step2(k) ==
  IF fresh THEN "next"
  ELSE IF k.k = "repair" THEN (IF RepairFree /\ RepairCredit THEN "admit" ELSE "refuse")
  ELSE IF k.k = "kept" \/ (Rebuild(k) /\ k.p > Seven) THEN "refuse"
  ELSE "next"
PredOK(k) == M("no_pred") \/ (Pred(k.p) /\ Joined(k.p))
OrdOK(k) == k.p = cur /\ PredOK(k) /\ Room(k.p) /\ CreditOK(k)
JoinOK(k) ==
  /\ M("no_join_target") \/ k.p = cur + 1
  /\ k.p \notin joined
  /\ M("no_pred") \/ Pred(k.p)
  /\ Room(k.p) /\ CreditOK(k)
KeptOK(k) == (k.p \in KeptPieces \/ k.p = cur) /\ CreditOK(k)
Step3(k) ==
  IF Stops = {} THEN "next"
  ELSE CASE k.k = "cause" ->
              IF /\ stop[k.p] # "none"
                 /\ k.p = Low \/ (k.p \in KeptPieces /\ ~M("cause_only_low"))
                 /\ Slot(k) /\ CreditOK(k)
              THEN "admit" ELSE "refuse"
         [] k.k = "kept" ->
              IF ~M("kept_refused_under_stop") /\ k.p \in KeptPieces /\ CreditOK(k) THEN "admit" ELSE "refuse"
         [] k.k = "ord"  -> IF k.p < Low /\ OrdOK(k) THEN "admit" ELSE "refuse"
         [] k.k = "join" -> IF k.p < Low /\ JoinOK(k) THEN "admit" ELSE "refuse"
         [] k.k = "b0"   -> "admit"
         [] k.k = "cont" -> IF CreditOK(k) THEN "admit" ELSE "refuse"
         [] OTHER -> "refuse"
StepOrder(k) ==
  CASE k.k = "ord"  -> IF OrdOK(k) THEN "admit" ELSE "refuse"
    [] k.k = "join" -> IF JoinOK(k) THEN "admit" ELSE "refuse"
    [] k.k = "kept" -> IF KeptOK(k) THEN "admit" ELSE "refuse"
    [] k.k = "cont" -> IF CreditOK(k) THEN "admit" ELSE "refuse"
    [] k.k = "b0"   -> "admit"
    [] OTHER -> "refuse"
First(a, b, c, d, e) ==
  IF a # "next" THEN a ELSE IF b # "next" THEN b ELSE IF c # "next" THEN c ELSE IF d # "next" THEN d ELSE e
Decide(k) ==
  IF k.k = "ro" /\ ~M("ro_after_safety") THEN "admit"
  ELSE IF M("stale_before_safety") THEN First(Step0b(k), Step2(k), Step1(k), Step3(k), StepOrder(k))
  ELSE IF M("stop_before_stale") THEN First(Step0b(k), Step1(k), Step3(k), Step2(k), StepOrder(k))
  ELSE IF M("cont_in_safety") /\ k.k = "cont" THEN "admit"
  ELSE IF k.k = "ro" THEN First(Step1(k), "next", "next", "next", "admit")
  ELSE First(Step0b(k), Step1(k), Step2(k), Step3(k), StepOrder(k))

\* ---- what the order forbids, written from the property and not from the table ----
\* WIP room and credit are not here: they have their own properties.
OrderForbids(k) ==
  \/ (k.k = "ord" /\ ~(k.p = cur /\ Pred(k.p) /\ Joined(k.p)))
  \/ (k.k = "join" /\ ~(k.p = cur + 1 /\ k.p \notin joined /\ Pred(k.p)))
Against(k) ==
  IF k.k = "ro" THEN FALSE
  ELSE IF k.r = "priv" /\ k.k # "scause" THEN TRUE
  ELSE IF k.k = "em" THEN storeUp /\ ~probeDown
  ELSE IF safety # {} THEN ~((k.k = "scause" /\ k.s \in safety) \/ (k.k = "repair" /\ ~fresh))
  ELSE IF k.k = "scause" THEN TRUE
  ELSE IF ~fresh /\ (k.k = "kept" \/ (Rebuild(k) /\ k.p > Seven)) THEN TRUE
  ELSE IF ~fresh /\ k.k = "repair" THEN FALSE
  ELSE IF k.k \in {"other", "repair"} THEN TRUE
  ELSE IF Stops # {} THEN
       \/ (k.k = "kept" /\ k.p \notin KeptPieces)
       \/ (k.k = "cause" /\ (stop[k.p] = "none" \/ (k.p # Low /\ k.p \notin KeptPieces)))
       \/ (k.k \in {"ord", "join"} /\ k.p >= Low)
       \/ OrderForbids(k)
  ELSE \/ k.k = "cause"
       \/ (k.k = "kept" /\ k.p \notin KeptPieces /\ k.p # cur)
       \/ OrderForbids(k)
\* A stop already on the ledger that the gate store has not taken: only under gate_lag.
LagAgainst(k) ==
  /\ pend # {} /\ safety = {}
  /\ k.k \in {"ord", "join"} /\ k.p >= Min(Stops \cup pend)

\* The epoch re-read does not cover what changes with no write: age and the table it ages.
TimeOK(k) ==
  /\ (safety = {} /\ ~fresh) => ~(k.k = "kept" \/ (Rebuild(k) /\ k.p > Seven))
  /\ k.k \in {"ord", "join"} => Pred(k.p)

\* ---- the fence: who may begin an outward step ----
OwnCause(t) ==
  \/ (K(t).k = "scause" /\ K(t).s \in safety)
  \/ (K(t).k = "repair" /\ ~fresh)
  \/ K(t).k = "em"
SafetyParks(t) == safety # {} /\ ~OwnCause(t)
StopParks(t) ==
  /\ Stops # {}
  /\ \/ (K(t).k \in {"ord", "join"} /\ K(t).p >= Low)
     \/ (K(t).k = "cause" /\ K(t).p > Low /\ K(t).p \notin KeptPieces)
ParkCond(t) == SafetyParks(t) \/ StopParks(t)

\* ---- the epoch ----
Kill ==
  IF M("no_epoch") THEN pc
  ELSE [t \in Tasks |-> IF pc[t] = "resv" THEN (IF M("consume_after_epoch") THEN "rold" ELSE "idle")
                        ELSE pc[t]]

Init ==
  /\ pc = [t \in Tasks |-> "idle"] /\ parked = {} /\ inflight = {}
  /\ overrun = [t \in FlightTasks |-> 0] /\ dup = {}
  /\ src = [s \in SafetyIds |-> FALSE] /\ safety = {} /\ sEsc = {}
  /\ fresh = TRUE /\ pubUp = TRUE /\ broken = {} /\ working = Pieces
  /\ stop = [p \in Pieces |-> "none"] /\ rows = [p \in Pieces |-> {}]
  /\ cur = 1 /\ joined = {} /\ started = {} /\ credit = 1
  /\ storeUp = TRUE /\ probeDown = FALSE /\ emCount = 0
  /\ up = [x \in Procs |-> TRUE] /\ handed = FALSE
  /\ phase = [x \in Procs |-> 0] /\ snap = [x \in Procs |-> <<TRUE, FALSE>>]
  /\ keys = [x \in Procs |-> {}] /\ effects = [i \in Steps |-> 0] /\ escUp = TRUE
  /\ ghSee = [p \in Pieces |-> FALSE] /\ ghRead = [p \in Pieces |-> TRUE] /\ bad = {}
  /\ tok = {} /\ checked = {} /\ oldResv = {} /\ pend = {} /\ pendKey = [x \in Procs |-> 0]

\* ---- the admission path, each a store transaction ----
Reserve(t) ==
  /\ pc[t] = "idle" /\ K(t).k # "em"
  /\ storeUp \/ M("fail_open_on_down")
  /\ M("holder_in_engine") => up[Owner]
  /\ Decide(K(t)) = "admit"
  /\ pc' = [pc EXCEPT ![t] = IF M("late_bind") THEN "dec" ELSE "resv"]
  /\ UNCHANGED <<parked, inflight, overrun, dup, gateV, storeV, engV, ghV, mutV>>
Bind(t) ==     \* late_bind only: the facts were read before the epoch
  /\ M("late_bind") /\ pc[t] = "dec"
  /\ pc' = [pc EXCEPT ![t] = "resv"]
  /\ UNCHANGED <<parked, inflight, overrun, dup, gateV, storeV, engV, ghV, mutV>>
Expire(t) ==   \* a lifetime ended: the store re-lets the place
  /\ pc[t] \in {"resv", "rold", "dec"} /\ storeUp
  /\ pc' = [pc EXCEPT ![t] = "idle"]
  /\ tok' = IF M("expired_token") /\ pc[t] = "resv" THEN tok \cup {t} ELSE tok
  /\ oldResv' = oldResv \ {t}
  /\ UNCHANGED <<parked, inflight, overrun, dup, gateV, storeV, engV, ghV, checked, pend, pendKey>>
Consume(t) ==  \* the compare: reservation live (a killed one is gone), expiry, age, credit
  /\ \/ pc[t] = "resv"
     \/ (M("consume_after_epoch") /\ pc[t] = "rold")
     \/ (M("expired_token") /\ pc[t] = "idle" /\ t \in tok)
  /\ storeUp \/ M("fail_open_on_down")
  /\ M("holder_in_engine") => up[Owner]
  /\ M("no_recheck") \/ TimeOK(K(t))
  /\ (CreditOn /\ NeedsCredit(K(t)) /\ ~M("credit_no_compare")) => credit > 0
  /\ pc' = [pc EXCEPT ![t] = IF M("row_gap") /\ Wip(K(t)) THEN "cons" ELSE "run"]
  /\ credit' = IF CreditOn /\ NeedsCredit(K(t)) THEN credit - 1 ELSE credit
  /\ dup' = IF t \in DupTasks THEN dup \cup {t} ELSE dup
  /\ tok' = tok \ {t} /\ oldResv' = oldResv \ {t}
  /\ started' = IF K(t).k = "join" THEN started \cup {K(t).p} ELSE started
  \* a cause lane's lease voids the evidence that came before it
  /\ rows' = IF K(t).k = "cause" /\ ~M("see_before_lease") THEN [rows EXCEPT ![K(t).p] = @ \ {"see"}] ELSE rows
  /\ working' = IF K(t).k = "cause" /\ ~M("lease_keeps_reading") THEN working \ {K(t).p} ELSE working
  /\ ghSee' = IF K(t).k = "cause" THEN [ghSee EXCEPT ![K(t).p] = FALSE] ELSE ghSee
  /\ ghRead' = IF K(t).k = "cause" THEN [ghRead EXCEPT ![K(t).p] = FALSE] ELSE ghRead
  /\ bad' = bad \cup (IF Against(K(t)) \/ LagAgainst(K(t)) THEN {"start"} ELSE {})
                \cup (IF pc[t] = "rold" \/ t \in oldResv THEN {"epoch"} ELSE {})
                \cup (IF ~storeUp THEN {"down"} ELSE {})
                \cup (IF credit' < 0 THEN {"credit"} ELSE {})
  /\ UNCHANGED <<parked, inflight, overrun, src, safety, sEsc, fresh, pubUp, broken, stop, cur, joined,
                 storeV, engV, checked, pend, pendKey>>
Redeliver(t) ==  \* the bus delivers the start again, maybe after a new reservation exists
  /\ t \in dup /\ dup' = dup \ {t}
  /\ bad' = bad \cup (IF (M("reservation_reused") /\ pc[t] = "run") \/ (M("redelivery_aba") /\ pc[t] = "resv")
                      THEN {"double"} ELSE {})
  /\ UNCHANGED <<pc, parked, inflight, overrun, gateV, storeV, engV, ghSee, ghRead, mutV>>
EmStart(t) ==    \* with the store unreachable: a local probe and a journalled single-use credential
  /\ K(t).k = "em" /\ pc[t] = "idle"
  /\ probeDown \/ M("em_caller_word")
  /\ t = "e1" \/ M("em_per_host")
  /\ emCount = 0 \/ M("em_per_host")
  /\ pc' = [pc EXCEPT ![t] = "run"]
  /\ emCount' = IF emCount < 2 THEN emCount + 1 ELSE emCount
  /\ bad' = bad \cup (IF Against(K(t)) THEN {"start"} ELSE {}) \cup (IF emCount >= 1 THEN {"em"} ELSE {})
  /\ UNCHANGED <<parked, inflight, overrun, dup, gateV, storeUp, probeDown, engV, ghSee, ghRead, mutV>>

\* ---- a running lane ----
Ends(t) ==       \* the process exits; a lane that fails ends the same way and leaves nothing
  /\ pc[t] = "run"
  /\ pc' = [pc EXCEPT ![t] = IF t \in DeadTasks THEN "dead" ELSE "idle"]
  /\ parked' = parked \ {t} /\ inflight' = inflight \ {t} /\ checked' = checked \ {t}
  /\ UNCHANGED <<overrun, dup, gateV, storeV, engV, ghV, tok, oldResv, pend, pendKey>>
ReadGone(t) ==   \* the only thing that frees a place: the process read as gone
  /\ pc[t] = "dead" /\ (storeUp \/ K(t).k = "em")   \* the credential's holder reads its own lane locally
  /\ pc' = [pc EXCEPT ![t] = "idle"]
  /\ UNCHANGED <<parked, inflight, overrun, dup, gateV, storeV, engV, ghV, mutV>>
ClaimRow(t) ==   \* row_gap only: the lane writes its CLAIM after it started
  /\ pc[t] = "cons" /\ pc' = [pc EXCEPT ![t] = "run"]
  /\ UNCHANGED <<parked, inflight, overrun, dup, gateV, storeV, engV, ghV, mutV>>
CloseRow(t) ==   \* third_close only: a TERMINAL by another writer frees the place, the process lives
  /\ M("third_close") /\ pc[t] = "run" /\ Wip(K(t)) /\ pc' = [pc EXCEPT ![t] = "zombie"]
  /\ UNCHANGED <<parked, inflight, overrun, dup, gateV, storeV, engV, ghV, mutV>>
ZombieEnds(t) ==
  /\ pc[t] = "zombie" /\ pc' = [pc EXCEPT ![t] = "idle"]
  /\ UNCHANGED <<parked, inflight, overrun, dup, gateV, storeV, engV, ghV, mutV>>
FenceHolds(t) ==
  LET checkedFor ==
        IF M("fence_off") \/ (M("plan_cloud_lane") /\ K(t).k = "cont") THEN FALSE
        ELSE IF M("fence_cont_only") /\ K(t).k # "cont" THEN FALSE
        ELSE IF M("no_stop_park") THEN SafetyParks(t)
        ELSE ParkCond(t)
  IN /\ ~checkedFor
     /\ storeUp \/ OwnCause(t) \/ M("fence_off") \/ M("plan_cloud_lane")
Check(t) ==      \* fence_split only: the fence read, the write to come
  /\ M("fence_split") /\ pc[t] = "run" /\ t \in FlightTasks /\ t \notin checked /\ FenceHolds(t)
  /\ checked' = checked \cup {t}
  /\ UNCHANGED <<laneV, gateV, storeV, engV, ghV, tok, oldResv, pend, pendKey>>
Begin(t) ==      \* a store write: the fence is checked in it
  /\ pc[t] = "run" /\ t \in FlightTasks /\ t \notin inflight /\ t \notin parked
  /\ IF M("fence_split") THEN t \in checked ELSE FenceHolds(t)
  /\ inflight' = inflight \cup {t} /\ checked' = checked \ {t}
  /\ overrun' = IF TrackOverrun /\ ~ParkCond(t) THEN [overrun EXCEPT ![t] = 0] ELSE overrun
  /\ bad' = IF ParkCond(t) THEN bad \cup {"begin"} ELSE bad
  /\ UNCHANGED <<pc, parked, dup, gateV, storeV, engV, ghSee, ghRead, tok, oldResv, pend, pendKey>>
Act(t) ==        \* the push, merge or deploy: outside the store, and it may land after a failure opens
  /\ t \in inflight
  /\ inflight' = IF M("fence_once") THEN inflight ELSE inflight \ {t}
  /\ overrun' = IF TrackOverrun /\ ParkCond(t) /\ overrun[t] < 2 THEN [overrun EXCEPT ![t] = @ + 1] ELSE overrun
  /\ UNCHANGED <<pc, parked, dup, gateV, storeV, engV, ghV, mutV>>
Sweep ==         \* one read: lanes under a park condition park, the others come back. Each keeps its place
  LET now == {t \in Tasks : pc[t] = "run" /\ ParkCond(t)} IN
  /\ SweepOn /\ storeUp /\ parked # now /\ parked' = now
  /\ UNCHANGED <<pc, inflight, overrun, dup, gateV, storeV, engV, ghV, mutV>>
Unparked(t) == pc[t] = "run" /\ t \notin parked /\ ~ParkCond(t) /\ (storeUp \/ OwnCause(t))
JoinPasses(t) ==
  /\ K(t).k = "join" /\ Unparked(t) /\ storeUp /\ K(t).p \notin joined
  /\ joined' = joined \cup {K(t).p}
  /\ UNCHANGED <<laneV, src, safety, sEsc, fresh, pubUp, broken, working, stop, rows, cur, started, credit,
                 storeV, engV, ghV, mutV>>
RepairWorks(t) ==  \* it may never happen: a repair lane can end without this
  /\ K(t).k = "repair" /\ Unparked(t) /\ ~pubUp /\ pubUp' = TRUE
  /\ UNCHANGED <<laneV, src, safety, sEsc, fresh, broken, working, stop, rows, cur, joined, started, credit,
                 storeV, engV, ghV, mutV>>
Fix(t) ==          \* nothing assumes a cause lane ever succeeds
  /\ K(t).k = "cause" /\ Unparked(t) /\ K(t).p \in broken
  /\ broken' = broken \ {K(t).p}
  /\ UNCHANGED <<laneV, src, safety, sEsc, fresh, pubUp, working, stop, rows, cur, joined, started, credit,
                 storeV, engV, ghV, mutV>>
SafetyFix(t) ==
  /\ pc[t] = "run" /\ K(t).k = "scause" /\ K(t).s \in safety /\ src[K(t).s]
  /\ src' = [src EXCEPT ![K(t).s] = FALSE]
  /\ UNCHANGED <<laneV, safety, sEsc, fresh, pubUp, broken, working, stop, rows, cur, joined, started, credit,
                 storeV, engV, ghV, mutV>>
CreditFreed ==     \* a merge or closure read in the class
  /\ CreditOn /\ credit = 0 /\ credit' = 1
  /\ UNCHANGED <<laneV, src, safety, sEsc, fresh, pubUp, broken, working, stop, rows, cur, joined, started,
                 storeV, engV, ghV, mutV>>

\* ---- safety failures: from their own sources, never the table ----
SrcBad(s) ==       \* the world changes; the store has not read it: the base lag
  /\ SrcOn /\ ~src[s] /\ s \notin safety /\ src' = [src EXCEPT ![s] = TRUE]
  /\ UNCHANGED <<laneV, safety, sEsc, fresh, pubUp, broken, working, stop, rows, cur, joined, started, credit,
                 storeV, engV, ghV, mutV>>
SafetyOpens(s) ==
  /\ src[s] /\ s \notin safety /\ storeUp
  /\ safety' = safety \cup {s} /\ pc' = Kill
  /\ UNCHANGED <<parked, inflight, overrun, dup, src, sEsc, fresh, pubUp, broken, working, stop, rows, cur, joined,
                 started, credit, storeV, engV, ghV, mutV>>
SafetyCloses(s) ==   \* its own source read clean after it opened (the read is at the commit)
  /\ s \in safety /\ ~src[s] /\ storeUp
  /\ M("close_on_table") => fresh
  /\ safety' = safety \ {s} /\ sEsc' = sEsc \ {s} /\ pc' = Kill
  /\ UNCHANGED <<parked, inflight, overrun, dup, src, fresh, pubUp, broken, working, stop, rows, cur, joined,
                 started, credit, storeV, engV, ghV, mutV>>
Timer == escUp /\ storeUp /\ (M("escalate_on_reader") => pubUp)
SafetyEscalates(s) ==
  /\ EscOn /\ ~M("safety_no_escalation") /\ s \in safety /\ s \notin sEsc /\ Timer
  /\ sEsc' = sEsc \cup {s}
  /\ UNCHANGED <<laneV, src, safety, fresh, pubUp, broken, working, stop, rows, cur, joined, started, credit,
                 storeV, engV, ghV, mutV>>
EscDies ==
  /\ EscOn /\ EscDieOn /\ escUp /\ escUp' = FALSE
  /\ UNCHANGED <<laneV, gateV, storeV, up, handed, phase, snap, keys, effects, ghV, mutV>>
EscBack ==
  /\ ~escUp /\ escUp' = TRUE
  /\ UNCHANGED <<laneV, gateV, storeV, up, handed, phase, snap, keys, effects, ghV, mutV>>

\* ---- the reading writer and the table; readings live in the store ----
PubDies ==
  /\ PubDieOn /\ pubUp /\ pubUp' = FALSE
  /\ UNCHANGED <<laneV, src, safety, sEsc, fresh, broken, working, stop, rows, cur, joined, started, credit,
                 storeV, engV, ghV, mutV>>
GoesStale ==       \* time passing: no writer, so no advance; Consume re-reads it
  /\ fresh /\ ~pubUp /\ fresh' = FALSE
  /\ UNCHANGED <<laneV, src, safety, sEsc, pubUp, broken, working, stop, rows, cur, joined,
                 started, credit, storeV, engV, ghV, mutV>>
Publish ==         \* a republish with the same readings is no write that advances
  LET w == Pieces \ broken IN
  /\ pubUp /\ storeUp /\ (~fresh \/ w # working)
  /\ working' = w /\ fresh' = TRUE /\ pc' = Kill
  /\ ghRead' = [q \in Pieces |-> ghRead[q] \/ q \in w]
  /\ UNCHANGED <<parked, inflight, overrun, dup, src, safety, sEsc, pubUp, broken, stop, rows, cur, joined,
                 started, credit, storeV, engV, ghSee, bad, mutV>>

\* ---- the stop; every row binds to the open stop, which Breaks and Clears end ----
Started(p) == p <= cur \/ p \in KeptPieces \/ p \in joined \/ p \in started
Breaks(p) ==
  /\ ~M("gate_lag") /\ p \in BreakPieces /\ stop[p] = "none" /\ storeUp
  /\ Started(p) \/ M("stop_on_unstarted")
  /\ stop' = [stop EXCEPT ![p] = "open"] /\ broken' = broken \cup {p}
  /\ working' = IF M("old_reading") THEN working ELSE working \ {p}
  /\ rows' = IF M("rows_survive") THEN rows ELSE [rows EXCEPT ![p] = {}]
  /\ ghSee' = [ghSee EXCEPT ![p] = FALSE] /\ ghRead' = [ghRead EXCEPT ![p] = FALSE]
  /\ pc' = Kill
  /\ UNCHANGED <<parked, inflight, overrun, dup, src, safety, sEsc, fresh, pubUp, cur, joined, started, credit,
                 storeV, engV, bad, mutV>>
RowOnLedger(p) ==  \* gate_lag only: the row is on the ledger, the gate has not taken it
  /\ M("gate_lag") /\ p \in BreakPieces /\ stop[p] = "none" /\ p \notin pend
  /\ pend' = pend \cup {p}
  /\ UNCHANGED <<laneV, gateV, storeV, engV, ghV, tok, checked, oldResv, pendKey>>
GateTakes(p) ==
  /\ p \in pend /\ pend' = pend \ {p}
  /\ stop' = [stop EXCEPT ![p] = "open"] /\ broken' = broken \cup {p} /\ working' = working \ {p}
  /\ pc' = Kill
  /\ UNCHANGED <<parked, inflight, overrun, dup, src, safety, sEsc, fresh, pubUp, rows, cur, joined, started, credit,
                 storeV, engV, ghV, tok, checked, oldResv, pendKey>>
EvidenceRow(p, r) ==  \* go-and-see is read-only: it needs no admission; it counts once durable
  /\ stop[p] # "none" /\ storeUp /\ r \notin rows[p] /\ (r = "typed" => NeedTyped)
  /\ rows' = [rows EXCEPT ![p] = @ \cup {r}]
  /\ ghSee' = IF r = "see" THEN [ghSee EXCEPT ![p] = TRUE] ELSE ghSee
  /\ UNCHANGED <<laneV, src, safety, sEsc, fresh, pubUp, broken, working, stop, cur, joined, started, credit,
                 storeV, engV, ghRead, bad, mutV>>
Required == IF NeedTyped THEN {"see", "typed"} ELSE {"see"}
Evidence(p) == Required \subseteq rows[p]
Clears(p) ==
  /\ stop[p] # "none" /\ storeUp
  /\ IF M("clear_no_rows") THEN Working(p)
     ELSE IF M("clear_no_reading") THEN Evidence(p)
     ELSE Evidence(p) /\ Working(p)
  /\ stop' = [stop EXCEPT ![p] = "none"]
  /\ rows' = IF M("rows_survive") THEN rows ELSE [rows EXCEPT ![p] = {}]
  /\ bad' = bad \cup (IF ghSee[p] /\ ghRead[p] /\ fresh /\ p \notin broken THEN {} ELSE {"clear"})
                \cup (IF M("track_cleared") THEN {"wit_cleared"} ELSE {})
  /\ ghSee' = [ghSee EXCEPT ![p] = FALSE]
  /\ pc' = Kill
  /\ UNCHANGED <<parked, inflight, overrun, dup, src, safety, sEsc, fresh, pubUp, broken, working, cur, joined,
                 started, credit, storeV, engV, ghRead, mutV>>
Escalates(p) ==
  /\ EscOn /\ ~M("no_escalation") /\ stop[p] = "open" /\ Timer
  /\ stop' = [stop EXCEPT ![p] = "esc"]
  /\ UNCHANGED <<laneV, src, safety, sEsc, fresh, pubUp, broken, working, rows, cur, joined, started, credit,
                 storeV, engV, ghV, mutV>>

\* ---- the store and the emergency start ----
StoreDown ==
  /\ StoreDownOn /\ storeUp /\ storeUp' = FALSE /\ emCount' = 0
  /\ UNCHANGED <<laneV, gateV, probeDown, engV, ghV, mutV>>
StoreBack ==
  /\ ~storeUp /\ storeUp' = TRUE
  /\ UNCHANGED <<laneV, gateV, probeDown, emCount, engV, ghV, mutV>>
ProbeRead ==       \* the caller's own library: it may be behind the truth when it is used
  /\ StoreDownOn /\ probeDown # ~storeUp /\ probeDown' = ~storeUp
  /\ UNCHANGED <<laneV, gateV, storeUp, emCount, engV, ghV, mutV>>

\* ---- the engines. The record, the keys and the fence are in the store, outside both engines ----
Ready(i) == cur = i /\ Working(i) /\ (i + 1) \in joined /\ stop[i] = "none" /\ safety = {}
FenceCmp(x) ==
  IF M("no_fence") \/ (M("record_per_engine") /\ x = "H") THEN TRUE
  ELSE IF M("fence_read") THEN snap[x][1]
  ELSE x = Owner
KeyCmp(i, x) ==
  IF M("unkeyed") THEN FALSE
  ELSE IF M("key_read") THEN snap[x][2]
  ELSE IF M("keys_per_engine") THEN i \in keys[x]
  ELSE i \in keys["H"]
KeyStore(x) == IF M("keys_per_engine") THEN x ELSE "H"
EngRead(x) ==      \* the engine's own history: what it replays after a restart, stale or not
  /\ EngRace /\ up[x] /\ phase[x] = 0 /\ cur <= NSteps
  /\ phase' = [phase EXCEPT ![x] = cur]
  /\ snap' = IF M("fence_read") \/ M("key_read")
             THEN [snap EXCEPT ![x] = <<x = Owner, cur \in keys["H"]>>] ELSE snap
  /\ UNCHANGED <<laneV, gateV, storeV, up, handed, keys, effects, escUp, ghV, mutV>>
StepPut(x) ==      \* one put in the store: compare, key and effect together
  LET i == phase[x] IN
  /\ EngRace /\ i # 0 /\ up[x] /\ storeUp
  /\ FenceCmp(x) /\ ~KeyCmp(i, x)
  /\ cur > i \/ Ready(i)
  /\ effects' = [effects EXCEPT ![i] = IF @ < 2 THEN @ + 1 ELSE @]
  /\ keys' = IF M("effect_then_key") THEN keys ELSE [keys EXCEPT ![KeyStore(x)] = @ \cup {i}]
  /\ pendKey' = IF M("effect_then_key") THEN [pendKey EXCEPT ![x] = i] ELSE pendKey
  /\ cur' = IF cur = i THEN i + 1 ELSE cur
  /\ pc' = IF cur = i THEN Kill ELSE pc
  /\ bad' = bad \cup (IF x # Owner THEN {"owner"} ELSE {}) \cup (IF effects[i] >= 1 THEN {"repeat"} ELSE {})
  /\ UNCHANGED <<parked, inflight, overrun, dup, src, safety, sEsc, fresh, pubUp, broken, working, stop, rows, joined,
                 started, credit, storeV, up, handed, phase, snap, escUp, ghSee, ghRead, tok, checked, oldResv, pend>>
KeyWrite(x) ==     \* effect_then_key only
  /\ pendKey[x] # 0 /\ up[x]
  /\ keys' = [keys EXCEPT ![KeyStore(x)] = @ \cup {pendKey[x]}] /\ pendKey' = [pendKey EXCEPT ![x] = 0]
  /\ UNCHANGED <<laneV, gateV, storeV, up, handed, phase, snap, effects, escUp, ghV, tok, checked, oldResv, pend>>
StepDone(x) ==     \* the compare refuses a stale engine or a replay; the engine reads afresh
  /\ EngRace /\ phase[x] # 0 /\ up[x] /\ storeUp
  /\ ~FenceCmp(x) \/ KeyCmp(phase[x], x)
  /\ phase' = [phase EXCEPT ![x] = 0]
  /\ UNCHANGED <<laneV, gateV, storeV, up, handed, snap, keys, effects, escUp, ghV, mutV>>
EngStep(x) ==      \* outside the holder part: the read and the put as one, with no local state
  /\ ~EngRace /\ up[x] /\ storeUp /\ x = Owner /\ cur <= NSteps /\ Ready(cur)
  /\ effects' = [effects EXCEPT ![cur] = @ + 1] /\ keys' = [keys EXCEPT !["H"] = @ \cup {cur}]
  /\ cur' = cur + 1 /\ pc' = Kill
  /\ UNCHANGED <<parked, inflight, overrun, dup, src, safety, sEsc, fresh, pubUp, broken, working, stop, rows, joined,
                 started, credit, storeV, up, handed, phase, snap, escUp, ghV, mutV>>
\* Outside the holder part nothing reads a non-owner's up bit, so only the owner's is tracked there.
Crash(x) ==        \* the engine's history survives: phase is what it replays
  /\ CrashOn /\ up[x] /\ (EngRace \/ x = Owner) /\ up' = [up EXCEPT ![x] = FALSE] /\ pendKey' = [pendKey EXCEPT ![x] = 0]
  /\ UNCHANGED <<laneV, gateV, storeV, handed, phase, snap, keys, effects, escUp, ghV, tok, checked, oldResv, pend>>
Recover(x) ==
  /\ CrashOn /\ ~M("no_restart") /\ ~up[x] /\ up' = [up EXCEPT ![x] = TRUE]
  /\ UNCHANGED <<laneV, gateV, storeV, handed, phase, snap, keys, effects, escUp, ghV, mutV>>
Handover ==        \* one way, a write in the store; the old engine may not know
  /\ CrashOn /\ ~handed /\ storeUp /\ handed' = TRUE
  /\ pc' = IF M("handover_keeps_reservation") THEN pc ELSE Kill
  /\ oldResv' = IF M("handover_keeps_reservation") THEN {t \in Tasks : pc[t] = "resv"} ELSE oldResv
  /\ up' = IF EngRace THEN up ELSE [up EXCEPT !["H"] = TRUE]
  /\ UNCHANGED <<parked, inflight, overrun, dup, gateV, storeV, phase, snap, keys, effects, escUp, ghV,
                 tok, checked, pend, pendKey>>

Next ==
  \/ \E t \in Tasks :
       \/ Reserve(t) \/ Bind(t) \/ Expire(t) \/ Consume(t) \/ Redeliver(t) \/ EmStart(t)
       \/ Ends(t) \/ ReadGone(t) \/ ClaimRow(t) \/ CloseRow(t) \/ ZombieEnds(t)
       \/ Check(t) \/ Begin(t) \/ Act(t)
       \/ JoinPasses(t) \/ RepairWorks(t) \/ Fix(t) \/ SafetyFix(t)
  \/ \E s \in SafetyIds : SrcBad(s) \/ SafetyOpens(s) \/ SafetyCloses(s) \/ SafetyEscalates(s)
  \/ \E p \in Pieces :
       \/ Breaks(p) \/ RowOnLedger(p) \/ GateTakes(p) \/ Clears(p) \/ Escalates(p)
       \/ \E r \in {"see", "typed"} : EvidenceRow(p, r)
  \/ \E x \in Procs :
       \/ EngRead(x) \/ StepPut(x) \/ KeyWrite(x) \/ StepDone(x) \/ EngStep(x) \/ Crash(x) \/ Recover(x)
  \/ Sweep \/ CreditFreed \/ PubDies \/ GoesStale \/ Publish \/ EscDies \/ EscBack
  \/ StoreDown \/ StoreBack \/ ProbeRead \/ Handover

Spec == Init /\ [][Next]_vars

\* ---- fairness. Each clause names what it assumes, and a ctl_drop_* config shows a property
\* needs it; one that hid nothing was deleted. No clause assumes a lane fixes anything, that an
\* operator acts, that the reading writer stops dying, or that a lane other than SFTasks asks.
On(x) == IF x \in Drop THEN {} ELSE {"on"}
Flap(S) == IF StoreDownOn THEN S ELSE {}         \* enabling goes through a store that flaps: strong
Calm(S) == IF StoreDownOn THEN {} ELSE S         \* the store stays up: weak is enough
TimerFlaps == EscDieOn \/ StoreDownOn
Fairness ==
  \* an admissible request is asked again until it starts (or, as a control, only weakly)
  /\ \A t \in SFTasks : SF_vars(Reserve(t)) /\ SF_vars(Consume(t))
  /\ \A t \in WFTasks : WF_vars(Reserve(t)) /\ WF_vars(Consume(t))
  \* a dead process is read
  /\ \A o \in On("readgone"), t \in Flap(DeadTasks) : SF_vars(ReadGone(t))
  /\ \A o \in On("readgone"), t \in Calm(DeadTasks) : WF_vars(ReadGone(t))
  \* time passes on a dead writer
  /\ \A o \in On("stale") : WF_vars(GoesStale)
  \* a safety source is read, bad or clean
  /\ \A o \in On("safetyopen"), s \in Flap(SafetyIds) : SF_vars(SafetyOpens(s))
  /\ \A o \in On("safetyopen"), s \in Calm(SafetyIds) : WF_vars(SafetyOpens(s))
  /\ \A o \in On("safetyclose"), s \in Flap(SafetyIds) : SF_vars(SafetyCloses(s))
  /\ \A o \in On("safetyclose"), s \in Calm(SafetyIds) : WF_vars(SafetyCloses(s))
  \* the escalation timer fires, off the table, the holder and the reading writer
  /\ \A o \in On("esc"), p \in (IF EscOn /\ TimerFlaps THEN BreakPieces ELSE {}) : SF_vars(Escalates(p))
  /\ \A o \in On("esc"), p \in (IF EscOn /\ ~TimerFlaps THEN BreakPieces ELSE {}) : WF_vars(Escalates(p))
  /\ \A o \in On("esc"), s \in (IF EscOn /\ TimerFlaps THEN SafetyIds ELSE {}) : SF_vars(SafetyEscalates(s))
  /\ \A o \in On("esc"), s \in (IF EscOn /\ ~TimerFlaps THEN SafetyIds ELSE {}) : WF_vars(SafetyEscalates(s))
  /\ \A o \in On("escback") : WF_vars(EscBack)
  \* an outage ends; the local probe runs; the credential's holder asks
  /\ \A o \in On("storeback") : WF_vars(StoreBack)
  /\ \A o \in On("probe") : WF_vars(ProbeRead)
  /\ \A o \in On("emstart"), t \in {"e1"} \cap Tasks : SF_vars(EmStart(t))
  \* the supervisor restarts an engine; the owner writes a ready step
  /\ \A o \in On("recover"), x \in Procs : WF_vars(Recover(x))
  /\ \A o \in On("step"), x \in (IF CrashOn THEN Procs ELSE {}) : SF_vars(EngStep(x))
  /\ \A o \in On("step"), x \in (IF CrashOn THEN {} ELSE Procs) : WF_vars(EngStep(x))

LiveSpec == Spec /\ Fairness

(***************************************************************************)
(* Property 1: admission follows the order of 7.1. "Open" means recorded   *)
(* in the gate store.                                                      *)
(***************************************************************************)
\* The table against an order written apart from it, for every kind in every reachable state.
\* Against holds the predecessor, join and next-piece rules, so this is also StrongOrderTable.
DecideFollowsOrder == \A k \in AllKinds : Decide(k) = "admit" => ~Against(k)
\* The reservation was decided earlier: this is carried by the epoch and the re-read.
NoStartAgainstOrder == "start" \notin bad
\* Step 0b, stale table, stops, credit at zero: a safety cause lane still gets in.
SafetyCauseAdmitted == \A k \in AllKinds : (k.k = "scause" /\ k.s \in safety) => Decide(k) = "admit"
RepairNeverRefusedForCredit == (~fresh /\ RepairFree) => Decide([k |-> "repair", p |-> 0, s |-> "-", r |-> "pub"]) = "admit"
ReadOnlyAdmitted == \A k \in AllKinds : k.k = "ro" => Decide(k) = "admit"
\* A stop gates no kept fix; a cause lane of 8 or 9 runs whatever the active piece.
KeptPassesStop == \A q \in KeptPieces :
  (Stops # {} /\ safety = {} /\ fresh /\ credit > 0) => Decide([k |-> "kept", p |-> q, s |-> "-", r |-> "pub"]) = "admit"
CauseOfKeptRuns == \A q \in KeptPieces :
  LET c == [k |-> "cause", p |-> q, s |-> "-", r |-> "pub"] IN
  (stop[q] # "none" /\ safety = {} /\ fresh /\ Slot(c)) => Decide(c) = "admit"
\* No wedge: the lowest stop, nothing else in the way, has a way in; a stale table has the repair lane.
EveryStopHasAWayIn ==
  (Stops # {} /\ safety = {}) =>
    LET c == [k |-> "cause", p |-> Low, s |-> "-", r |-> "pub"]
        r == [k |-> "repair", p |-> 0, s |-> "-", r |-> "pub"]
    IN IF fresh \/ Low <= Seven THEN (Decide(c) = "admit" \/ ~Slot(c)) ELSE (Decide(r) = "admit" \/ ~RepairFree)
OneRepairLane == Cardinality({t \in Tasks : K(t).k = "repair" /\ pc[t] \in {"resv", "run"}}) <= 1
OneCauseLane == \A p \in Pieces :
  Cardinality({t \in Tasks : K(t).k = "cause" /\ K(t).p = p /\ pc[t] \in {"resv", "run"}}) <= 1
\* "No continuation runs" as the text can mean it: no step begins, across every mutating lane.
NoStepBeginsWhileParked == "begin" \notin bad
\* The in-flight bound: at most one outward action of a lane lands after its park condition holds.
AtMostOneStepAfterPark == \A t \in FlightTasks : overrun[t] <= 1

(***************************************************************************)
(* Property 2: lanes per piece stay within the limit. Alive counts          *)
(* processes, not rows, so the row gap and a third-party TERMINAL show.    *)
(***************************************************************************)
AliveWithinLimit == \A p \in Pieces : Alive(p) <= Limit
CreditNotOverdrawn == "credit" \notin bad

(***************************************************************************)
(* Property 3: a stop clears only with its evidence, and clears or         *)
(* escalates. A stop opens only on a started piece or 8 or 9.              *)
(***************************************************************************)
StopClearsOnlyWithEvidence == "clear" \notin bad
StopsOnlyOnStarted == \A p \in Pieces : stop[p] # "none" => Started(p)
StopClearsOrEscalates == \A p \in BreakPieces : (stop[p] = "open") ~> (stop[p] # "open")
SafetyStopClearsOrEscalates == \A s \in SafetyIds : (s \in safety) ~> (s \notin safety \/ s \in sEsc)

(***************************************************************************)
(* Property 4: no step repeats, one owner writes, a reservation starts     *)
(* once and never after an advance; a store outage admits one emergency.   *)
(***************************************************************************)
RestartRepeatsNoStep == "repeat" \notin bad
NoWriteByLostOwner == "owner" \notin bad
ConsumedAtMostOnce == "double" \notin bad
NeverConsumedAfterEpoch == "epoch" \notin bad
NoStartWhileStoreDown == "down" \notin bad
OneEmergencyPerOutage == "em" \notin bad

(***************************************************************************)
(* Liveness: each is a chain of actions another writer can interrupt.      *)
(***************************************************************************)
AdmissibleStarts ==
  []<>(pc[LiveTask] = "idle" /\ storeUp /\ Decide(K(LiveTask)) = "admit") => []<>(pc[LiveTask] = "run")
\* Through a dead lane holding the slot, a stale table, stops, an unread holder and no credit.
SafetyCauseStarts == \A s \in SafetyIds :
  (s \in safety) ~> (s \notin safety \/ \E t \in Tasks : K(t).s = s /\ pc[t] = "run")
\* Retried while each repair lane fails, whatever else is open.
WriterRepairStarts == (~pubUp) ~> (pubUp \/ \E t \in Tasks : K(t).k = "repair" /\ pc[t] = "run")
StopCauseStarts == \A p \in BreakPieces :
  (stop[p] # "none" /\ Low = p) ~> (stop[p] = "none" \/ Low # p \/ safety # {} \/ ~fresh \/ \E t \in Tasks : K(t).k = "cause" /\ K(t).p = p /\ pc[t] = "run")
FixedSafetyCloses == \A s \in SafetyIds : (s \in safety /\ ~src[s]) ~> (s \notin safety)
BadSourceRecorded == \A s \in SafetyIds : (src[s]) ~> (s \in safety \/ ~src[s])
ReadyStepAdvances == \A i \in Steps : ([]<>Ready(i)) => <>(cur > i)
EmergencyStarts == (~storeUp /\ emCount = 0) ~> (storeUp \/ emCount > 0 \/ AnyRun("em"))

(***************************************************************************)
(* Witnesses: each must be violated, so the race it names is reachable and *)
(* a pass is not vacuous.                                                  *)
(***************************************************************************)
Has(t) == t \in Tasks
WitOrderAdvances == cur = 1
WitLimitReached == Alive(1) < Limit
WitSafetyCauseStartsOnStaleTable == ~(~fresh /\ safety # {} /\ AnyRun("scause"))
WitRepairDuringSafety == ~(~fresh /\ safety # {} /\ AnyRun("repair"))
WitLowerPieceRunsUnderStop == ~(\E t \in Ords : pc[t] = "run" /\ Stops # {} /\ Low > 1)
WitTwoStopsCauseOfKept == ~(stop[2] # "none" /\ stop[3] # "none" /\ \E t \in Tasks : K(t).k = "cause" /\ K(t).p = 3 /\ pc[t] = "run")
WitKeptRunsUnderStop == ~(AnyRun("kept") /\ Stops # {} /\ safety = {})
WitStopCleared == "wit_cleared" \notin bad
WitEscalated == \A p \in Pieces : stop[p] # "esc"
WitSafetyEscalated == sEsc = {}
\* Races the protocol lives with.
WitStartDuringSourceLag == ~(\E s \in SafetyIds : src[s] /\ s \notin safety /\ AnyRun("cont"))
WitStaleReservationHeld == ~(~fresh /\ \E t \in Tasks : K(t).k = "kept" /\ pc[t] = "resv")
WitActAfterFailureOpens == ~(\E t \in inflight : SafetyParks(t))
WitParkedHoldsPlace == ~(\E t \in parked : Wip(K(t)) /\ GateCount(K(t).p) >= Limit)
WitParkedByStop == ~(\E t \in parked : StopParks(t) /\ ~SafetyParks(t))
WitDeadCauseHoldsSlot == ~(\E t \in Tasks : K(t).k = "cause" /\ pc[t] = "dead" /\ stop[K(t).p] # "none")
WitOldEngineStillActing == ~(handed /\ up["H"] /\ phase["H"] # 0)
WitFallbackAdvances == ~(handed /\ cur > 1)
WitStartWithBothEnginesDown == ~(~up["H"] /\ ~up["F"] /\ \E t \in Tasks : pc[t] = "run")
WitEmergencyInOutage == ~(~storeUp /\ AnyRun("em"))
WitEmergencyAfterReturn == ~(storeUp /\ probeDown /\ AnyRun("em"))
WitCreditZeroCauseRuns == ~(credit = 0 /\ AnyRun("scause"))

OrdSymmetry == Permutations(Ords)

TypeOK ==
  /\ pc \in [Tasks -> {"idle", "dec", "resv", "rold", "cons", "run", "dead", "zombie"}]
  /\ cur \in 1..(NSteps + 1)
  /\ safety \subseteq SafetyIds
  /\ stop \in [Pieces -> {"none", "open", "esc"}]
=============================================================================
