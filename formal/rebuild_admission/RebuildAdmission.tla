-------------------------- MODULE RebuildAdmission --------------------------
(***************************************************************************)
(* M0 of the subsystem rebuild plan (OMN-22478, plan section 12), round 3: *)
(* admission, the stop and the holder as one state machine, written to    *)
(* the 11-06 text of sections 6.2, 7.1 and 7.5. It implements and wires   *)
(* nothing.                                                                *)
(*                                                                         *)
(* Mut = {} is the protocol; each name in Mut is one break, and the scope  *)
(* constants bound the environment only. Liveness and its fairness are in  *)
(* RebuildAdmissionExt.tla.                                                *)
(*                                                                         *)
(* The gate store is one serializable store (7.5), so a transaction on it  *)
(* is one action. Everything outside it is its own action: a source going  *)
(* bad, time passing, a process ending, a begun step landing, an engine    *)
(* acting on what it read before, a crash, a duplicate delivery. A split   *)
(* inside a transaction is a mutation.                                     *)
(*                                                                         *)
(* Abstractions, stated: no epoch number (an advance releases every live   *)
(* reservation, which is what comparing numbers decides); no wall clock    *)
(* (a stale table and an expiry are actions); readings are information and *)
(* the only reading the model keeps is the WORKING probe a stop's clear    *)
(* needs; a stop may open on any piece in BreakPieces (when it may open is *)
(* 3.5's guard on one action and is not a property here).                  *)
(***************************************************************************)
EXTENDS Integers, FiniteSets, Sequences, TLC

CONSTANTS
  o1, o2, o3,     \* see Ords
  Ords,           \* model values: ordinary tasks of piece 1 (a symmetry set)
  Others,         \* the other tasks by name, see KindOf
  SafetyIds,      \* safety failures that can be open together
  Limit,          \* the order overlay's WIP limit per piece
  MaxPiece,       \* pieces are 1..MaxPiece; MaxPiece stands for 8 or 9, the kept pieces
  NSteps,         \* order steps the holder takes; step i starts piece i + 1
  BreakPieces,    \* pieces a stop can open on
  FlightTasks,    \* lanes whose outward step is two actions, Begin and Act
  DupTasks,       \* lanes whose start command the bus delivers twice
  SrcOn,          \* safety sources can go bad
  PubDieOn,       \* the reading writer can die, so the table goes stale
  StoreDownOn,    \* the gate store and broker can stop answering
  HolderOn,       \* the order writer takes steps
  CrashOn,        \* engines crash, restart and hand over
  CreditOn,       \* credits are consumed and freed
  EscOn,          \* the stop escalation timer runs
  Mut             \* named breaks; {} is the protocol

M(x) == x \in Mut

Tasks == Ords \cup Others
Procs == {"H", "F"}
Pieces == 1..MaxPiece
Steps == 1..NSteps

\* A request: kind, piece tag, safety id, repository, and whether it matches its approved scope.
KindOf(t) ==
  CASE t \in Ords -> [k |-> "ord", p |-> 1, s |-> "-", r |-> "pub", sc |-> TRUE]
    [] t = "x"  -> [k |-> "ord", p |-> 1, s |-> "-", r |-> "pub", sc |-> FALSE]
    [] t = "u"  -> [k |-> "ord", p |-> 2, s |-> "-", r |-> "pub", sc |-> TRUE]
    [] t = "j"  -> [k |-> "join", p |-> 1, s |-> "-", r |-> "pub", sc |-> TRUE]
    [] t \in {"c1", "c1b"} -> [k |-> "cause", p |-> 1, s |-> "-", r |-> "pub", sc |-> TRUE]
    [] t = "c3" -> [k |-> "cause", p |-> MaxPiece, s |-> "-", r |-> "pub", sc |-> TRUE]
    [] t = "kf" -> [k |-> "kept", p |-> MaxPiece, s |-> "-", r |-> "pub", sc |-> TRUE]
    [] t = "g"  -> [k |-> "cont", p |-> 0, s |-> "-", r |-> "pub", sc |-> TRUE]
    [] t = "b"  -> [k |-> "b0", p |-> 0, s |-> "-", r |-> "pub", sc |-> TRUE]
    [] t \in {"r1", "r2"} -> [k |-> "repair", p |-> 0, s |-> "-", r |-> "pub", sc |-> TRUE]
    [] t = "s1" -> [k |-> "scause", p |-> 0, s |-> "s1", r |-> "pub", sc |-> TRUE]
K(t) == KindOf(t)

\* Every request the table can see, checked in every reachable state. r = "priv" is a private
\* repository outside step 0b's list; sc = FALSE is a task outside its approved scope.
AllKinds ==
     [k : {"ord"}, p : Pieces \ {MaxPiece}, s : {"-"}, r : {"pub", "priv"}, sc : BOOLEAN]
  \cup [k : {"join"}, p : 1..(MaxPiece - 2), s : {"-"}, r : {"pub", "priv"}, sc : BOOLEAN]
  \cup [k : {"kept"}, p : Pieces, s : {"-"}, r : {"pub", "priv"}, sc : BOOLEAN]
  \cup [k : {"cause"}, p : Pieces, s : {"-"}, r : {"pub", "priv"}, sc : {TRUE}]
  \cup [k : {"b0", "cont"}, p : {0}, s : {"-"}, r : {"pub", "priv"}, sc : BOOLEAN]
  \cup [k : {"ro", "repair", "other"}, p : {0}, s : {"-"}, r : {"pub", "priv"}, sc : {TRUE}]
  \cup [k : {"scause"}, p : {0}, s : SafetyIds, r : {"pub", "priv"}, sc : {TRUE}]

VARIABLES
  pc,       \* [Tasks -> idle, resv, run]; dec and zombie only under a mutation
  unadm,    \* lanes that started with no answer from admission (logged, 7.1 "No answer")
  fl,       \* [FlightTasks -> none, chk, log]: a begun outward step, fenced in the store or logged
  overrun,  \* [FlightTasks -> 0..2]: outward actions landed after the lane's stop was recorded
  checked,  \* fence_split only: lanes whose fence was read apart from the write
  msgs,     \* [DupTasks -> SUBSET {1, 2}]: start deliveries the bus still holds
  rd,       \* consume_split only: deliveries whose compare was read apart from the write
  starts,   \* [DupTasks -> 0..2]: starts made from the current reservation
  started,  \* pieces the order has started (the holder's record)
  stop,     \* [Pieces -> none, open, esc], in the store
  safety,   \* safety failures recorded in the store
  src,      \* [SafetyIds -> BOOLEAN]: the source itself reads bad
  fresh,    \* the reading table is inside 30 minutes
  pubUp,    \* the reading writer is alive
  credit,   \* freed places at hand (the freed-place guard)
  storeUp,  \* the gate store and broker answer
  broken,   \* pieces truly not working
  rows,     \* [Pieces -> SUBSET {"see"}]: go-and-see rows bound to the open stop
  wread,    \* [Pieces -> BOOLEAN]: a WORKING probe after the stop opened and the last lease
  cur,      \* the next order step
  up,       \* [Procs -> BOOLEAN]
  handed,   \* the record names the fallback writer as owner
  phase,    \* [Procs -> 0..NSteps]: the step the engine read and has not finished
  snap,     \* [Procs -> BOOLEAN]: fence_read only, ownership as the engine read it
  keys,     \* [Procs -> SUBSET Steps]: the one key store is keys["H"]
  effects,  \* [Steps -> 0..2]: times each step's effect ran
  pendKey,  \* [Procs -> 0..NSteps]: effect_then_key only
  pendEff,  \* [Procs -> 0..NSteps]: key_then_effect only
  ghSee,    \* ghost: [Pieces -> a go-and-see row since the stop opened and the last lease]
  ghRead,   \* ghost: [Pieces -> a WORKING probe since the stop opened and the last lease]
  dirty,    \* ghost: reservations alive across a write to a fact; {} unless an advance is missing
  bad       \* ghost: names of safety properties that broke; {} in the protocol

laneV  == <<pc, unadm, fl, overrun, checked, msgs, rd, starts>>
factV  == <<started, stop, safety, src, fresh, pubUp, credit, storeUp>>
evV    == <<broken, rows, wread>>
engV   == <<cur, up, handed, phase, snap, keys, effects, pendKey, pendEff>>
ghV    == <<ghSee, ghRead, dirty, bad>>
vars   == <<laneV, factV, evV, engV, ghV>>

Owner == IF handed THEN "F" ELSE "H"
Stopped(p) == p \in Pieces /\ stop[p] # "none"

\* ---- places ----
Wip(k) == k.k \in {"ord", "join"}
\* A join's first task is the next piece's work: counted there (O6) and stopped there.
WorkPiece(k) == IF k.k = "join" THEN k.p + 1 ELSE k.p
GatePiece(k) == IF k.k = "join" /\ M("join_own_piece") THEN k.p ELSE WorkPiece(k)
Holding == IF M("rows_only") THEN {"run"} ELSE {"resv", "run"}
\* What the gate counts, in the reservation's own transaction, leaving out e (the lane deciding).
GateCount(p, e) ==
  Cardinality({t \in Tasks : t # e /\ t \notin unadm /\ Wip(K(t)) /\ GatePiece(K(t)) = p /\ pc[t] \in Holding})
Room(k, e) == M("no_wip") \/ GateCount(GatePiece(k), e) < Limit
Holds(t) == pc[t] \in {"resv", "run"} /\ t \notin unadm
CauseFree(k, e) ==
  M("cause_no_slot") \/ \A t \in Tasks : (t # e /\ K(t).k = "cause" /\ K(t).p = k.p) => ~Holds(t)
RepairFree(e) == \A t \in Tasks : (t # e /\ K(t).k = "repair") => ~Holds(t)

\* The freed-place guard; safety, repair and stop cause lanes debit afterwards (7.1).
NeedsCredit(k) == CreditOn /\ k.k \notin {"ro", "scause", "repair", "cause"}
CreditOK(k) == ~NeedsCredit(k) \/ credit > 0
\* Step 0b: a safety cause lane passes, and so does the repair lane during an open failure.
PrivPass(k) == k.k = "scause" \/ (k.k = "repair" /\ safety # {})

\* ---- 7.1 as a step table, the order the text gives ----
Decide(k, e) ==
  IF k.k = "ro" /\ ~M("ro_after_private") THEN "admit"                                  \* 0
  ELSE IF k.r = "priv" /\ ~PrivPass(k) /\ ~M("private_open") THEN "refuse"              \* 0b
  ELSE IF k.k = "ro" THEN "admit"
  ELSE IF M("holder_in_engine") /\ HolderOn /\ ~up[Owner] THEN "refuse"                 \* (dropped)
  ELSE IF M("safety_refuses") /\ safety # {} /\ k.k # "scause" THEN "refuse"            \* (dropped)
  ELSE IF k.k = "scause" THEN (IF k.s \in safety THEN "admit" ELSE "refuse")            \* 1
  ELSE IF M("stale_refuses") /\ ~fresh /\ k.k # "repair" THEN "refuse"                  \* (dropped)
  ELSE IF k.k = "repair" THEN (IF ~fresh /\ RepairFree(e) THEN "admit" ELSE "refuse")   \* 2
  ELSE IF k.k = "cause" THEN                                                            \* 3
         (IF Stopped(k.p) /\ CauseFree(k, e) THEN "admit" ELSE "refuse")
  ELSE IF Wip(k) /\ Stopped(WorkPiece(k)) /\ ~M("stop_lets_own") THEN "refuse"
  ELSE IF Wip(k) /\ M("stop_holds_above") /\ \E q \in 1..(WorkPiece(k) - 1) : Stopped(q) THEN "refuse"
  ELSE IF k.k = "kept" /\ M("kept_refused_under_stop") /\ \E q \in Pieces : Stopped(q) THEN "refuse"
  ELSE IF Wip(k) THEN                                                                   \* 4
         (IF k.p \in started /\ (k.sc \/ M("no_scope")) /\ Room(k, e) /\ CreditOK(k) THEN "admit" ELSE "refuse")
  ELSE IF k.k = "b0" THEN (IF k.sc /\ CreditOK(k) THEN "admit" ELSE "refuse")           \* 4a
  ELSE IF k.k = "kept" THEN                                                             \* 5
         (IF k.p = MaxPiece /\ k.sc /\ CreditOK(k) THEN "admit" ELSE "refuse")
  ELSE IF k.k = "cont" THEN (IF k.sc /\ CreditOK(k) THEN "admit" ELSE "refuse")         \* 6
  ELSE "refuse"                                                                         \* 7

\* ---- the same rules written apart: the reasons the text allows a refusal for ----
\* Private repository, stop, scope, WIP, credit. A reading, a safety failure, a stale table, the
\* holder and an unanswered request are not reasons (RULING 2026-11-06T12:05:56Z, 12:10:28Z).
OutOfScope(k) ==
  \/ k.k = "other" \/ ~k.sc
  \/ (Wip(k) /\ k.p \notin started)
  \/ (k.k = "cause" /\ ~Stopped(k.p))
  \/ (k.k = "scause" /\ k.s \notin safety)
  \/ (k.k = "repair" /\ fresh)
  \/ (k.k = "kept" /\ k.p # MaxPiece)
Why(k, e) ==
  IF k.k = "ro" THEN {}
  ELSE    (IF k.r = "priv" /\ ~PrivPass(k) THEN {"private"} ELSE {})
     \cup (IF Wip(k) /\ Stopped(WorkPiece(k)) THEN {"stop"} ELSE {})
     \cup (IF k.k = "cause" /\ Stopped(k.p) /\ ~CauseFree(k, e) THEN {"stop"} ELSE {})
     \cup (IF OutOfScope(k) THEN {"scope"} ELSE {})
     \cup (IF (Wip(k) /\ GateCount(GatePiece(k), e) >= Limit) \/ (k.k = "repair" /\ ~RepairFree(e))
           THEN {"wip"} ELSE {})
     \cup (IF NeedsCredit(k) /\ credit <= 0 THEN {"credit"} ELSE {})

\* ---- the epoch: every write to a fact Decide reads releases the live reservations ----
Kill(site) ==
  IF M("no_epoch") \/ M("no_advance_" \o site) THEN pc
  ELSE [t \in Tasks |-> IF pc[t] = "resv" THEN "idle" ELSE pc[t]]
\* Reservations alive across a write. No guard reads this; a consume reports one consumed.
Older(site) == {t \in Tasks : Kill(site)[t] = "resv"} \cup dirty

Init ==
  /\ pc = [t \in Tasks |-> "idle"] /\ unadm = {}
  /\ fl = [t \in FlightTasks |-> "none"] /\ overrun = [t \in FlightTasks |-> 0] /\ checked = {}
  /\ msgs = [t \in DupTasks |-> {}] /\ rd = [t \in DupTasks |-> {}] /\ starts = [t \in DupTasks |-> 0]
  /\ started = {1} /\ stop = [p \in Pieces |-> "none"] /\ safety = {}
  /\ src = [s \in SafetyIds |-> FALSE] /\ fresh = TRUE /\ pubUp = TRUE /\ credit = 1 /\ storeUp = TRUE
  /\ broken = {} /\ rows = [p \in Pieces |-> {}] /\ wread = [p \in Pieces |-> FALSE]
  /\ cur = 1 /\ up = [x \in Procs |-> TRUE] /\ handed = FALSE /\ phase = [x \in Procs |-> 0]
  /\ snap = [x \in Procs |-> FALSE] /\ keys = [x \in Procs |-> {}] /\ effects = [i \in Steps |-> 0]
  /\ pendKey = [x \in Procs |-> 0] /\ pendEff = [x \in Procs |-> 0]
  /\ ghSee = [p \in Pieces |-> FALSE] /\ ghRead = [p \in Pieces |-> FALSE] /\ dirty = {} /\ bad = {}

\* ---- admission: each a store transaction ----
Reserve(t) ==
  /\ pc[t] = "idle" /\ storeUp
  /\ Decide(K(t), t) = "admit"
  /\ pc' = [pc EXCEPT ![t] = IF M("late_bind") THEN "dec" ELSE "resv"]
  /\ msgs' = IF t \in DupTasks THEN [msgs EXCEPT ![t] = {1, 2}] ELSE msgs
  /\ starts' = IF t \in DupTasks THEN [starts EXCEPT ![t] = 0] ELSE starts
  /\ UNCHANGED <<unadm, fl, overrun, checked, rd, factV, evV, engV, ghV>>
Bind(t) ==       \* late_bind only: the epoch is read after the facts were
  /\ M("late_bind") /\ pc[t] = "dec" /\ storeUp
  /\ pc' = [pc EXCEPT ![t] = "resv"]
  /\ UNCHANGED <<unadm, fl, overrun, checked, msgs, rd, starts, factV, evV, engV, ghV>>
Expire(t) ==     \* the reservation's lifetime ends; the store re-lets its place
  /\ pc[t] \in {"resv", "dec"} /\ storeUp
  /\ pc' = [pc EXCEPT ![t] = "idle"] /\ dirty' = dirty \ {t}
  /\ UNCHANGED <<unadm, fl, overrun, checked, msgs, rd, starts, factV, evV, engV, ghSee, ghRead, bad>>
\* The compare of the consume: the reservation live and unused, and the credit.
Used(t) == pc[t] = "resv" \/ (M("no_used_check") /\ pc[t] = "run")
CreditCmp(t) == (NeedsCredit(K(t)) /\ ~M("credit_no_compare")) => credit > 0
\* The start itself. A cause lane's lease voids the evidence before it (6.2).
StartBody(t) ==
  LET c == K(t).k = "cause"
      q == K(t).p
  IN
  /\ pc' = [pc EXCEPT ![t] = "run"]
  /\ credit' = IF NeedsCredit(K(t)) THEN credit - 1 ELSE credit
  /\ rows' = IF c /\ ~M("see_before_lease") THEN [rows EXCEPT ![q] = {}] ELSE rows
  /\ wread' = IF c /\ ~M("see_before_lease") THEN [wread EXCEPT ![q] = FALSE] ELSE wread
  /\ ghSee' = IF c THEN [ghSee EXCEPT ![q] = FALSE] ELSE ghSee
  /\ ghRead' = IF c THEN [ghRead EXCEPT ![q] = FALSE] ELSE ghRead
  /\ dirty' = dirty \ {t}
  /\ bad' = bad \cup (IF Why(K(t), t) # {} THEN {"start"} ELSE {})
                \cup (IF t \in dirty THEN {"epoch"} ELSE {})
                \cup (IF credit' < 0 THEN {"credit"} ELSE {})
Consume(t) ==    \* one delivery
  /\ t \notin DupTasks /\ pc[t] = "resv" /\ storeUp /\ CreditCmp(t)
  /\ StartBody(t)
  /\ UNCHANGED <<unadm, fl, overrun, checked, msgs, rd, starts, started, stop, safety, src, fresh, pubUp,
                 storeUp, broken, engV>>
ConsumeD(t, d) ==   \* a delivery of a start the bus may deliver twice: compare and write as one
  /\ t \in DupTasks /\ d \in msgs[t] /\ ~M("consume_split") /\ storeUp /\ Used(t) /\ CreditCmp(t)
  /\ StartBody(t)
  /\ msgs' = [msgs EXCEPT ![t] = @ \ {d}]
  /\ starts' = [starts EXCEPT ![t] = IF @ < 2 THEN @ + 1 ELSE @]
  /\ UNCHANGED <<unadm, fl, overrun, checked, rd, started, stop, safety, src, fresh, pubUp,
                 storeUp, broken, engV>>
DropD(t, d) ==      \* the compare refuses a delivery: used, released or expired
  /\ t \in DupTasks /\ d \in msgs[t] /\ storeUp /\ ~Used(t)
  /\ msgs' = [msgs EXCEPT ![t] = @ \ {d}]
  /\ UNCHANGED <<pc, unadm, fl, overrun, checked, rd, starts, factV, evV, engV, ghV>>
ConsRead(t, d) ==   \* consume_split only: the compare read
  /\ M("consume_split") /\ t \in DupTasks /\ d \in msgs[t] /\ storeUp /\ Used(t)
  /\ msgs' = [msgs EXCEPT ![t] = @ \ {d}] /\ rd' = [rd EXCEPT ![t] = @ \cup {d}]
  /\ UNCHANGED <<pc, unadm, fl, overrun, checked, starts, factV, evV, engV, ghV>>
ConsWrite(t, d) ==  \* consume_split only: the write, on what was read
  /\ t \in DupTasks /\ d \in rd[t] /\ storeUp /\ CreditCmp(t)
  /\ StartBody(t)
  /\ rd' = [rd EXCEPT ![t] = @ \ {d}]
  /\ starts' = [starts EXCEPT ![t] = IF @ < 2 THEN @ + 1 ELSE @]
  /\ UNCHANGED <<unadm, fl, overrun, checked, msgs, started, stop, safety, src, fresh, pubUp,
                 storeUp, broken, engV>>
NoAnswer(t) ==      \* admission gives no answer by the deadline: the start proceeds, logged (7.1)
  /\ pc[t] = "idle" /\ ~storeUp /\ ~M("down_refuses")
  /\ pc' = [pc EXCEPT ![t] = "run"] /\ unadm' = unadm \cup {t}
  /\ UNCHANGED <<fl, overrun, checked, msgs, rd, starts, factV, evV, engV, ghV>>

\* ---- a running lane ----
Ends(t) ==          \* the process exits and is read as gone: only this frees what it held
  /\ pc[t] \in {"run", "zombie"}
  /\ pc' = [pc EXCEPT ![t] = "idle"] /\ unadm' = unadm \ {t}
  /\ fl' = IF t \in FlightTasks THEN [fl EXCEPT ![t] = "none"] ELSE fl
  /\ overrun' = IF t \in FlightTasks THEN [overrun EXCEPT ![t] = 0] ELSE overrun
  /\ checked' = checked \ {t}
  /\ UNCHANGED <<msgs, rd, starts, factV, evV, engV, ghV>>
CloseRow(t) ==      \* close_frees only: a TERMINAL row frees the place while the process lives
  /\ M("close_frees") /\ pc[t] = "run" /\ Wip(K(t)) /\ t \notin unadm
  /\ pc' = [pc EXCEPT ![t] = "zombie"]
  /\ UNCHANGED <<unadm, fl, overrun, checked, msgs, rd, starts, factV, evV, engV, ghV>>
\* The fence: a running lane of a stopped piece takes no outward step; cause lanes and kept
\* fixes are excepted because they are not ordinary or join work.
Fenced(t) == Wip(K(t)) /\ Stopped(WorkPiece(K(t)))
Check(t) ==         \* fence_split only: the fence read apart from the write
  /\ M("fence_split") /\ t \in FlightTasks /\ pc[t] = "run" /\ fl[t] = "none" /\ storeUp
  /\ ~Fenced(t) /\ t \notin checked
  /\ checked' = checked \cup {t}
  /\ UNCHANGED <<pc, unadm, fl, overrun, msgs, rd, starts, factV, evV, engV, ghV>>
Begin(t) ==         \* a store write with the fence in it; with no answer the step proceeds, logged
  /\ t \in FlightTasks /\ pc[t] = "run" /\ fl[t] = "none"
  /\ IF storeUp
     THEN /\ IF M("fence_split") THEN t \in checked ELSE (M("fence_off") \/ ~Fenced(t))
          /\ fl' = [fl EXCEPT ![t] = "chk"]
          /\ bad' = IF Fenced(t) THEN bad \cup {"begin"} ELSE bad
          /\ overrun' = IF Fenced(t) THEN overrun ELSE [overrun EXCEPT ![t] = 0]
     ELSE /\ fl' = [fl EXCEPT ![t] = "log"]
          /\ UNCHANGED <<bad, overrun>>
  /\ checked' = checked \ {t}
  /\ UNCHANGED <<pc, unadm, msgs, rd, starts, factV, evV, engV, ghSee, ghRead, dirty>>
Act(t) ==           \* the push, merge or deploy, outside the store: it may land after a stop opens
  /\ t \in FlightTasks /\ fl[t] # "none"
  /\ fl' = IF M("fence_once") THEN fl ELSE [fl EXCEPT ![t] = "none"]
  /\ overrun' = IF fl[t] = "chk" /\ Fenced(t) /\ overrun[t] < 2
                THEN [overrun EXCEPT ![t] = @ + 1] ELSE overrun
  /\ UNCHANGED <<pc, unadm, checked, msgs, rd, starts, factV, evV, engV, ghV>>
Fix(t) ==           \* a cause lane may fix its piece; nothing assumes it does
  /\ K(t).k = "cause" /\ pc[t] = "run" /\ K(t).p \in broken
  /\ broken' = broken \ {K(t).p}
  /\ UNCHANGED <<laneV, factV, rows, wread, engV, ghV>>
SafetyFix(t) ==
  /\ K(t).k = "scause" /\ pc[t] = "run" /\ src[K(t).s]
  /\ src' = [src EXCEPT ![K(t).s] = FALSE]
  /\ UNCHANGED <<laneV, started, stop, safety, fresh, pubUp, credit, storeUp, evV, engV, ghV>>
RepairWorks(t) ==   \* it may never happen
  /\ K(t).k = "repair" /\ pc[t] = "run" /\ ~pubUp
  /\ pubUp' = TRUE
  /\ UNCHANGED <<laneV, started, stop, safety, src, fresh, credit, storeUp, evV, engV, ghV>>
CreditFreed ==      \* a merge or closure read
  /\ CreditOn /\ credit = 0 /\ credit' = 1
  /\ UNCHANGED <<laneV, started, stop, safety, src, fresh, pubUp, storeUp, evV, engV, ghV>>

\* ---- safety failures: their own sources; each starts a cause lane and decides no other start ----
SrcBad(s) ==
  /\ SrcOn /\ ~src[s] /\ src' = [src EXCEPT ![s] = TRUE]
  /\ UNCHANGED <<laneV, started, stop, safety, fresh, pubUp, credit, storeUp, evV, engV, ghV>>
SafetyOpens(s) ==
  /\ src[s] /\ s \notin safety /\ storeUp
  /\ safety' = safety \cup {s} /\ pc' = Kill("safety") /\ dirty' = Older("safety")
  /\ UNCHANGED <<unadm, fl, overrun, checked, msgs, rd, starts, started, stop, src, fresh, pubUp, credit,
                 storeUp, evV, engV, ghSee, ghRead, bad>>
SafetyCloses(s) ==  \* on a clean read of its own source (one read: an abstraction)
  /\ s \in safety /\ ~src[s] /\ storeUp
  /\ safety' = safety \ {s} /\ pc' = Kill("safety") /\ dirty' = Older("safety")
  /\ UNCHANGED <<unadm, fl, overrun, checked, msgs, rd, starts, started, stop, src, fresh, pubUp, credit,
                 storeUp, evV, engV, ghSee, ghRead, bad>>

\* ---- the reading table: information, read by the repair lane's admission only ----
PubDies ==
  /\ PubDieOn /\ pubUp /\ pubUp' = FALSE
  /\ UNCHANGED <<laneV, started, stop, safety, src, fresh, credit, storeUp, evV, engV, ghV>>
GoesStale ==        \* time passing: no writer, so no advance
  /\ fresh /\ ~pubUp /\ fresh' = FALSE
  /\ UNCHANGED <<laneV, started, stop, safety, src, pubUp, credit, storeUp, evV, engV, ghV>>
Publish ==          \* the table comes back: a write the repair lane's decision reads
  /\ pubUp /\ storeUp /\ ~fresh
  /\ fresh' = TRUE /\ pc' = Kill("publish") /\ dirty' = Older("publish")
  /\ UNCHANGED <<unadm, fl, overrun, checked, msgs, rd, starts, started, stop, safety, src, pubUp, credit,
                 storeUp, evV, engV, ghSee, ghRead, bad>>

\* ---- the stop; its evidence binds to the open stop ----
Breaks(p) ==
  /\ p \in BreakPieces /\ stop[p] = "none" /\ storeUp
  /\ stop' = [stop EXCEPT ![p] = "open"] /\ broken' = broken \cup {p}
  /\ rows' = IF M("rows_survive") THEN rows ELSE [rows EXCEPT ![p] = {}]
  /\ wread' = IF M("reading_survives") THEN wread ELSE [wread EXCEPT ![p] = FALSE]
  /\ ghSee' = [ghSee EXCEPT ![p] = FALSE] /\ ghRead' = [ghRead EXCEPT ![p] = FALSE]
  /\ pc' = Kill("stop") /\ dirty' = Older("stop")
  /\ UNCHANGED <<unadm, fl, overrun, checked, msgs, rd, starts, started, safety, src, fresh, pubUp, credit,
                 storeUp, engV, bad>>
SeeRow(p) ==        \* a go-and-see row by a read-only lane; it counts once the store holds it
  /\ Stopped(p) /\ storeUp /\ rows[p] = {}
  /\ rows' = [rows EXCEPT ![p] = {"see"}] /\ ghSee' = [ghSee EXCEPT ![p] = TRUE]
  /\ UNCHANGED <<laneV, factV, broken, wread, engV, ghRead, dirty, bad>>
Probe(p) ==         \* a WORKING probe, dated after the stop and the last lease
  /\ Stopped(p) /\ storeUp /\ p \notin broken /\ ~wread[p]
  /\ wread' = [wread EXCEPT ![p] = TRUE] /\ ghRead' = [ghRead EXCEPT ![p] = TRUE]
  /\ UNCHANGED <<laneV, factV, broken, rows, engV, ghSee, dirty, bad>>
Clears(p) ==
  /\ Stopped(p) /\ storeUp
  /\ "see" \in rows[p] \/ M("clear_no_rows")
  /\ wread[p] \/ M("clear_no_reading")
  /\ stop' = [stop EXCEPT ![p] = "none"]
  /\ rows' = IF M("rows_survive") THEN rows ELSE [rows EXCEPT ![p] = {}]
  /\ wread' = IF M("reading_survives") THEN wread ELSE [wread EXCEPT ![p] = FALSE]
  /\ bad' = bad \cup {"cleared"} \cup (IF ghSee[p] /\ ghRead[p] /\ p \notin broken THEN {} ELSE {"clear"})
  /\ ghSee' = [ghSee EXCEPT ![p] = FALSE] /\ ghRead' = [ghRead EXCEPT ![p] = FALSE]
  /\ pc' = Kill("clear") /\ dirty' = Older("clear")
  /\ UNCHANGED <<unadm, fl, overrun, checked, msgs, rd, starts, started, safety, src, fresh, pubUp, credit,
                 storeUp, broken, engV>>
\* node_stop_escalation_effect: its own timer, independent of the table, the holder and the reading
\* writer; it writes one row in the store and the stop stays open.
Escalates(p) ==
  /\ EscOn /\ stop[p] = "open" /\ storeUp /\ (M("escalate_on_reader") => pubUp)
  /\ stop' = [stop EXCEPT ![p] = "esc"]
  /\ UNCHANGED <<laneV, started, safety, src, fresh, pubUp, credit, storeUp, evV, engV, ghV>>

\* ---- the store ----
StoreDown ==
  /\ StoreDownOn /\ storeUp /\ storeUp' = FALSE
  /\ UNCHANGED <<laneV, started, stop, safety, src, fresh, pubUp, credit, evV, engV, ghV>>
StoreBack ==
  /\ ~storeUp /\ storeUp' = TRUE
  /\ UNCHANGED <<laneV, started, stop, safety, src, fresh, pubUp, credit, evV, engV, ghV>>

\* ---- the holder (7.5): the record, the keys and the fence are in the store, outside both
\* engines. A step is one put: fence compare, key compare, key, effect, and the order write. ----
FenceCmp(x) ==
  IF M("no_fence") THEN TRUE ELSE IF M("fence_read") THEN snap[x] ELSE x = Owner
KeyStore(x) == IF M("keys_per_engine") THEN x ELSE "H"
KeyCmp(i, x) == ~M("unkeyed") /\ i \in keys[KeyStore(x)]
EngRead(x) ==       \* the engine's own history: what it replays after a restart, stale or not
  /\ HolderOn /\ up[x] /\ phase[x] = 0 /\ cur <= NSteps
  /\ phase' = [phase EXCEPT ![x] = cur]
  /\ snap' = [snap EXCEPT ![x] = (x = Owner)]
  /\ UNCHANGED <<laneV, factV, evV, cur, up, handed, keys, effects, pendKey, pendEff, ghV>>
StepPut(x) ==
  LET i == phase[x] IN
  /\ HolderOn /\ i # 0 /\ up[x] /\ storeUp /\ pendKey[x] = 0 /\ pendEff[x] = 0
  /\ FenceCmp(x) /\ ~KeyCmp(i, x)
  /\ effects' = IF M("key_then_effect") THEN effects ELSE [effects EXCEPT ![i] = IF @ < 2 THEN @ + 1 ELSE @]
  /\ pendEff' = IF M("key_then_effect") THEN [pendEff EXCEPT ![x] = i] ELSE pendEff
  /\ keys' = IF M("effect_then_key") THEN keys ELSE [keys EXCEPT ![KeyStore(x)] = @ \cup {i}]
  /\ pendKey' = IF M("effect_then_key") THEN [pendKey EXCEPT ![x] = i] ELSE pendKey
  /\ cur' = IF cur = i THEN i + 1 ELSE cur
  /\ started' = IF cur = i THEN started \cup {i + 1} ELSE started
  /\ pc' = IF cur = i THEN Kill("step") ELSE pc
  /\ dirty' = IF cur = i THEN Older("step") ELSE dirty
  /\ bad' = bad \cup (IF x # Owner THEN {"owner"} ELSE {})
                \cup (IF ~M("key_then_effect") /\ effects[i] >= 1 THEN {"repeat"} ELSE {})
  /\ UNCHANGED <<unadm, fl, overrun, checked, msgs, rd, starts, stop, safety, src, fresh, pubUp, credit,
                 storeUp, evV, up, handed, phase, snap, ghSee, ghRead>>
KeyWrite(x) ==      \* effect_then_key only: the key after the effect
  /\ pendKey[x] # 0 /\ up[x] /\ storeUp
  /\ keys' = [keys EXCEPT ![KeyStore(x)] = @ \cup {pendKey[x]}] /\ pendKey' = [pendKey EXCEPT ![x] = 0]
  /\ UNCHANGED <<laneV, factV, evV, cur, up, handed, phase, snap, effects, pendEff, ghV>>
EffWrite(x) ==      \* key_then_effect only: the effect after the key
  /\ pendEff[x] # 0 /\ up[x]
  /\ effects' = [effects EXCEPT ![pendEff[x]] = IF @ < 2 THEN @ + 1 ELSE @]
  /\ bad' = IF effects[pendEff[x]] >= 1 THEN bad \cup {"repeat"} ELSE bad
  /\ pendEff' = [pendEff EXCEPT ![x] = 0]
  /\ UNCHANGED <<laneV, factV, evV, cur, up, handed, phase, snap, keys, pendKey, ghSee, ghRead, dirty>>
StepDone(x) ==      \* the compare refused a stale engine or a replay, or the step landed
  /\ phase[x] # 0 /\ up[x] /\ storeUp /\ pendKey[x] = 0 /\ pendEff[x] = 0
  /\ ~FenceCmp(x) \/ KeyCmp(phase[x], x)
  /\ phase' = [phase EXCEPT ![x] = 0]
  /\ UNCHANGED <<laneV, factV, evV, cur, up, handed, snap, keys, effects, pendKey, pendEff, ghV>>
Crash(x) ==         \* a kill at any point; the engine's history (phase) survives, unwritten work does not
  /\ HolderOn /\ CrashOn /\ up[x]
  /\ up' = [up EXCEPT ![x] = FALSE]
  /\ pendKey' = [pendKey EXCEPT ![x] = 0] /\ pendEff' = [pendEff EXCEPT ![x] = 0]
  /\ UNCHANGED <<laneV, factV, evV, cur, handed, phase, snap, keys, effects, ghV>>
Recover(x) ==       \* the supervisor restarts it
  /\ ~up[x] /\ up' = [up EXCEPT ![x] = TRUE]
  /\ UNCHANGED <<laneV, factV, evV, cur, handed, phase, snap, keys, effects, pendKey, pendEff, ghV>>
Handover ==         \* one way, a fenced write in the store; the old engine may not know
  /\ HolderOn /\ CrashOn /\ ~handed /\ storeUp
  /\ handed' = TRUE /\ pc' = Kill("handover") /\ dirty' = Older("handover")
  /\ UNCHANGED <<unadm, fl, overrun, checked, msgs, rd, starts, factV, evV, cur, up, phase, snap, keys,
                 effects, pendKey, pendEff, ghSee, ghRead, bad>>

Next ==
  \/ \E t \in Tasks :
       \/ Reserve(t) \/ Bind(t) \/ Expire(t) \/ Consume(t) \/ NoAnswer(t)
       \/ Ends(t) \/ CloseRow(t) \/ Check(t) \/ Begin(t) \/ Act(t)
       \/ Fix(t) \/ SafetyFix(t) \/ RepairWorks(t)
  \/ \E t \in DupTasks, d \in {1, 2} : ConsumeD(t, d) \/ DropD(t, d) \/ ConsRead(t, d) \/ ConsWrite(t, d)
  \/ \E s \in SafetyIds : SrcBad(s) \/ SafetyOpens(s) \/ SafetyCloses(s)
  \/ \E p \in Pieces : Breaks(p) \/ SeeRow(p) \/ Probe(p) \/ Clears(p) \/ Escalates(p)
  \/ \E x \in Procs : EngRead(x) \/ StepPut(x) \/ KeyWrite(x) \/ EffWrite(x) \/ StepDone(x) \/ Crash(x) \/ Recover(x)
  \/ CreditFreed \/ PubDies \/ GoesStale \/ Publish \/ StoreDown \/ StoreBack \/ Handover

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* P1. Admission follows 7.1 as rewritten on 11-06; no step of a stopped   *)
(* piece begins after the stop is recorded, one begun before may land.     *)
(***************************************************************************)
\* The step table and the reasons written apart agree, both ways, for every request in every
\* reachable state: nothing is admitted against a reason ...
DecideFollowsRules == \A k \in AllKinds : Decide(k, "none") = "admit" => Why(k, "none") = {}
\* ... and nothing is refused without one: no reading, safety failure, stale table or holder.
NoRefusalOutsideRules == \A k \in AllKinds : Decide(k, "none") = "refuse" => Why(k, "none") # {}
\* Carried by the epoch: a start consumed later has no reason against it at the consume.
NoStartAgainstRules == "start" \notin bad
NoStepBeginsAfterStop == "begin" \notin bad
AtMostOneStepAfterStop == \A t \in FlightTasks : overrun[t] <= 1

(***************************************************************************)
(* P2. Processes per piece within the limit, counted until read gone.      *)
(* Alive counts processes the store admitted; an unanswered start is       *)
(* logged and not counted (7.1 "No answer"), see WitOutageOverLimit.       *)
(***************************************************************************)
TruePiece(k) == WorkPiece(k)
Alive(p) ==
  Cardinality({t \in Tasks : t \notin unadm /\ Wip(K(t)) /\ TruePiece(K(t)) = p /\ pc[t] \in {"run", "zombie"}})
AliveWithinLimit == \A p \in Pieces : Alive(p) <= Limit
OneCauseLane == \A p \in Pieces :
  Cardinality({t \in Tasks : K(t).k = "cause" /\ K(t).p = p /\ t \notin unadm /\ pc[t] \in {"resv", "run"}}) <= 1
CreditNotOverdrawn == "credit" \notin bad

(***************************************************************************)
(* P3. A stop clears only with its evidence (liveness: Ext).               *)
(***************************************************************************)
StopClearsOnlyWithEvidence == "clear" \notin bad

(***************************************************************************)
(* P4. No step repeated or lost under kills; only the owner's write lands; *)
(* a reservation starts at most once and never after an advance.           *)
(***************************************************************************)
RestartRepeatsNoStep == "repeat" \notin bad
NoLostStep == \A i \in Steps : i < cur => (effects[i] >= 1 \/ \E x \in Procs : pendEff[x] = i)
NoWriteByLostOwner == "owner" \notin bad
ConsumedAtMostOnce == \A t \in DupTasks : starts[t] <= 1
NeverConsumedAfterEpoch == "epoch" \notin bad

(***************************************************************************)
(* Witnesses: each must be violated, so what it names is reachable.        *)
(***************************************************************************)
AnyRun(k) == \E t \in Tasks : K(t).k = k /\ pc[t] = "run" /\ t \notin unadm
WitOrdRunsDuringSafety == ~(safety # {} /\ \E t \in Ords : pc[t] = "run" /\ t \notin unadm)
WitOrdRunsOnStaleTable == ~(~fresh /\ \E t \in Ords : pc[t] = "run" /\ t \notin unadm)
WitAboveStopRuns == ~(Stopped(1) /\ \E t \in Tasks : K(t).k = "ord" /\ K(t).p = 2 /\ pc[t] = "run" /\ t \notin unadm)
WitUnadmittedStart == unadm = {}
WitOutageOverLimit == \A p \in Pieces :
  Cardinality({t \in Tasks : Wip(K(t)) /\ TruePiece(K(t)) = p /\ pc[t] = "run"}) <= Limit
WitLimitReached == Alive(1) < Limit
WitStopCleared == "cleared" \notin bad   \* "cleared" marks a clear; no property reads it
WitEscalated == \A p \in Pieces : stop[p] # "esc"
WitLandsAfterStop == \A t \in FlightTasks : overrun[t] = 0
WitLoggedStepUnderStop == ~(\E t \in FlightTasks : fl[t] = "log" /\ Fenced(t))
WitOldEngineStillActing == ~(handed /\ up["H"] /\ phase["H"] # 0)
WitFallbackAdvances == ~(handed /\ cur > NSteps)
WitDuplicateDelivered == ~(\E t \in DupTasks : pc[t] = "run" /\ msgs[t] # {})
WitPieceStarted == cur = 1

OrdSymmetry == Permutations(Ords)

TypeOK ==
  /\ pc \in [Tasks -> {"idle", "dec", "resv", "run", "zombie"}]
  /\ unadm \subseteq Tasks /\ safety \subseteq SafetyIds /\ started \subseteq Pieces
  /\ stop \in [Pieces -> {"none", "open", "esc"}]
  /\ cur \in 1..(NSteps + 1)
  /\ effects \in [Steps -> 0..2]
=============================================================================
