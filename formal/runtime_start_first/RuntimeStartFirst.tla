------------------------ MODULE RuntimeStartFirst ------------------------
(* The shared-listener start-first runtime handover, design revision 4 of    *)
(* OMN-22151 stage 2 (task T0, OMN-22376). It replaces the draft model of    *)
(* the retired front container.                                              *)
(*                                                                            *)
(* One spec, three pieces, each switched on by a constant, because the        *)
(* product of all three is not checkable and the pieces meet only through     *)
(* the agent's steps:                                                         *)
(*   HttpOn     A1 A5 A6 A2 A3: one shared accept queue (the listening       *)
(*              socket L), provisional and confirmed attachment sessions,     *)
(*              detach grants, the revocation hold, stop before teardown.     *)
(*              A poll is not an action: every state is judged by what a      *)
(*              poll arriving in it would meet, so no poll history is kept.   *)
(*   BusOn      S1 to S5: passive standby, stop as the fence, commits that    *)
(*              carry an epoch, seeding over three groups from the source's   *)
(*              snapshot, the owner record, the first-start rule (S3a).       *)
(*              One partition of LogLen records; a record's effect and its    *)
(*              synchronous commit are two steps (the X2 gap).                *)
(*   InstallOn  section 7: the first-install window, rollback, roll-forward,  *)
(*              the agent's hang, crash and systemd watchdog. Time is the     *)
(*              window clock `now`, which starts at the legacy stop.          *)
(* Redeploy (section 6) is the agent program `pc`, used when InstallOn is     *)
(* FALSE; the installed pool then holds runtime 1 as owner and each redeploy  *)
(* creates the next runtime id in the other slot (odd ids slot a, even b).    *)
(* Runtime 0 is the legacy runtime of the install: auto-commit, no standby,   *)
(* its commits carry no epoch (ep -1) and count as legEp.                    *)
(*                                                                            *)
(* Faults. The flag `faulted` records an HTTP-side fault of R1: a crash of a  *)
(* runtime that accepts on L, a hold that outlasts its bound, a successor     *)
(* that never attaches. AcceptedIsReady and NoRefusalWithoutFault are stated  *)
(* for runs without one, as the design argues. The effect-to-commit gap (X2)  *)
(* is a counter per record (x2), and ReplayOnlyX2 allows a record to run      *)
(* again only that many times. Residuals R2, R3, R4 and R5 and the            *)
(* legacy's X1 are not properties; see results/design_review_verdict.txt.     *)
(*                                                                            *)
(* Not modelled, by choice (each is a fault or belongs to a later task):     *)
(*   an aborted redeploy (never_ready, listener_not_attached) and the agent's *)
(*   crash recovery inside a redeploy (T4); L across the agent's own restarts *)
(*   and re-exec, and B unreachable (T3); the token mapping of the legacy     *)
(*   group ids and the start journal (T2); the offset retention of R5 and the *)
(*   final file; a non-revocation readiness loss (a fault by definition, so   *)
(*   no property constrains it); X1, the legacy's auto-commit past a polled   *)
(*   batch (the legacy commits only processed positions here). One partition  *)
(*   stands for the per-partition step of S4; the commit and the seek of an   *)
(*   activation are one step, so a crash inside it is a crash before it.      *)
(*                                                                            *)
(* MUT names one mutant (a deliberate break of one rule); "none" is the       *)
(* design. Each mutant must make TLC report exactly one named invariant.      *)
EXTENDS Naturals, Integers, FiniteSets, TLC

CONSTANTS
    Runtimes,       \* runtime ids; 0 is the legacy runtime, odd ids slot a, even slot b
    HttpOn, BusOn, InstallOn,
    MaxRedeploy,    \* redeploys of the installed pool
    LogLen,         \* records of the one partition
    MaxCrash,       \* crashes (MaxAgentCrash 2 and the others are set in each cfg)
    MaxRevoke,      \* rebalance revocations
    MaxAgentCrash,  \* agent hangs and crashes inside the install window
    WMax, RB, RF,   \* install window, rollback and roll-forward bounds, in ticks
    W, G,           \* systemd watchdog timeout and agent restart time, in ticks
    WBound,         \* the bound WindowBounded claims
    ClockMax,       \* time stops here; larger than WBound so a breach is reachable
    MUT

Offsets == 0..LogLen - 1
Groups == {0, 1, 2}
Grp(r) == IF r = 0 THEN 0 ELSE IF r % 2 = 1 THEN 1 ELSE 2

NoC == [has |-> FALSE, off |-> 0, ep |-> 0]

VARIABLES
    S,          \* per runtime record, see NewRt
    now, agent, tAt, nAg, ip,     \* install clock, agent state, install phase
    pc, rd, cur, handover,        \* redeploy program, redeploys, owner, B's flag
    grantTo, Lopen,               \* B's outstanding grant, the kernel socket L
    rec, commit, effects, x2,     \* owner record, group commits, effect and X2 counts
    faulted, nCrash, nRev, legEp

vars == <<S, now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
          rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

NewRt == [proc |-> "up", ready |-> FALSE, role |-> "passive", lst |-> "detached",
          conf |-> FALSE, stop |-> FALSE, rev |-> FALSE, pos |-> 0,
          infl |-> FALSE, ep |-> 0]
NoRt == [NewRt EXCEPT !.proc = "none"]

Up(r) == S[r].proc \in {"up", "stopping"}
Accepting(r) == S[r].lst \in {"attaching", "attached", "holding"}
Acc == {r \in Runtimes : Accepting(r)}
ConfAtt(r) == S[r].conf /\ S[r].lst \in {"attached", "holding"}
OtherConf(r) == \E s \in Runtimes \ {r} : ConfAtt(s)
Consuming(r) == Up(r) /\ (S[r].role = "active" \/ MUT = "no_standby")
Closed == \E r \in Runtimes : S[r].lst = "attached" /\ S[r].ready /\ (r = 0 \/ Lopen)
\* The window runs from the legacy stop to the first 200 on the host port. An
\* outage after that (ip "post": N crashing once attached) is a fault R1, not the
\* window.
WindowOpen == InstallOn /\ ~Closed /\
    ip \in {"stopl", "wait_attach", "rb1", "rb3", "rb4", "rf1", "rf2"}

Eff(c) == IF c.ep < 0 THEN legEp ELSE c.ep
BestOf(cs, hs) == CHOOSE h \in hs : \A k \in hs : Eff(cs[h]) >= Eff(cs[k])
Stamp(r) == IF r = 0 THEN -1 ELSE S[r].ep

StopRec(x) == [x EXCEPT !.proc = "stopping", !.stop = TRUE, !.ready = FALSE,
               !.lst = "detached", !.conf = FALSE]

(* ---------------------------------------------------------------------- *)
(* Init                                                                    *)
(* ---------------------------------------------------------------------- *)
Init ==
    /\ now = 0 /\ agent = "up" /\ tAt = 0 /\ nAg = 0 /\ ip = "pre"
    /\ pc = "idle" /\ rd = 0 /\ handover = FALSE /\ grantTo = 0
    /\ faulted = FALSE /\ nCrash = 0 /\ nRev = 0 /\ legEp = 0
    /\ x2 = [o \in Offsets |-> 0]
    /\ IF InstallOn
       THEN \E k \in 0..1 :
              /\ cur = 0 /\ Lopen = FALSE
              /\ S = [r \in Runtimes |->
                       IF r = 0 THEN [NewRt EXCEPT !.ready = TRUE, !.role = "active",
                                       !.lst = "attached", !.conf = TRUE, !.pos = k]
                       ELSE IF r = 1 THEN [NewRt EXCEPT !.ready = TRUE]
                       ELSE NoRt]
              /\ rec = [ep |-> 0, inst |-> 0, src |-> 0, srcg |-> 0, snap |-> NoC]
              /\ commit = [g \in Groups |->
                            IF g = 0 /\ k = 1 THEN [has |-> TRUE, off |-> 1, ep |-> -1] ELSE NoC]
              /\ effects = [o \in Offsets |-> IF o < k THEN 1 ELSE 0]
       ELSE /\ cur = 1 /\ Lopen = TRUE
            /\ S = [r \in Runtimes |->
                     IF r = 1 THEN [NewRt EXCEPT !.ready = TRUE, !.role = "active",
                                     !.lst = "attached", !.conf = TRUE, !.ep = 1]
                     ELSE NoRt]
            /\ rec = [ep |-> 1, inst |-> 1, src |-> 0, srcg |-> 0, snap |-> NoC]
            /\ commit = [g \in Groups |->
                          IF g = 1 THEN [has |-> TRUE, off |-> 0, ep |-> 1] ELSE NoC]
            /\ effects = [o \in Offsets |-> 0]

(* ---------------------------------------------------------------------- *)
(* Time and the agent's faults                                             *)
(* ---------------------------------------------------------------------- *)
DueRollback == ip \in {"stopl", "wait_attach"} /\ ~Closed /\ now >= WMax
DueForward  == ip \in {"rb1", "rb3", "rb4"} /\ ~Closed /\ now >= WMax + RB
DueFail     == ip \in {"rf1", "rf2"} /\ ~Closed /\ now >= WMax + RB + RF
Urgent ==
    \/ agent = "up" /\ (DueRollback \/ DueForward \/ DueFail)
    \/ agent = "hung" /\ MUT /= "no_watchdog" /\ now - tAt >= W
    \/ agent = "down" /\ now - tAt >= G

Tick ==
    /\ InstallOn /\ now < ClockMax /\ ~Urgent
    /\ (WindowOpen \/ agent /= "up")
    /\ now' = now + 1
    /\ UNCHANGED <<S, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

AgentFault ==
    /\ InstallOn /\ agent = "up" /\ WindowOpen /\ nAg < MaxAgentCrash
    /\ \/ agent' = "hung"     \* a hang: only the watchdog ends it
       \/ agent' = "down"     \* a crash: systemd restarts it after G
    /\ tAt' = now /\ nAg' = nAg + 1
    /\ UNCHANGED <<S, now, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

Watchdog ==
    /\ MUT /= "no_watchdog" /\ agent = "hung" /\ now - tAt >= W
    /\ agent' = "down" /\ tAt' = now
    /\ UNCHANGED <<S, now, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

AgentRestart ==
    /\ agent = "down" /\ now - tAt >= G
    /\ agent' = "up"
    /\ UNCHANGED <<S, now, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

(* ---------------------------------------------------------------------- *)
(* Runtime: boot, bus (S2 to S4), crash                                    *)
(* ---------------------------------------------------------------------- *)
Boot(r) ==
    /\ S[r].proc = "up" /\ ~S[r].ready /\ ~S[r].stop /\ ~S[r].rev
    /\ S' = [S EXCEPT ![r].ready = TRUE]
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* S3: the effect runs, then the commit is its own synchronous step.
Process(r) ==
    /\ BusOn /\ S[r].proc = "up" /\ S[r].role = "active"
    /\ ~S[r].infl /\ S[r].pos < LogLen
    /\ effects' = [effects EXCEPT ![S[r].pos] = @ + 1]
    /\ S' = [S EXCEPT ![r].infl = TRUE]
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, x2, faulted, nCrash, nRev, legEp>>

CommitDone(r) ==
    /\ BusOn /\ Up(r) /\ S[r].infl
    /\ commit' = [commit EXCEPT ![Grp(r)] =
                    [has |-> TRUE, off |-> S[r].pos + 1, ep |-> Stamp(r)]]
    /\ S' = [S EXCEPT ![r].pos = @ + 1, ![r].infl = FALSE]
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, effects, x2, faulted, nCrash, nRev, legEp>>

\* Mutant autocommit: the commit follows the fetch position, past a record
\* that has not been processed (the revision-3 gap S3 closes).
FetchAhead(r) ==
    /\ MUT = "autocommit" /\ BusOn /\ S[r].proc = "up" /\ S[r].role = "active"
    /\ ~S[r].infl /\ S[r].pos < LogLen /\ commit[Grp(r)].off <= S[r].pos
    /\ commit' = [commit EXCEPT ![Grp(r)] =
                    [has |-> TRUE, off |-> S[r].pos + 1, ep |-> Stamp(r)]]
    /\ UNCHANGED <<S, now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, effects, x2, faulted, nCrash, nRev, legEp>>

\* S3a and S4: the seed over the source's snapshot and the other groups.
Seed(r) ==
    LET g == Grp(r)
        cs == [h \in Groups |-> IF h = rec.srcg THEN rec.snap ELSE commit[h]]
        hs == {h \in Groups : cs[h].has}
    IN IF hs = {} THEN (IF MUT = "first_sight_latest" THEN LogLen ELSE 0)
       ELSE IF MUT = "seed_own_group" /\ cs[g].has THEN cs[g].off
       ELSE cs[BestOf(cs, hs)].off

Activate(r) ==
    /\ S[r].proc = "up" /\ S[r].role = "passive" /\ rec.inst = r
    /\ S[r].ready /\ ~S[r].rev /\ ~S[r].stop
    /\ LET sd == IF BusOn THEN Seed(r) ELSE 0 IN
       /\ S' = [S EXCEPT ![r].role = "active", ![r].pos = sd, ![r].ep = rec.ep]
       /\ commit' = IF BusOn
                    THEN [commit EXCEPT ![Grp(r)] = [has |-> TRUE, off |-> sd, ep |-> rec.ep]]
                    ELSE commit
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, effects, x2, faulted, nCrash, nRev, legEp>>

\* A crash or a kill after the stop grace. A crash of a runtime that accepts on
\* L is the fault R1; a record whose commit had not returned may run again (X2).
Crash(r) ==
    /\ nCrash < MaxCrash /\ r /= 0 /\ Up(r)
    /\ S' = [S EXCEPT ![r].proc = IF @ = "up" THEN "crashed" ELSE "exited",
                      ![r].role = "passive", ![r].ready = FALSE, ![r].lst = "detached",
                      ![r].conf = FALSE, ![r].infl = FALSE, ![r].rev = FALSE,
                      ![r].stop = FALSE]
    /\ x2' = IF S[r].infl THEN [x2 EXCEPT ![S[r].pos] = @ + 1] ELSE x2
    /\ faulted' = (faulted \/ S[r].lst \in {"attaching", "attached", "holding", "draining"})
    /\ grantTo' = IF grantTo = r THEN 0 ELSE grantTo
    /\ nCrash' = nCrash + 1
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, Lopen,
                   rec, commit, effects, nRev, legEp>>

\* Docker restarts a crashed container; S1: it always boots passive.
Restart(r) ==
    /\ S[r].proc = "crashed"
    /\ S' = [S EXCEPT ![r] = NewRt]
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* The graceful exit: after A3, after the teardown, with every commit returned.
Exit(r) ==
    /\ S[r].proc = "stopping" /\ ~S[r].infl /\ S[r].lst = "detached" /\ ~S[r].ready
    /\ S' = [S EXCEPT ![r].proc = "exited", ![r].role = "passive"]
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

(* ---------------------------------------------------------------------- *)
(* Runtime: the listener (A1 to A6)                                        *)
(* ---------------------------------------------------------------------- *)
\* A1/A5 step one: both latches and readiness checked, fd fetched, session
\* opened as provisional, the L site started.
Attach1(r) ==
    /\ HttpOn /\ r /= 0 /\ S[r].proc = "up" /\ S[r].ready /\ ~S[r].stop /\ ~S[r].rev
    /\ S[r].lst = "detached" /\ Lopen
    /\ S' = [S EXCEPT ![r].lst = "attaching"]
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* Step two: confirmed, or back to detached when a latch or readiness changed.
Attach2(r) ==
    /\ HttpOn /\ S[r].lst = "attaching"
    /\ IF ~S[r].stop /\ ~S[r].rev /\ S[r].ready
       THEN S' = [S EXCEPT ![r].lst = "attached", ![r].conf = TRUE]
       ELSE S' = [S EXCEPT ![r].lst = "detached", ![r].conf = FALSE]
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* The install runs without the HTTP piece: attachment is one step.
AttachAtomic(r) ==
    /\ ~HttpOn /\ InstallOn /\ r = 1 /\ S[r].proc = "up" /\ S[r].ready /\ ~S[r].stop
    /\ S[r].lst = "detached" /\ Lopen
    /\ S' = [S EXCEPT ![r].lst = "attached", ![r].conf = TRUE]
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* The legacy publishes through Docker's own mapping: it serves once ready.
LegacyServe(r) ==
    /\ InstallOn /\ r = 0 /\ S[r].proc = "up" /\ S[r].ready /\ S[r].lst = "detached"
    /\ S' = [S EXCEPT ![r].lst = "attached", ![r].conf = TRUE]
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* A6: B's answer. Granted needs another confirmed session and no standing grant.
Granted(r) == MUT = "no_grants" \/ (OtherConf(r) /\ grantTo = 0)
Alone(r) == ~handover /\ ~OtherConf(r) /\ grantTo = 0

\* A3 (1): stop accepting by closing this process's duplicate. Never shutdown()
\* the socket: that is the mutant.
DoDetach(r, g) ==
    /\ S' = [S EXCEPT ![r].lst = "draining"]
    /\ grantTo' = IF g THEN r ELSE grantTo
    /\ Lopen' = IF MUT = "shutdown_socket" THEN FALSE ELSE Lopen

StopDetach(r) ==
    /\ HttpOn /\ S[r].stop /\ ConfAtt(r) /\ (Granted(r) \/ Alone(r))
    /\ DoDetach(r, Granted(r))
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* The successor never comes (R1): the stopping runtime detaches after the bound.
StopDetachTimeout(r) ==
    /\ HttpOn /\ S[r].stop /\ ConfAtt(r) /\ ~(Granted(r) \/ Alone(r)) /\ ~faulted
    /\ DoDetach(r, FALSE)
    /\ faulted' = TRUE
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover,
                   rec, commit, effects, x2, nCrash, nRev, legEp>>

StopAbortAttach(r) ==
    /\ HttpOn /\ S[r].stop /\ S[r].lst = "attaching"
    /\ S' = [S EXCEPT ![r].lst = "detached", ![r].conf = FALSE]
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* A3 (2) to (4): the accepted connections are answered, then the session closes.
DrainDone(r) ==
    /\ HttpOn /\ S[r].lst = "draining"
    /\ S' = [S EXCEPT ![r].lst = "detached", ![r].conf = FALSE]
    /\ grantTo' = IF grantTo = r THEN 0 ELSE grantTo
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* A5: the readiness teardown starts only after A3 has finished.
Teardown(r) ==
    /\ HttpOn /\ S[r].stop /\ S[r].lst = "detached" /\ S[r].ready
    /\ S' = [S EXCEPT ![r].ready = FALSE]
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* A2: on_partitions_revoked sets the latch first.
Revoke(r) ==
    /\ HttpOn /\ nRev < MaxRevoke /\ S[r].proc = "up" /\ S[r].ready
    /\ ~S[r].rev /\ ~S[r].stop /\ S[r].lst \in {"detached", "attached"}
    /\ S' = [S EXCEPT ![r].rev = TRUE]
    /\ nRev' = nRev + 1
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, legEp>>

\* A2/A6: ask B before the assignment is cleared; granted detaches, otherwise hold.
RevAsk(r) ==
    /\ HttpOn /\ S[r].rev /\ S[r].ready /\ ConfAtt(r) /\ S[r].lst = "attached"
    /\ IF Granted(r)
       THEN /\ DoDetach(r, TRUE)
       ELSE /\ S' = [S EXCEPT ![r].lst = "holding", ![r].ready = FALSE]
            /\ UNCHANGED <<grantTo, Lopen>>
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* A runtime that never attached, or has detached, loses readiness with the assignment.
RevClear(r) ==
    /\ HttpOn /\ S[r].rev /\ S[r].ready /\ S[r].lst = "detached"
    /\ S' = [S EXCEPT ![r].ready = FALSE]
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* on_partitions_assigned finished: readiness is back, held polls answer 200.
Reassigned(r) ==
    /\ HttpOn /\ S[r].rev /\ ~S[r].ready /\ S[r].proc = "up"
    /\ S' = [S EXCEPT ![r].rev = FALSE, ![r].ready = TRUE,
                      ![r].lst = IF @ = "holding" THEN "attached" ELSE @]
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* R1: a revocation whose reassignment outlasts the hold bound.
HoldTimeout(r) ==
    /\ HttpOn /\ S[r].lst = "holding" /\ ~faulted
    /\ faulted' = TRUE
    /\ UNCHANGED <<S, now, agent, tAt, nAg, ip, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, nCrash, nRev, legEp>>

(* ---------------------------------------------------------------------- *)
(* The redeploy program (section 6)                                        *)
(* ---------------------------------------------------------------------- *)
Nx == cur + 1

RCreate ==
    /\ ~InstallOn /\ pc = "idle" /\ rd < MaxRedeploy
    /\ S[cur].proc = "up" /\ S[cur].role = "active" /\ S[cur].ready
    /\ S' = [S EXCEPT ![Nx] = NewRt]
    /\ pc' = "wait_ready" /\ rd' = rd + 1
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

RReady ==
    /\ ~InstallOn /\ pc = "wait_ready" /\ S[Nx].proc = "up" /\ S[Nx].ready
    /\ pc' = "wait_attach" /\ handover' = TRUE
    /\ UNCHANGED <<S, now, agent, tAt, nAg, ip, rd, cur, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* Step 3: wait for the replacement's confirmed session before retiring the old.
RAttach ==
    /\ ~InstallOn /\ pc = "wait_attach"
    /\ (~HttpOn \/ MUT = "no_attach_gate" \/ (ConfAtt(Nx) /\ S[Nx].lst = "attached"))
    /\ pc' = "retire"
    /\ UNCHANGED <<S, now, agent, tAt, nAg, ip, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* Step 4: docker stop. Without exit verification the record is written at once.
RStop ==
    /\ ~InstallOn /\ pc = "retire" /\ S[cur].proc = "up"
    /\ S' = [S EXCEPT ![cur] = IF HttpOn
                               THEN [@ EXCEPT !.proc = "stopping", !.stop = TRUE,
                                      !.ready = IF MUT = "teardown_first" THEN FALSE ELSE @]
                               ELSE StopRec(@)]
    /\ pc' = IF MUT = "no_verify_exit" THEN "record" ELSE "wait_exit"
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

RExit ==
    /\ ~InstallOn /\ pc = "wait_exit" /\ S[cur].proc = "exited"
    /\ pc' = "remove"
    /\ UNCHANGED <<S, now, agent, tAt, nAg, ip, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

RRemove ==
    /\ ~InstallOn /\ pc = "remove"
    /\ S' = [S EXCEPT ![cur].proc = "removed"]
    /\ pc' = "record"
    /\ UNCHANGED <<now, agent, tAt, nAg, ip, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* Step 5: epoch plus one, the new owner, the source's snapshot.
RRecord ==
    /\ ~InstallOn /\ pc = "record"
    /\ rec' = [ep |-> rec.ep + 1, inst |-> Nx, src |-> cur, srcg |-> Grp(cur),
               snap |-> commit[Grp(cur)]]
    /\ pc' = "wait_active"
    /\ UNCHANGED <<S, now, agent, tAt, nAg, ip, rd, cur, handover, grantTo, Lopen,
                   commit, effects, x2, faulted, nCrash, nRev, legEp>>

RActive ==
    /\ ~InstallOn /\ pc = "wait_active" /\ S[Nx].role = "active"
    /\ pc' = "idle" /\ cur' = Nx /\ handover' = FALSE
    /\ UNCHANGED <<S, now, agent, tAt, nAg, ip, rd, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

(* ---------------------------------------------------------------------- *)
(* The first install (section 7). The clock `now` starts at the legacy stop.*)
(* ---------------------------------------------------------------------- *)
AU == InstallOn /\ agent = "up"

IStart ==
    /\ AU /\ ip = "pre" /\ S[0].proc = "up" /\ S[1].proc = "up" /\ S[1].ready
    /\ S' = [S EXCEPT ![0] = StopRec(@)]
    /\ ip' = "stopl"
    /\ UNCHANGED <<now, agent, tAt, nAg, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

IBind ==
    /\ AU /\ ip = "stopl" /\ S[0].proc = "exited"
    /\ Lopen' = TRUE /\ ip' = "wait_attach"
    /\ UNCHANGED <<S, now, agent, tAt, nAg, pc, rd, cur, handover, grantTo,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* After N's attachment: snapshot the legacy's groups, write the next epoch.
ISnap ==
    /\ AU /\ ip = "wait_attach" /\ ConfAtt(1)
    /\ rec' = [ep |-> rec.ep + 1, inst |-> 1, src |-> 0, srcg |-> 0, snap |-> commit[0]]
    /\ ip' = "post"
    /\ UNCHANGED <<S, now, agent, tAt, nAg, pc, rd, cur, handover, grantTo, Lopen,
                   commit, effects, x2, faulted, nCrash, nRev, legEp>>

ISettle ==
    /\ AU /\ ip = "post" /\ S[1].role = "active"
    /\ ip' = "done"
    /\ UNCHANGED <<S, now, agent, tAt, nAg, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* Rollback at WMax: stop N and verify, drop L, copy, record, start the legacy.
IRollback ==
    /\ AU /\ ip \in {"stopl", "wait_attach"} /\ ~Closed
    /\ now >= WMax /\ now < WMax + RB
    /\ S' = [S EXCEPT ![1] = IF @.proc = "up" THEN StopRec(@) ELSE @]
    /\ ip' = "rb1"
    /\ UNCHANGED <<now, agent, tAt, nAg, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* The copy onto the legacy's group is the S4 seed over N's group and the
\* legacy's, the legacy's unstamped commit counting as legEp; only then the record.
IRb2 ==
    /\ AU /\ ip = "rb1" /\ S[1].proc = "exited" /\ S[0].proc = "exited"
    /\ LET hs == {h \in {0, 1} : commit[h].has}
           best == commit[BestOf(commit, hs)]
       IN commit' = [commit EXCEPT ![0] =
                       IF hs = {} THEN NoC
                       ELSE [has |-> TRUE, off |-> best.off, ep |-> -1]]
    /\ Lopen' = FALSE
    /\ rec' = [ep |-> rec.ep + 1, inst |-> 0, src |-> 1, srcg |-> 1, snap |-> NoC]
    /\ legEp' = rec.ep + 1
    /\ ip' = "rb3"
    /\ UNCHANGED <<S, now, agent, tAt, nAg, pc, rd, cur, handover, grantTo,
                   effects, x2, faulted, nCrash, nRev>>

IRb3 ==
    /\ AU /\ ip = "rb3" /\ S[0].proc = "exited"
    /\ S' = [S EXCEPT ![0] = [NewRt EXCEPT !.role = "active",
                              !.pos = IF commit[0].has THEN commit[0].off ELSE 0]]
    /\ ip' = "rb4"
    /\ UNCHANGED <<now, agent, tAt, nAg, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

IRbDone ==
    /\ AU /\ ip = "rb4" /\ S[0].lst = "attached" /\ S[0].ready
    /\ ip' = "rolledback"
    /\ UNCHANGED <<S, now, agent, tAt, nAg, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* Roll forward when the rollback missed a deadline: stop whichever runs.
IRollForward ==
    /\ AU /\ ip \in {"stopl", "wait_attach", "rb1", "rb3", "rb4"} /\ ~Closed
    /\ now >= WMax + RB
    /\ S' = [r \in Runtimes |-> IF S[r].proc = "up" THEN StopRec(S[r]) ELSE S[r]]
    /\ ip' = "rf1"
    /\ UNCHANGED <<now, agent, tAt, nAg, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* Both exits verified; make L available; the record's source is read, never
\* assumed; start N.
IRf1 ==
    /\ AU /\ ip = "rf1" /\ S[0].proc \in {"exited", "removed"} /\ S[1].proc = "exited"
    /\ Lopen' = TRUE
    /\ rec' = IF rec.inst = 1 THEN rec
              ELSE [ep |-> rec.ep + 1, inst |-> 1, src |-> 0, srcg |-> 0, snap |-> commit[0]]
    /\ S' = [S EXCEPT ![1] = NewRt]
    /\ ip' = "rf2"
    /\ UNCHANGED <<now, agent, tAt, nAg, pc, rd, cur, handover, grantTo,
                   commit, effects, x2, faulted, nCrash, nRev, legEp>>

IRf2 ==
    /\ AU /\ ip = "rf2" /\ ConfAtt(1)
    /\ ip' = "post"
    /\ UNCHANGED <<S, now, agent, tAt, nAg, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

\* R3: a second fault ends install_failed; the refusal stands.
IFail ==
    /\ AU /\ ip \in {"rf1", "rf2"} /\ ~Closed /\ now >= WMax + RB + RF
    /\ ip' = "failed"
    /\ UNCHANGED <<S, now, agent, tAt, nAg, pc, rd, cur, handover, grantTo, Lopen,
                   rec, commit, effects, x2, faulted, nCrash, nRev, legEp>>

(* ---------------------------------------------------------------------- *)
RtNext(r) ==
    \/ Boot(r) \/ Process(r) \/ CommitDone(r) \/ FetchAhead(r) \/ Activate(r)
    \/ Crash(r) \/ Restart(r) \/ Exit(r)
    \/ Attach1(r) \/ Attach2(r) \/ AttachAtomic(r) \/ LegacyServe(r)
    \/ StopDetach(r) \/ StopDetachTimeout(r) \/ StopAbortAttach(r)
    \/ DrainDone(r) \/ Teardown(r)
    \/ Revoke(r) \/ RevAsk(r) \/ RevClear(r) \/ Reassigned(r) \/ HoldTimeout(r)

Next ==
    \/ Tick \/ AgentFault \/ Watchdog \/ AgentRestart
    \/ \E r \in Runtimes : RtNext(r)
    \/ RCreate \/ RReady \/ RAttach \/ RStop \/ RExit \/ RRemove \/ RRecord \/ RActive
    \/ IStart \/ IBind \/ ISnap \/ ISettle \/ IRollback \/ IRb2 \/ IRb3 \/ IRbDone
    \/ IRollForward \/ IRf1 \/ IRf2 \/ IFail

Spec == Init /\ [][Next]_vars

(* ---------------------------------------------------------------------- *)
(* Properties                                                              *)
(* ---------------------------------------------------------------------- *)
TypeOK ==
    /\ \A r \in Runtimes : S[r].proc \in {"none", "up", "stopping", "exited", "crashed", "removed"}
    /\ \A r \in Runtimes : S[r].lst \in {"detached", "attaching", "attached", "draining", "holding"}
    /\ now \in 0..ClockMax
    /\ agent \in {"up", "hung", "down"}
    /\ ip \in {"pre", "stopl", "wait_attach", "post", "done", "rb1", "rb3", "rb4",
               "rolledback", "rf1", "rf2", "failed"}
    /\ \A g \in Groups : commit[g].off \in 0..LogLen

\* A connection accepted from L is accepted by a runtime that reads ready (or
\* holds through its own revocation), in a run without a fault.
AcceptedIsReady ==
    HttpOn => (~faulted => \A r \in Acc : S[r].ready \/ S[r].lst = "holding")

\* A poll is never refused (the socket exists) and never left with no acceptor.
NoRefusalWithoutFault ==
    HttpOn => (~faulted => (Lopen /\ Acc /= {}))

\* The standby is paused and a runtime consumes only after S4: one at a time.
AtMostOneConsuming ==
    BusOn => Cardinality({r \in Runtimes : Consuming(r)}) <= 1

\* A commit, and a seek, never move past a record whose effect has not run.
NoSkip ==
    BusOn =>
      /\ \A g \in Groups : commit[g].has =>
            \A o \in Offsets : o < commit[g].off => effects[o] >= 1
      /\ \A r \in Runtimes : S[r].role = "active" =>
            \A o \in Offsets : o < S[r].pos => effects[o] >= 1

\* A record runs again only after a kill or crash in its effect-to-commit gap.
ReplayOnlyX2 ==
    BusOn => \A o \in Offsets : effects[o] <= 1 + x2[o]

\* A window that is not the standing refusal of R3 closes within WBound ticks.
WindowBounded ==
    InstallOn => ((WindowOpen /\ ip /= "failed") => now <= WBound)

(* ---------------------------------------------------------------------- *)
(* Witnesses: each is an invariant that a reachable state violates, so that  *)
(* a passing run is shown not to be vacuous (the path it names completes).   *)
(* ---------------------------------------------------------------------- *)
NotRedeployed == ~(rd = MaxRedeploy /\ pc = "idle")
NotHolding == \A r \in Runtimes : S[r].lst /= "holding"
NotGranted == grantTo = 0
NotReplayed == \A o \in Offsets : effects[o] <= 1
NotInstalled == ip /= "done"
NotRolledBack == ip /= "rolledback"
NotRolledForward == ip /= "rf2"
NotFailed == ip /= "failed"
NotTwoAgentFaults == nAg < 2

=============================================================================
