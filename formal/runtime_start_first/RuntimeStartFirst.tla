------------------------- MODULE RuntimeStartFirst -------------------------
(* OMN-22151 stage 1: the dev runtime redeploy that starts the replacement   *)
(* beside the old runtime, hands the consumers over, and stops the old one   *)
(* only after the ready-aware front routes to the new one. It also models    *)
(* the FIRST install of the front, the one declared window in which          *)
(* /ready on the host ports may read non-200.                                *)
(*                                                                            *)
(* Two runtime instances exist, "o" (the old, serving) and "n" (the          *)
(* replacement). Each consumes through its OWN consumer group (the slot is   *)
(* part of KAFKA_INSTANCE_ID), so exclusivity of a partition is not given by *)
(* the broker's group protocol: it is the standby's pause plus the agent's   *)
(* fence-then-resume order that makes it. An instance is in one state:       *)
(*   none     not created                                                    *)
(*   boot     container started, nothing joined, /ready 503                  *)
(*   standby  booted, joined its own group, consumers PAUSED, /ready 200     *)
(*            with role passive                                              *)
(*   serving  consuming, /ready 200 with role active                         *)
(*   fenced   consumers paused after its last commit, /ready 200, passive    *)
(*   stopping stop signalled, /ready 503, host port still held              *)
(*   dead     removed or crashed                                             *)
(* Processing an event and committing it is one step (the T8 model in        *)
(* formal/consumer_group_handover holds the idempotency-key caveat).         *)
(* A resume seeks the new instance to the committed position of the side    *)
(* that LAST RAN, the T8 cutover rule, here by a seek on the paused member   *)
(* instead of an admin offset write (a group with a live member refuses the  *)
(* latter).                                                                   *)
(*                                                                            *)
(* The front probes each instance and routes to a believed-ready one,        *)
(* preferring an active role. A request that finds its target gone is tried *)
(* on the other believed-ready instance (the retry). The deploy agent drives *)
(*   redeploy: create n, wait standby, fence o, resume n (seek), wait until *)
(*             the front routes to n, stop o. A timeout or a dead n aborts: *)
(*             fence n, unfence o with a seek back, remove n.                *)
(*   install:  the same until n serves; then the window opens (stop the old *)
(*             publisher of the host ports, create the front, wait until   *)
(*             the front answers 200 through n). At the bound the window    *)
(*             rolls back (remove the front, fence n, restart o with a      *)
(*             reverse seek, remove n).                                      *)
(* The agent may crash and is restarted with its job record; time is a tick *)
(* counter, not a clock.                                                     *)
(*                                                                            *)
(* Property (config that must break it)                                       *)
(* AtMostOneServing       mut_no_fence, mut_rollback_no_fence                *)
(* Contiguous             mut_no_seek (resume at the group default, latest)  *)
(* NoDoubleEffect         mut_abort_no_seek, mut_rollback_no_copy            *)
(* GaplessWhileOldHealthy mut_no_retry                                        *)
(* GaplessThroughStop     mut_stop_early                                      *)
(* FailedLeavesOldServing mut_abort_stops_old                                 *)
(* WindowStartsWithNewServed mut_open_early                                   *)
(* WindowBounded          mut_no_bound                                        *)
(* RolledBackIsOld        mut_rollback_keeps_front                            *)
(* window ends in one arrangement mut_no_rollback (TLC reports a deadlock)   *)
(* Witness cfgs wit_*.cfg each violate their invariant: the path completes.  *)
(*                                                                            *)
(* Not modelled: partitions beyond one consumer position (the handover is    *)
(* per group and partition and the argument is per partition); the broker's  *)
(* rebalance inside one slot group; the front or the host crashing; two      *)
(* agents; the effects runtime as a second pool (it is the same protocol    *)
(* on its own pool, and its first install shares the one window); a first   *)
(* install whose old container is already dead; image and overlay rolling   *)
(* back together; HTTP requests that were mid-flight when an instance      *)
(* stopped (the agent waits the settle bound after the switch).             *)
EXTENDS Naturals, Integers, FiniteSets, TLC

CONSTANTS Mode,           \* "install" or "redeploy"
          N,              \* events the producer publishes
          WMax,           \* ticks the first-install window may last before rollback
          G,              \* ticks the agent may stay down
          RB,             \* ticks the rollback may take
          Bound,          \* ticks a replacement may take before the agent aborts
          MaxCrash,       \* runtime crashes
          MaxAgentCrash,  \* agent crashes
          MUT

ASSUME Mode \in {"install", "redeploy"}

Insts == {"o", "n"}
ReadySt == {"standby", "serving", "fenced"}
WinPh == {"window", "rolling_back"}
TermPh == {"done", "rolled_back", "failed"}
PrePh == {"created", "fenced", "promoted", "switched"}
MaxAge == WMax + G + RB + 1

VARIABLES st,        \* instance state
          pos,       \* pos[i]: events 1..pos[i] committed by i's group, -1 = no commit
          pub,       \* events published
          eff,       \* eff[e]: times event e ran (capped at 2)
          lastRan,   \* the instance whose group committed last
          fv,        \* the front's belief: "down", "passive" or "active"
          frontUp,
          installed,
          ph,        \* the agent's phase
          age,       \* ticks in the phase (kept from window into rolling_back)
          dage,      \* ticks the agent has been down
          agentUp,
          crashed,   \* ghost: the instance crashed
          nCrashes, aCrashes,
          nServed    \* ghost: n has served

vars == <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage,
          agentUp, crashed, nCrashes, aCrashes, nServed>>

Ready(i) == st[i] \in ReadySt
PV(i) == IF st[i] = "serving" THEN "active" ELSE IF Ready(i) THEN "passive" ELSE "down"
Target == IF fv["n"] = "active" THEN "n"
          ELSE IF fv["o"] = "active" THEN "o"
          ELSE IF fv["n"] # "down" THEN "n"
          ELSE IF fv["o"] # "down" THEN "o"
          ELSE "none"
FrontAnswer ==
  IF MUT = "no_retry"
    THEN IF Target # "none" /\ Ready(Target) THEN 200 ELSE 503
    ELSE IF \E i \in Insts : fv[i] # "down" /\ Ready(i) THEN 200 ELSE 503
Answer == IF frontUp THEN FrontAnswer ELSE IF Ready("o") THEN 200 ELSE 503

TypeOK ==
  /\ st \in [Insts -> {"none", "boot", "standby", "serving", "fenced", "stopping", "dead"}]
  /\ pos \in [Insts -> -1..N]
  /\ pub \in 0..N
  /\ eff \in [1..N -> 0..2]
  /\ lastRan \in Insts
  /\ fv \in [Insts -> {"down", "passive", "active"}]
  /\ frontUp \in BOOLEAN /\ installed \in BOOLEAN
  /\ ph \in {"idle", "aborting"} \cup PrePh \cup WinPh \cup TermPh \cup {"stopping"}
  /\ age \in 0..MaxAge /\ dage \in 0..G
  /\ agentUp \in BOOLEAN
  /\ crashed \in [Insts -> BOOLEAN]
  /\ nCrashes \in 0..MaxCrash /\ aCrashes \in 0..MaxAgentCrash
  /\ nServed \in BOOLEAN

Init ==
  /\ st = [i \in Insts |-> IF i = "o" THEN "serving" ELSE "none"]
  /\ pos = [i \in Insts |-> IF i = "o" THEN 0 ELSE -1]
  /\ pub = 0
  /\ eff = [e \in 1..N |-> 0]
  /\ lastRan = "o"
  /\ fv = [i \in Insts |-> IF Mode = "redeploy" /\ i = "o" THEN "active" ELSE "down"]
  /\ frontUp = (Mode = "redeploy")
  /\ installed = (Mode = "redeploy")
  /\ ph = "idle" /\ age = 0 /\ dage = 0 /\ agentUp = TRUE
  /\ crashed = [i \in Insts |-> FALSE]
  /\ nCrashes = 0 /\ aCrashes = 0 /\ nServed = FALSE

Goto(p) == ph' = p /\ age' = 0

(* ---- the world ---- *)

Publish ==
  /\ pub < N /\ pub' = pub + 1
  /\ UNCHANGED <<st, pos, eff, lastRan, fv, frontUp, installed, ph, age, dage,
                 agentUp, crashed, nCrashes, aCrashes, nServed>>

Consume(i) ==
  /\ st[i] = "serving" /\ pos[i] < pub
  /\ eff' = [eff EXCEPT ![pos[i] + 1] = IF @ < 2 THEN @ + 1 ELSE 2]
  /\ pos' = [pos EXCEPT ![i] = @ + 1]
  /\ lastRan' = i
  /\ UNCHANGED <<st, pub, fv, frontUp, installed, ph, age, dage, agentUp,
                 crashed, nCrashes, aCrashes, nServed>>

BootN ==
  /\ st["n"] = "boot"
  /\ st' = [st EXCEPT !["n"] = "standby"]
  /\ pos' = [pos EXCEPT !["n"] = IF MUT = "no_seek" THEN pub ELSE -1]
  /\ UNCHANGED <<pub, eff, lastRan, fv, frontUp, installed, ph, age, dage,
                 agentUp, crashed, nCrashes, aCrashes, nServed>>

OldDies ==
  /\ st["o"] = "stopping"
  /\ st' = [st EXCEPT !["o"] = "dead"]
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage,
                 agentUp, crashed, nCrashes, aCrashes, nServed>>

Crash(i) ==
  /\ nCrashes < MaxCrash
  /\ st[i] \in {"boot", "standby", "serving", "fenced"}
  /\ st' = [st EXCEPT ![i] = "dead"]
  /\ crashed' = [crashed EXCEPT ![i] = TRUE]
  /\ nCrashes' = nCrashes + 1
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage,
                 agentUp, aCrashes, nServed>>

Probe(i) ==
  /\ frontUp /\ fv[i] # PV(i)
  /\ fv' = [fv EXCEPT ![i] = PV(i)]
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, frontUp, installed, ph, age, dage,
                 agentUp, crashed, nCrashes, aCrashes, nServed>>

Tick ==
  /\ age < MaxAge
  /\ (agentUp \/ dage < G)
  /\ (MUT = "no_bound" \/ ph \notin WinPh \/ age < WMax + G + RB)
  /\ (MUT = "no_bound" \/ ~(ph = "window" /\ agentUp /\ age >= WMax))
  /\ age' = age + 1
  /\ dage' = IF agentUp THEN dage ELSE dage + 1
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph,
                 agentUp, crashed, nCrashes, aCrashes, nServed>>

AgentCrash ==
  /\ agentUp /\ aCrashes < MaxAgentCrash /\ ph \notin TermPh /\ ph # "idle"
  /\ agentUp' = FALSE /\ dage' = 0 /\ aCrashes' = aCrashes + 1
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age,
                 crashed, nCrashes, nServed>>

AgentRestart ==
  /\ ~agentUp /\ agentUp' = TRUE
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age,
                 dage, crashed, nCrashes, aCrashes, nServed>>

TerminalStutter == ph \in TermPh /\ UNCHANGED vars

(* ---- the deploy agent: handover ---- *)

StartN ==
  /\ agentUp /\ ph = "idle" /\ st["n"] = "none"
  /\ st' = [st EXCEPT !["n"] = IF MUT = "stand_unpaused" THEN "serving" ELSE "boot"]
  /\ Goto("created")
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp,
                 crashed, nCrashes, aCrashes, nServed>>

Fence ==
  /\ agentUp /\ ph = "created" /\ st["n"] = "standby"
  /\ st' = IF st["o"] = "serving" /\ MUT # "no_fence"
             THEN [st EXCEPT !["o"] = "fenced"] ELSE st
  /\ Goto("fenced")
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp,
                 crashed, nCrashes, aCrashes, nServed>>

Resume ==
  /\ agentUp /\ ph = "fenced" /\ st["n"] = "standby"
  /\ (MUT = "no_fence" \/ st["o"] # "serving")
  /\ st' = [st EXCEPT !["n"] = "serving"]
  /\ pos' = [pos EXCEPT !["n"] = IF MUT = "no_seek" THEN @ ELSE pos[lastRan]]
  /\ nServed' = TRUE
  /\ Goto("promoted")
  /\ UNCHANGED <<pub, eff, lastRan, fv, frontUp, installed, dage, agentUp,
                 crashed, nCrashes, aCrashes>>

AbortStart ==
  /\ agentUp /\ ph \in PrePh
  /\ \/ st["n"] = "dead"
     \/ (age >= Bound /\ ph # "switched")
     \/ (age >= Bound /\ ph = "switched" /\ fv["n"] # "active")
  /\ st' = IF st["n"] = "serving" THEN [st EXCEPT !["n"] = "fenced"] ELSE st
  /\ Goto("aborting")
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp,
                 crashed, nCrashes, aCrashes, nServed>>

AbortUnfence ==
  /\ agentUp /\ ph = "aborting" /\ st["o"] = "fenced" /\ st["n"] # "serving"
  /\ st' = [st EXCEPT !["o"] = "serving"]
  /\ pos' = [pos EXCEPT !["o"] = IF MUT = "abort_no_seek" THEN @ ELSE pos[lastRan]]
  /\ UNCHANGED <<pub, eff, lastRan, fv, frontUp, installed, ph, age, dage,
                 agentUp, crashed, nCrashes, aCrashes, nServed>>

AbortFinish ==
  /\ agentUp /\ ph = "aborting" /\ st["o"] \in {"serving", "dead"}
  /\ st' = IF MUT = "abort_stops_old" THEN [st EXCEPT !["o"] = "dead"]
           ELSE IF st["n"] \in {"none", "dead"} THEN st
           ELSE [st EXCEPT !["n"] = "dead"]
  /\ Goto("failed")
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp,
                 crashed, nCrashes, aCrashes, nServed>>

(* ---- redeploy: switch, stop the old ---- *)

Switch ==
  /\ Mode = "redeploy" /\ agentUp /\ ph = "promoted"
  /\ st["n"] = "serving" /\ fv["n"] = "active"
  /\ Goto("switched")
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, dage,
                 agentUp, crashed, nCrashes, aCrashes, nServed>>

StopOld ==
  /\ Mode = "redeploy" /\ agentUp
  /\ \/ (ph = "switched" /\ st["n"] = "serving" /\ fv["n"] = "active")
     \/ (MUT = "stop_early" /\ ph = "promoted" /\ st["n"] = "serving")
  /\ st["o"] \in {"fenced", "dead"}
  /\ st' = IF st["o"] = "fenced" THEN [st EXCEPT !["o"] = "stopping"] ELSE st
  /\ Goto("stopping")
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp,
                 crashed, nCrashes, aCrashes, nServed>>

DoneRedeploy ==
  /\ Mode = "redeploy" /\ agentUp /\ ph = "stopping" /\ st["o"] = "dead"
  /\ Goto("done")
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, dage,
                 agentUp, crashed, nCrashes, aCrashes, nServed>>

(* ---- install: the one declared window ---- *)

OpenWindow ==
  /\ Mode = "install" /\ agentUp
  /\ \/ (ph = "promoted" /\ st["n"] = "serving")
     \/ (MUT = "open_early" /\ ph = "created" /\ st["n"] = "standby")
  /\ st["o"] \in {"serving", "fenced", "dead"}
  /\ st' = IF st["o"] \in {"serving", "fenced"} THEN [st EXCEPT !["o"] = "stopping"] ELSE st
  /\ Goto("window")
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp,
                 crashed, nCrashes, aCrashes, nServed>>

CreateFront ==
  /\ Mode = "install" /\ agentUp /\ ph = "window" /\ st["o"] = "dead" /\ ~frontUp
  /\ frontUp' = TRUE
  /\ fv' = [i \in Insts |-> "down"]
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, installed, ph, age, dage, agentUp,
                 crashed, nCrashes, aCrashes, nServed>>

Complete ==
  /\ Mode = "install" /\ agentUp /\ ph = "window" /\ frontUp
  /\ fv["n"] = "active" /\ Answer = 200
  /\ Goto("done") /\ installed' = TRUE
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, dage, agentUp,
                 crashed, nCrashes, aCrashes, nServed>>

RollbackStart ==
  /\ Mode = "install" /\ agentUp /\ ph = "window" /\ age >= WMax /\ MUT # "no_rollback"
  /\ frontUp' = (MUT = "rollback_keeps_front")
  /\ fv' = [i \in Insts |-> "down"]
  /\ st' = IF st["n"] = "serving" /\ MUT # "rollback_no_fence"
             THEN [st EXCEPT !["n"] = "fenced"] ELSE st
  /\ ph' = "rolling_back"
  /\ UNCHANGED <<pos, pub, eff, lastRan, installed, age, dage, agentUp, crashed,
                 nCrashes, aCrashes, nServed>>

RestartOld ==
  /\ Mode = "install" /\ agentUp /\ ph = "rolling_back" /\ st["o"] = "dead"
  /\ (MUT = "rollback_no_fence" \/ st["n"] # "serving")
  /\ st' = [st EXCEPT !["o"] = "serving"]
  /\ pos' = [pos EXCEPT !["o"] = IF MUT = "rollback_no_copy" THEN @ ELSE pos[lastRan]]
  /\ UNCHANGED <<pub, eff, lastRan, fv, frontUp, installed, ph, age, dage,
                 agentUp, crashed, nCrashes, aCrashes, nServed>>

RolledBack ==
  /\ Mode = "install" /\ agentUp /\ ph = "rolling_back" /\ st["o"] = "serving"
  /\ st' = IF st["n"] \in {"none", "dead"} THEN st ELSE [st EXCEPT !["n"] = "dead"]
  /\ Goto("rolled_back")
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp,
                 crashed, nCrashes, aCrashes, nServed>>

Next ==
  \/ Publish \/ BootN \/ OldDies \/ Tick \/ AgentCrash \/ AgentRestart \/ TerminalStutter
  \/ \E i \in Insts : Consume(i) \/ Crash(i) \/ Probe(i)
  \/ StartN \/ Fence \/ Resume \/ AbortStart \/ AbortUnfence \/ AbortFinish
  \/ Switch \/ StopOld \/ DoneRedeploy
  \/ OpenWindow \/ CreateFront \/ Complete \/ RollbackStart \/ RestartOld \/ RolledBack

Spec == Init /\ [][Next]_vars

(* ---- properties ---- *)

AtMostOneServing == Cardinality({i \in Insts : st[i] = "serving"}) <= 1

Contiguous == \A i \in Insts : st[i] = "serving" => \A e \in 1..pos[i] : eff[e] >= 1

NoDoubleEffect == \A e \in 1..N : eff[e] <= 1

GaplessWhileOldHealthy ==
  (installed /\ frontUp /\ Ready("o")) => FrontAnswer = 200

GaplessThroughStop ==
  (installed /\ frontUp /\ ph \in {"stopping", "done"}
     /\ ~crashed["o"] /\ ~crashed["n"]) => FrontAnswer = 200

FailedLeavesOldServing == ph = "failed" => (st["o"] = "serving" \/ crashed["o"])

WindowStartsWithNewServed == ph \in WinPh => nServed

WindowBounded == ph \in WinPh => age <= WMax + G + RB

RolledBackIsOld ==
  ph = "rolled_back" => (~frontUp /\ (st["o"] = "serving" \/ crashed["o"]))

DoneIsNew == (Mode = "install" /\ ph = "done") => (frontUp /\ installed /\ nServed)

(* ---- witnesses: each is violated, which shows its path completes ---- *)

RedeployDoneWitness == ~(Mode = "redeploy" /\ ph = "done")
RedeployFailedWitness == ~(Mode = "redeploy" /\ ph = "failed")
InstallDoneWitness == ~(Mode = "install" /\ ph = "done")
InstallRolledBackWitness == ~(Mode = "install" /\ ph = "rolled_back")
OverlapWitness == ~(Ready("o") /\ Ready("n") /\ pos["o"] >= 1)
==========================================================================
