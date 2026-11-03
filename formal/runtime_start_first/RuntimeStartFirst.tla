------------------------- MODULE RuntimeStartFirst -------------------------
(* OMN-22151 stage 1: the dev runtime redeploy that starts the replacement   *)
(* beside the old runtime, hands the consumers over, and stops the old one   *)
(* only after the ready-aware front routes to the new one. It also models    *)
(* the FIRST install of the front, the one declared window in which          *)
(* /ready on the host ports may read non-200.                                *)
(*                                                                            *)
(* Two runtime instances exist, "o" (the old) and "n" (the replacement).     *)
(* Each consumes through its OWN consumer group (the slot is part of         *)
(* KAFKA_INSTANCE_ID), so exclusivity of a partition is not given by the    *)
(* broker's group protocol: it is the standby's pause plus the order of      *)
(* the deploy agent's steps. An instance is in one state:                    *)
(*   none     not created                                                    *)
(*   boot     container started, nothing joined, /ready 503                  *)
(*   standby  booted, joined its own group, consumers PAUSED before any     *)
(*            fetch, /ready 200 role passive                                 *)
(*   serving  consuming, /ready 200 role active                              *)
(*   fenced   dispatch stopped, in-flight work drained and committed,       *)
(*            consumers paused, /ready 200 role passive                      *)
(*   stopping stop signalled, /ready 503, host port still held              *)
(*   dead     removed or crashed                                             *)
(* Readiness can also drop without a crash (assignment loss, a failed       *)
(* probe): unready[i]. Processing an event and committing it is one step     *)
(* here; a crash between the effect and the commit is the at-least-once     *)
(* case every runtime has today and needs the handlers' idempotency key     *)
(* (the T8 model, formal/consumer_group_handover, holds the same caveat).    *)
(* pos[i] is the DURABLE committed position of i's group. A resume commits  *)
(* the position of the side that LAST RAN to the destination group before   *)
(* it unpauses (the T8 cutover rule), by a seek plus a commit on a paused   *)
(* member, or by an offset write onto an empty group while the destination  *)
(* is stopped.                                                                *)
(*                                                                            *)
(* The front probes each instance and routes to a believed-ready one,        *)
(* preferring an active role. A request whose target is gone or answers 503 *)
(* is tried on the other believed-ready instance before anything is sent to *)
(* the client (the retry). The deploy agent drives                           *)
(*   redeploy: create n, wait standby, fence o (still ready), resume n,     *)
(*             wait until the front routes to n, re-verify n, stop o. A      *)
(*             timeout or a dead n BEFORE the stop aborts: fence n, resume o *)
(*             from n's position, remove n. After the stop there is no abort.*)
(*   install:  the legacy old runtime cannot be fenced (it runs the code     *)
(*             that has no control). n is created beside it and waits in     *)
(*             standby. The window opens when the old publishers of the host *)
(*             ports are stopped; after they are dead n resumes from their   *)
(*             committed offsets and the front is created on the freed port. *)
(*             At the bound the window rolls back: remove the front, fence n,*)
(*             write n's offsets onto the stopped legacy groups, THEN start  *)
(*             the legacy containers, then remove n.                         *)
(* Each agent step changes the world and then records the phase; the agent   *)
(* may crash between the two and is restarted with its job record, and it   *)
(* reconciles by reading the world (a command is idempotent per handover).   *)
(* The agent may crash and is restarted; time is a tick counter.            *)
(*                                                                            *)
(* Property (config that must break it)                                       *)
(* AtMostOneServing       mut_no_fence, mut_rollback_no_fence,               *)
(*                        mut_stand_unpaused, mut_install_resume_early       *)
(* Contiguous             mut_no_seek (resume at the group default, latest)  *)
(* NoDoubleEffect         mut_abort_no_seek, mut_rollback_no_copy,           *)
(*                        mut_resume_replay                                  *)
(* GaplessWhileOldHealthy mut_no_retry                                        *)
(* GaplessThroughStop     mut_stop_early                                      *)
(* StopOnlyWhenLit        mut_stop_unverified                                 *)
(* FailedLeavesOldServing mut_abort_stops_old                                 *)
(* WindowStartsWithNewStandby mut_open_early                                  *)
(* WindowBounded          mut_no_bound                                        *)
(* RolledBackIsOld        mut_rollback_keeps_front                            *)
(* window ends in one arrangement mut_no_rollback (TLC reports a deadlock)   *)
(* Witness cfgs wit_*.cfg each violate their invariant: the path completes.  *)
(* WindowResolves and RedeployResolves are temporal properties checked under *)
(* weak fairness of the agent, the world and the ticks (FairSpec).           *)
(*                                                                            *)
(* Not modelled: partitions beyond one consumer position (the handover is    *)
(* per group and partition and the argument is per partition); the broker's  *)
(* rebalance inside one slot group; the front or the host crashing; two      *)
(* agents; a crash between a handler's effect and its commit; the pair of   *)
(* pools: main and effects are one unit here (their old publishers stop     *)
(* together, the front is created when both ports are free, and a rollback  *)
(* restarts both), so a partial failure of one pool's restart is the crash   *)
(* of "o"; a first install whose old container is already dead; image and   *)
(* overlay rolling back together; HTTP requests that were mid-flight when an *)
(* instance stopped (the agent waits the settle bound before the stop); the *)
(* crash of the only remaining runtime after the old was retired (the model *)
(* ends in "lost": an ordinary single-runtime failure, outside the gapless  *)
(* claim).                                                                    *)
EXTENDS Naturals, Integers, FiniteSets, TLC

CONSTANTS Mode,           \* "install" or "redeploy"
          N,              \* events the producer publishes
          WMax,           \* ticks the first-install window may last before rollback
          G,              \* ticks the agent may stay down
          RB,             \* ticks the rollback may take
          Bound,          \* ticks a replacement may take before the agent aborts
          MaxCrash,       \* runtime crashes
          MaxAgentCrash,  \* agent crashes
          MaxDegrade,     \* readiness losses without a crash
          MUT

ASSUME Mode \in {"install", "redeploy"}

Insts == {"o", "n"}
ReadySt == {"standby", "serving", "fenced"}
WinPh == {"window", "rolling_back"}
TermPh == {"done", "rolled_back", "failed", "lost"}
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
          nServed,   \* ghost: n has served
          unready,   \* alive but reading not ready (assignment loss, failed probe)
          degraded,  \* ghost: the instance lost readiness without a crash
          nDeg,
          copied,    \* rollback: n's offsets were written onto the stopped legacy groups
          litAtStop, \* ghost: n was ready and the front routed to it when the old was stopped
          nStandby   \* ghost: n has been in standby

vars == <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage,
          agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg,
          copied, litAtStop, nStandby>>

Ready(i) == st[i] \in ReadySt /\ ~unready[i]
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
  /\ ph \in {"idle", "aborting", "stopping"} \cup PrePh \cup WinPh \cup TermPh
  /\ age \in 0..MaxAge /\ dage \in 0..G
  /\ agentUp \in BOOLEAN
  /\ crashed \in [Insts -> BOOLEAN]
  /\ nCrashes \in 0..MaxCrash /\ aCrashes \in 0..MaxAgentCrash
  /\ nServed \in BOOLEAN
  /\ unready \in [Insts -> BOOLEAN] /\ degraded \in [Insts -> BOOLEAN]
  /\ nDeg \in 0..MaxDegrade
  /\ copied \in BOOLEAN /\ litAtStop \in BOOLEAN /\ nStandby \in BOOLEAN

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
  /\ unready = [i \in Insts |-> FALSE] /\ degraded = [i \in Insts |-> FALSE]
  /\ nDeg = 0 /\ copied = FALSE /\ litAtStop = TRUE /\ nStandby = FALSE

Goto(p) == ph' = p /\ age' = 0

(* ---- the world, the deploy agent ---- *)

Publish ==
  /\ pub < N
  /\ pub' = pub + 1
  /\ UNCHANGED <<st, pos, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

Consume(i) ==
  /\ st[i] = "serving"
  /\ pos[i] < pub
  /\ eff' = [eff EXCEPT ![pos[i] + 1] = IF @ < 2 THEN @ + 1 ELSE 2]
  /\ pos' = [pos EXCEPT ![i] = @ + 1]
  /\ lastRan' = i
  /\ UNCHANGED <<st, pub, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

BootN ==
  /\ st["n"] = "boot"
  /\ st' = [st EXCEPT !["n"] = "standby"]
  /\ pos' = [pos EXCEPT !["n"] = IF MUT = "no_seek" THEN pub ELSE -1]
  /\ nStandby' = TRUE
  /\ UNCHANGED <<pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop>>

OldDies ==
  /\ st["o"] = "stopping"
  /\ st' = [st EXCEPT !["o"] = "dead"]
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

Crash(i) ==
  /\ nCrashes < MaxCrash
  /\ st[i] \in {"boot", "standby", "serving", "fenced"}
  /\ st' = [st EXCEPT ![i] = "dead"]
  /\ crashed' = [crashed EXCEPT ![i] = TRUE]
  /\ nCrashes' = nCrashes + 1
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

Degrade(i) ==
  /\ nDeg < MaxDegrade
  /\ Ready(i)
  /\ unready' = [unready EXCEPT ![i] = TRUE]
  /\ degraded' = [degraded EXCEPT ![i] = TRUE]
  /\ nDeg' = nDeg + 1
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, copied, litAtStop, nStandby>>

Recover(i) ==
  /\ unready[i]
  /\ unready' = [unready EXCEPT ![i] = FALSE]
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, degraded, nDeg, copied, litAtStop, nStandby>>

Probe(i) ==
  /\ frontUp
  /\ fv[i] # PV(i)
  /\ fv' = [fv EXCEPT ![i] = PV(i)]
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

Tick ==
  /\ age < MaxAge
  /\ (agentUp \/ dage < G)
  /\ (MUT = "no_bound" \/ ph \notin WinPh \/ age < WMax + G + RB)
  /\ (MUT = "no_bound" \/ ~(ph = "window" /\ agentUp /\ age >= WMax))
  /\ age' = age + 1
  /\ dage' = IF agentUp THEN dage ELSE dage + 1
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

AgentCrash ==
  /\ agentUp
  /\ aCrashes < MaxAgentCrash
  /\ ph \notin TermPh
  /\ ph # "idle"
  /\ agentUp' = FALSE
  /\ dage' = 0
  /\ aCrashes' = aCrashes + 1
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, crashed, nCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

AgentRestart ==
  /\ ~agentUp
  /\ agentUp' = TRUE
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

TerminalStutter == ph \in TermPh /\ UNCHANGED vars

StartN ==
  /\ agentUp
  /\ ph = "idle"
  /\ st["n"] = "none"
  /\ st' = [st EXCEPT !["n"] = IF MUT = "stand_unpaused" THEN "serving" ELSE "boot"]
  /\ ph' = "created"
  /\ age' = 0
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

Fence ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "created"
  /\ st["n"] = "standby"
  /\ st["o"] = "serving"
  /\ MUT # "no_fence"
  /\ st' = [st EXCEPT !["o"] = "fenced"]
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

RecordFence ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "created"
  /\ st["n"] = "standby"
  /\ (st["o"] \in {"fenced", "dead"} \/ MUT = "no_fence")
  /\ ph' = "fenced"
  /\ age' = 0
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

Resume ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "fenced"
  /\ st["n"] = "standby"
  /\ (MUT = "no_fence" \/ st["o"] # "serving")
  /\ st' = [st EXCEPT !["n"] = "serving"]
  /\ pos' = [pos EXCEPT !["n"] = IF MUT = "no_seek" THEN @ ELSE pos[lastRan]]
  /\ nServed' = TRUE
  /\ UNCHANGED <<pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, unready, degraded, nDeg, copied, litAtStop, nStandby>>

ResumeReplay ==
  /\ MUT = "resume_replay"
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "fenced"
  /\ st["n"] = "serving"
  /\ pos' = [pos EXCEPT !["n"] = pos["o"]]
  /\ UNCHANGED <<st, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

RecordResume ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "fenced"
  /\ st["n"] = "serving"
  /\ ph' = "promoted"
  /\ age' = 0
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

AbortStart ==
  /\ agentUp
  /\ ph \in PrePh
  /\ \/ st["n"] = "dead"
     \/ (age >= Bound /\ ph # "switched")
     \/ (age >= Bound /\ ph = "switched" /\ fv["n"] # "active")
  /\ st' = IF st["n"] = "serving" THEN [st EXCEPT !["n"] = "fenced"] ELSE st
  /\ ph' = "aborting"
  /\ age' = 0
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

AbortUnfence ==
  /\ agentUp
  /\ ph = "aborting"
  /\ st["o"] = "fenced"
  /\ st["n"] # "serving"
  /\ st' = [st EXCEPT !["o"] = "serving"]
  /\ pos' = [pos EXCEPT !["o"] = IF MUT = "abort_no_seek" THEN @ ELSE pos[lastRan]]
  /\ UNCHANGED <<pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

AbortFinish ==
  /\ agentUp
  /\ ph = "aborting"
  /\ st["o"] \in {"serving", "dead"}
  /\ st' = IF MUT = "abort_stops_old" THEN [st EXCEPT !["o"] = "dead"]
           ELSE IF st["n"] \in {"none", "dead"} THEN st
           ELSE [st EXCEPT !["n"] = "dead"]
  /\ ph' = "failed"
  /\ age' = 0
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

Switch ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "promoted"
  /\ st["n"] = "serving"
  /\ fv["n"] = "active"
  /\ ph' = "switched"
  /\ age' = 0
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

StopOld ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ \/ (ph = "switched" /\ fv["n"] = "active" /\ (MUT = "stop_unverified" \/ Ready("n")))
     \/ (MUT = "stop_early" /\ ph = "promoted" /\ st["n"] = "serving")
  /\ st["o"] \in {"fenced", "dead"}
  /\ st' = IF st["o"] = "fenced" THEN [st EXCEPT !["o"] = "stopping"] ELSE st
  /\ litAtStop' = Ready("n")
  /\ ph' = "stopping"
  /\ age' = 0
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, nStandby>>

DoneRedeploy ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "stopping"
  /\ st["o"] = "dead"
  /\ Ready("n")
  /\ ph' = "done"
  /\ age' = 0
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

LostAfterStop ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "stopping"
  /\ st["o"] = "dead"
  /\ ~Ready("n")
  /\ ph' = "lost"
  /\ age' = 0
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

OpenWindow ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph = "created"
  /\ (st["n"] = "standby" \/ MUT = "open_early")
  /\ st["o"] \in {"serving", "dead"}
  /\ st' = IF st["o"] = "serving" THEN [st EXCEPT !["o"] = "stopping"] ELSE st
  /\ ph' = "window"
  /\ age' = 0
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

ResumeInstall ==
  /\ Mode = "install"
  /\ agentUp
  /\ \/ (ph = "window" /\ st["n"] = "standby" /\ st["o"] = "dead")
     \/ (MUT = "install_resume_early" /\ ph = "created" /\ st["n"] = "standby" /\ st["o"] = "serving")
  /\ st' = [st EXCEPT !["n"] = "serving"]
  /\ pos' = [pos EXCEPT !["n"] = IF MUT = "no_seek" THEN @ ELSE pos[lastRan]]
  /\ nServed' = TRUE
  /\ UNCHANGED <<pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, unready, degraded, nDeg, copied, litAtStop, nStandby>>

CreateFront ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph = "window"
  /\ st["o"] = "dead"
  /\ ~frontUp
  /\ frontUp' = TRUE
  /\ fv' = [i \in Insts |-> "down"]
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

Complete ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph = "window"
  /\ frontUp
  /\ fv["n"] = "active"
  /\ Answer = 200
  /\ installed' = TRUE
  /\ ph' = "done"
  /\ age' = 0
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

RollbackStart ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph = "window"
  /\ age >= WMax
  /\ MUT # "no_rollback"
  /\ frontUp' = (MUT = "rollback_keeps_front")
  /\ fv' = [i \in Insts |-> "down"]
  /\ st' = IF st["n"] = "serving" /\ MUT # "rollback_no_fence" THEN [st EXCEPT !["n"] = "fenced"] ELSE st
  /\ ph' = "rolling_back"
  /\ UNCHANGED <<pos, pub, eff, lastRan, installed, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

CopyBack ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph = "rolling_back"
  /\ st["o"] = "dead"
  /\ ~copied
  /\ (MUT = "rollback_no_fence" \/ st["n"] # "serving")
  /\ MUT # "rollback_no_copy"
  /\ pos' = [pos EXCEPT !["o"] = pos[lastRan]]
  /\ copied' = TRUE
  /\ UNCHANGED <<st, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, litAtStop, nStandby>>

RestartOld ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph = "rolling_back"
  /\ st["o"] = "dead"
  /\ (copied \/ MUT = "rollback_no_copy")
  /\ (MUT = "rollback_no_fence" \/ st["n"] # "serving")
  /\ st' = [st EXCEPT !["o"] = "serving"]
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

RolledBack ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph = "rolling_back"
  /\ st["o"] = "serving"
  /\ st' = IF st["n"] \in {"none", "dead"} THEN st ELSE [st EXCEPT !["n"] = "dead"]
  /\ ph' = "rolled_back"
  /\ age' = 0
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby>>

Next ==
  \/ Publish \/ BootN \/ OldDies \/ Tick \/ AgentCrash \/ AgentRestart \/ TerminalStutter
  \/ \E i \in Insts : Consume(i) \/ Crash(i) \/ Probe(i) \/ Degrade(i) \/ Recover(i)
  \/ StartN \/ Fence \/ RecordFence \/ Resume \/ ResumeReplay \/ RecordResume
  \/ AbortStart \/ AbortUnfence \/ AbortFinish \/ Switch \/ StopOld \/ DoneRedeploy \/ LostAfterStop
  \/ OpenWindow \/ ResumeInstall \/ CreateFront \/ Complete \/ RollbackStart \/ CopyBack
  \/ RestartOld \/ RolledBack

AgentStep ==
  \/ StartN \/ Fence \/ RecordFence \/ Resume \/ RecordResume \/ AbortStart \/ AbortUnfence
  \/ AbortFinish \/ Switch \/ StopOld \/ DoneRedeploy \/ LostAfterStop \/ OpenWindow
  \/ ResumeInstall \/ CreateFront \/ Complete \/ RollbackStart \/ CopyBack \/ RestartOld
  \/ RolledBack

Spec == Init /\ [][Next]_vars

Fair == /\ WF_vars(AgentStep) /\ WF_vars(AgentRestart) /\ WF_vars(OldDies) /\ WF_vars(BootN)
        /\ WF_vars(Tick) /\ (\A i \in Insts : WF_vars(Probe(i)) /\ WF_vars(Recover(i)))

FairSpec == Spec /\ Fair

(* ---- properties ---- *)

AtMostOneServing == Cardinality({i \in Insts : st[i] = "serving"}) <= 1

Contiguous == \A i \in Insts : st[i] = "serving" => \A e \in 1..pos[i] : eff[e] >= 1

NoDoubleEffect == \A e \in 1..N : eff[e] <= 1

GaplessWhileOldHealthy ==
  (installed /\ frontUp /\ st["o"] \in ReadySt /\ ~degraded["o"]) => FrontAnswer = 200

GaplessThroughStop ==
  (installed /\ frontUp /\ ph \in {"stopping", "done"}
     /\ ~crashed["o"] /\ ~crashed["n"] /\ ~degraded["o"] /\ ~degraded["n"]) => FrontAnswer = 200

StopOnlyWhenLit == ph \in {"stopping", "done", "lost"} => litAtStop

FailedLeavesOldServing == ph = "failed" => (st["o"] = "serving" \/ crashed["o"])

WindowStartsWithNewStandby == ph \in WinPh => nStandby

WindowBounded == ph \in WinPh => age <= WMax + G + RB

RolledBackIsOld ==
  ph = "rolled_back" => (~frontUp /\ (st["o"] = "serving" \/ crashed["o"]))

DoneIsNew == (Mode = "install" /\ ph = "done") => (frontUp /\ installed /\ nServed)

WindowResolves == (ph = "window") ~> (ph \in {"done", "rolled_back"})

Resolves == (ph # "idle") ~> (ph \in TermPh)

(* ---- witnesses: each is violated, which shows its path completes ---- *)

RedeployDoneWitness == ~(Mode = "redeploy" /\ ph = "done")
RedeployFailedWitness == ~(Mode = "redeploy" /\ ph = "failed")
InstallDoneWitness == ~(Mode = "install" /\ ph = "done")
InstallRolledBackWitness == ~(Mode = "install" /\ ph = "rolled_back")
OverlapWitness == ~(Ready("o") /\ Ready("n") /\ pos["o"] >= 1)
==========================================================================
