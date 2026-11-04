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
(*   fenced   dispatch gate closed, in-flight work drained and committed,    *)
(*            consumers paused (assignments and consume loops kept),         *)
(*            /ready 200 role passive                                        *)
(*   stopping stop signalled, /ready 503, host port still held; in-flight    *)
(*            work is committed before the process exits                     *)
(*   dead     removed or crashed                                             *)
(* Readiness can also drop without a crash (assignment loss, a failed       *)
(* probe): unready[i]. A handler EFFECT and its offset COMMIT are two        *)
(* steps (Process, Commit): fl[i] is an effect run and not yet committed. A  *)
(* crash between them loses the commit and the next runner repeats the       *)
(* effect: the at-least-once case every runtime has, absorbed by the         *)
(* handlers' idempotency key, which the build tests (T2). The handover       *)
(* itself must repeat nothing: replayed is set when an event that SOME group *)
(* already committed (cm) runs again, and eff counts runs of an event.       *)
(* pos[i] is the DURABLE committed position of i's group.                    *)
(*                                                                            *)
(* A resume is three durable phases kept in a journal jr[i] (the control     *)
(* node's handover journal): none -> committed -> active.                    *)
(*   committed: the seek and COMMIT of the source's committed position to    *)
(*              i's own group is done (a durable write; an assignment loss   *)
(*              cannot restore an older commit); nothing has been dispatched *)
(*   active:    the paused consumers are released, the dispatch gate is     *)
(*              reopened and the emitters are enabled; i consumes.           *)
(* A repeated resume reads the journal: committed or later never seeks      *)
(* again. A fence or an abort bumps the instance's control EPOCH, which      *)
(* here resets jr[i] to none, so a resume command delivered late (StaleGoN)  *)
(* is rejected. The SOURCE of a seek is the instance whose group commits     *)
(* are the most advanced (lastRan, read from the group commits and the       *)
(* journal), not the instance named by the last recorded result event        *)
(* (evSrc): the event is published after the resume and can lag a crash.     *)
(*                                                                            *)
(* The front probes each instance and routes to a believed-ready one,        *)
(* preferring an active role. A request whose target is gone or answers 503 *)
(* is tried on the other believed-ready instance before anything is sent to *)
(* the client (the retry). The design's front also sweeps every configured   *)
(* instance directly before ANY 503, the empty-belief path included; the     *)
(* model credits no sweep and checks the stricter believed-set path.        *)
(* STABILITY ASSUMPTION: a request spans its attempts and finishes before a *)
(* second readiness change, so the model lets a fault (crash, readiness      *)
(* loss) happen only when the front's belief is current for every instance   *)
(* (Settled). The front contract bounds the request duration by the probe    *)
(* interval (T3 validator).                                                  *)
(*                                                                            *)
(* The deploy agent drives                                                    *)
(*   redeploy: create n, wait standby, fence o (still ready), resume n,     *)
(*             wait until the front routes to n, hold, observe n ready,      *)
(*             stop o. A timeout or a dead n BEFORE the stop aborts: fence n,*)
(*             resume o from n's position, remove n. After the stop there is *)
(*             no abort. The guarantee is CONDITIONED: a refusal needs a     *)
(*             named cause (RefusalHasCause), a crash or a readiness loss of *)
(*             an instance, and no healthy believed-ready instance. The old  *)
(*             runtime losing readiness by itself while the replacement boots*)
(*             is such a cause: it is an old-runtime outage, not a gap the  *)
(*             handover creates. The replacement losing readiness between the*)
(*             last observation and the stop is likewise a cause (residual   *)
(*             R4): the observation is fresh within one tick (obs) and the   *)
(*             model does not claim the stop and the observation are atomic. *)
(*   install:  the legacy old runtimes (main and effects pools: o and le)    *)
(*             cannot be fenced. n is created beside them and waits in       *)
(*             standby. The agent records the window start durably (ph       *)
(*             window), THEN stops the legacy main and the legacy effects     *)
(*             (two steps, an agent crash may fall between them). After both  *)
(*             are dead n resumes from the committed offsets and the front is *)
(*             created, only when both host ports are free. At the bound the  *)
(*             window rolls back: remove the front, fence n, write n's        *)
(*             offsets onto the stopped legacy groups, THEN start the legacy  *)
(*             containers. If the legacy pair is not ready by WMax + RB the   *)
(*             install rolls FORWARD: stop and verify exit of BOTH legacy     *)
(*             containers, transfer the last-ran commits, resume n, create    *)
(*             the front, all inside the reserved RF ticks. A second fault    *)
(*             ends it install_failed with the refusal standing (R3).         *)
(* Each agent step changes the world and then records the phase; the agent   *)
(* may crash between the two and is restarted with its job record, and it   *)
(* reconciles by reading the world (a command is idempotent per handover).   *)
(*                                                                            *)
(* TIME is a tick counter. Operations shorter than a tick (a probe round, a  *)
(* graceful stop, an effect/commit pair, the delivery of a resume command in *)
(* the install, an agent step) complete before the                           *)
(* next tick: Tick is disabled while one is pending (WorldUrgent, and an     *)
(* enabled agent step). That is the declared operation bound; each is a      *)
(* policy key the build measures. A deadline that has come is an enabled     *)
(* agent step, so time does not pass beyond it; what the bound property      *)
(* WindowBounded checks is that the agent's deadlines and the reserved       *)
(* roll-forward fit in WMax + G + RB + RF, with G the agent's down time.      *)
(*                                                                            *)
(* Property (config that must break it)                                       *)
(* AtMostOneServing       mut_no_fence, mut_rollback_no_fence,               *)
(*                        mut_stand_unpaused, mut_install_resume_early,      *)
(*                        mut_fwd_no_stop, mut_no_epoch                       *)
(* Contiguous             mut_no_seek (resume at the group default, latest)  *)
(* NoDoubleEffect         mut_abort_no_seek, mut_rollback_no_copy,           *)
(*                        mut_resume_replay, mut_src_from_events             *)
(* RefusalHasCause        mut_no_retry, mut_stop_early, mut_fence_unready    *)
(* StopOnlyWhenLit        mut_stop_unverified                                 *)
(* FailedLeavesOldServing mut_abort_stops_old, mut_abort_after_stop              *)
(* WindowStartsWithNewStandby mut_open_early                                  *)
(* WindowBounded          mut_no_bound                                        *)
(* RolledBackIsOld        mut_rollback_keeps_front                            *)
(* WindowRecordedBeforeStop mut_stop_before_record                            *)
(* FrontOnFreePorts       mut_front_one_port                                  *)
(* InstallFailedNeedsTwoFaults mut_rollback_no_restart                        *)
(* RollForwardHasItsTime  mut_no_fwd_reserve                                  *)
(* window ends in one arrangement mut_no_rollback (TLC reports a deadlock)   *)
(* Witness cfgs wit_*.cfg each violate their invariant: the path completes.  *)
(* WindowResolves and Resolves are temporal properties checked under weak    *)
(* fairness of the agent, the world and the ticks (FairSpec).                *)
(*                                                                            *)
(* Not modelled: partitions beyond one consumer position (the handover is    *)
(* per group and partition and the argument is per partition; a partition    *)
(* acquired after an assignment loss and a topic with no source commit are   *)
(* initialised by the rule in the design, section C2, falsified by T2); the  *)
(* broker's rebalance inside one slot group; the front or the host crashing; *)
(* two agents; the effects pool in a REDEPLOY (the pools hand over one after *)
(* the other, share no group, port or offset, and each is the protocol       *)
(* checked here; the install does model both legacy pools, le); a first      *)
(* install whose old container is already dead; image and overlay rolling   *)
(* back together; HTTP requests that were mid-flight when an instance        *)
(* stopped (the agent waits the settle bound before the stop); the crash of  *)
(* the only remaining runtime after the old was retired (the model ends in   *)
(* "lost": an ordinary single-runtime failure, outside the gapless claim);   *)
(* a redeploy started before the install completed (the agent refuses it,    *)
(* T5).                                                                       *)
EXTENDS Naturals, Integers, FiniteSets, TLC

CONSTANTS Mode,           \* "install" or "redeploy"
          N,              \* events the producer publishes
          WMax,           \* ticks the first-install window may last before rollback
          G,              \* ticks the agent may stay down
          RB,             \* ticks the rollback may take
          RF,             \* ticks reserved for the roll-forward after a failed rollback
          Bound,          \* ticks a replacement may take before the agent aborts
          MaxCrash,       \* runtime crashes
          MaxAgentCrash,  \* agent crashes
          MaxDegrade,     \* readiness losses without a crash
          MUT

ASSUME Mode \in {"install", "redeploy"}

Insts == {"o", "n"}
ReadySt == {"standby", "serving", "fenced"}
WinPh == {"window", "rolling_back", "rolling_fwd"}
TermPh == {"done", "rolled_back", "failed", "lost", "install_failed"}
PrePh == {"created", "fenced", "promoted", "switched"}
Limit == WMax + G + RB + RF
MaxAge == Limit + 1

VARIABLES st,        \* instance state
          pos,       \* pos[i]: events 1..pos[i] committed by i's group, -1 = no commit
          pub,       \* events published
          eff,       \* eff[e]: times event e ran (capped at 2)
          lastRan,   \* the instance whose group committed last
          fv,        \* the front's belief: "down", "passive" or "active"
          frontUp,
          installed,
          ph,        \* the agent's phase
          age,       \* ticks in the phase (kept from window through the rollback)
          dage,      \* ticks the agent has been down
          agentUp,
          crashed,   \* ghost: the instance crashed
          nCrashes, aCrashes,
          nServed,   \* ghost: n has served
          unready,   \* alive but reading not ready (assignment loss, failed probe)
          degraded,  \* ghost: the instance lost readiness without a crash
          nDeg,
          copied,    \* rollback: n's offsets were written onto the stopped legacy groups
          litAtStop, \* ghost: n was ready when the old was stopped
          nStandby,  \* ghost: n has been in standby
          fl,        \* fl[i]: an effect ran and its commit has not landed
          cm,        \* highest event any group committed
          replayed,  \* ghost: an event some group had committed ran again
          jr,        \* jr[i]: the resume journal phase of i
          evSrc,     \* the instance named by the last published resume result
          obs,       \* the agent observed n ready and named by the front, this tick
          obsAtStop, \* ghost: the stop was licensed by such an observation
          le,        \* the legacy effects pool: "up", "stopping" or "dead"
          stopped,   \* ghost: the agent has stopped a legacy publisher
          fwd,       \* ghost: the install rolled forward
          gop,       \* a resume-go command for n is in flight (sent, not delivered)
          rm, re,    \* the rollback has issued the start of the legacy main / effects
          rfs        \* ghost: ticks spent in the roll-forward (capped at RF)

vars == <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

eage == age - dage

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

\* faults happen only when the front's belief is current (the stability assumption)
Settled == ~frontUp \/ \A i \in Insts : fv[i] = PV(i)

Src == IF MUT = "src_from_events" THEN evSrc ELSE lastRan

CauseFault == \E i \in Insts : degraded[i] \/ crashed[i]
HealthyBelieved == \E i \in Insts : fv[i] # "down" /\ Ready(i) /\ ~degraded[i]

WorldUrgent ==
  \/ (st["o"] = "stopping" /\ ~fl["o"] /\ ph # "rolling_fwd")
  \/ (le = "stopping" /\ ph # "rolling_fwd")
  \/ (\E i \in Insts : st[i] \in {"serving", "stopping"} /\ fl[i])
  \/ (frontUp /\ \E i \in Insts : fv[i] # PV(i))
  \/ (Mode = "install" /\ gop)

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
  /\ fl \in [Insts -> BOOLEAN] /\ cm \in 0..N /\ replayed \in BOOLEAN
  /\ jr \in [Insts -> {"none", "committed", "active"}]
  /\ evSrc \in Insts /\ obs \in BOOLEAN /\ obsAtStop \in BOOLEAN
  /\ le \in {"up", "stopping", "dead"}
  /\ stopped \in BOOLEAN /\ fwd \in BOOLEAN
  /\ gop \in BOOLEAN /\ rm \in BOOLEAN /\ re \in BOOLEAN /\ rfs \in 0..RF

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
  /\ fl = [i \in Insts |-> FALSE] /\ cm = 0 /\ replayed = FALSE
  /\ jr = [i \in Insts |-> "none"]
  /\ evSrc = "o" /\ obs = FALSE /\ obsAtStop = TRUE
  /\ le = (IF Mode = "install" THEN "up" ELSE "dead")
  /\ stopped = FALSE /\ fwd = FALSE /\ gop = FALSE /\ rm = FALSE /\ re = FALSE /\ rfs = 0

(* ---- the world ---- *)

Publish ==
  /\ pub < N
  /\ pub' = pub + 1
  /\ UNCHANGED <<st, pos, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

Process(i) ==
  /\ st[i] = "serving"
  /\ ~fl[i]
  /\ pos[i] < pub
  /\ eff' = [eff EXCEPT ![pos[i] + 1] = IF @ < 2 THEN @ + 1 ELSE 2]
  /\ replayed' = (replayed \/ pos[i] + 1 <= cm)
  /\ fl' = [fl EXCEPT ![i] = TRUE]
  /\ UNCHANGED <<st, pos, pub, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, cm, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

Commit(i) ==
  /\ st[i] \in {"serving", "stopping"}
  /\ fl[i]
  /\ pos' = [pos EXCEPT ![i] = @ + 1]
  /\ fl' = [fl EXCEPT ![i] = FALSE]
  /\ lastRan' = i
  /\ cm' = IF pos[i] + 1 > cm THEN pos[i] + 1 ELSE cm
  /\ UNCHANGED <<st, pub, eff, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

BootN ==
  /\ st["n"] = "boot"
  /\ st' = [st EXCEPT !["n"] = "standby"]
  /\ pos' = [pos EXCEPT !["n"] = IF MUT = "no_seek" THEN pub ELSE -1]
  /\ nStandby' = TRUE
  /\ UNCHANGED <<pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

OldDies ==
  /\ st["o"] = "stopping"
  /\ ~fl["o"]
  /\ st' = [st EXCEPT !["o"] = "dead"]
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

EffDies ==
  /\ le = "stopping"
  /\ le' = "dead"
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, stopped, fwd, gop, rm, re, rfs>>

Crash(i) ==
  /\ nCrashes < MaxCrash
  /\ st[i] \in {"boot", "standby", "serving", "fenced"}
  /\ Settled
  /\ st' = [st EXCEPT ![i] = "dead"]
  /\ crashed' = [crashed EXCEPT ![i] = TRUE]
  /\ nCrashes' = nCrashes + 1
  /\ fl' = [fl EXCEPT ![i] = FALSE]
  /\ gop' = IF i = "n" THEN FALSE ELSE gop
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, rm, re, rfs>>

CrashE ==
  /\ nCrashes < MaxCrash
  /\ le = "up"
  /\ Settled
  /\ le' = "dead"
  /\ nCrashes' = nCrashes + 1
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, stopped, fwd, gop, rm, re, rfs>>

Degrade(i) ==
  /\ nDeg < MaxDegrade
  /\ Ready(i)
  /\ Settled
  /\ unready' = [unready EXCEPT ![i] = TRUE]
  /\ degraded' = [degraded EXCEPT ![i] = TRUE]
  /\ nDeg' = nDeg + 1
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

Recover(i) ==
  /\ unready[i]
  /\ unready' = [unready EXCEPT ![i] = FALSE]
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

Probe(i) ==
  /\ frontUp
  /\ fv[i] # PV(i)
  /\ fv' = [fv EXCEPT ![i] = PV(i)]
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

\* the result event of a resume is published after the resume; a crash first loses it
PubResult(i) ==
  /\ jr[i] = "active"
  /\ evSrc # i
  /\ st[i] \in {"serving", "fenced"}
  /\ evSrc' = i
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

\* the resume-go command reaches n (the network is not instantaneous). The control epoch
\* rejects a command sent before a fence or an abort bumped it: the bump clears the
\* command here (mut_no_epoch does not bump, so the late command is delivered).
DeliverGo ==
  /\ gop
  /\ st["n"] \in {"standby", "fenced"}
  /\ jr["n"] = "committed"
  /\ st' = [st EXCEPT !["n"] = "serving"]
  /\ jr' = [jr EXCEPT !["n"] = "active"]
  /\ nServed' = TRUE
  /\ gop' = FALSE
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, evSrc, obs, obsAtStop, le, stopped, fwd, rm, re, rfs>>

ResumeReplay ==
  /\ MUT = "resume_replay"
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "fenced"
  /\ jr["n"] = "active"
  /\ pos' = [pos EXCEPT !["n"] = pos["o"]]
  /\ UNCHANGED <<st, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

AgentCrash ==
  /\ agentUp
  /\ aCrashes < MaxAgentCrash
  /\ ph \notin TermPh
  /\ ph # "idle"
  /\ agentUp' = FALSE
  /\ dage' = 0
  /\ aCrashes' = aCrashes + 1
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, crashed, nCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

AgentRestart ==
  /\ ~agentUp
  /\ agentUp' = TRUE
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

TerminalStutter == ph \in TermPh /\ UNCHANGED vars

(* ---- the deploy agent: redeploy ---- *)

StartN ==
  /\ agentUp
  /\ ph = "idle"
  /\ st["n"] = "none"
  /\ st' = [st EXCEPT !["n"] = IF MUT = "stand_unpaused" THEN "serving" ELSE "boot"]
  /\ ph' = "created"
  /\ age' = 0
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

Fence ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "created"
  /\ st["n"] = "standby"
  /\ st["o"] = "serving"
  /\ ~fl["o"]
  /\ MUT # "no_fence"
  /\ st' = [st EXCEPT !["o"] = "fenced"]
  /\ unready' = IF MUT = "fence_unready" THEN [unready EXCEPT !["o"] = TRUE] ELSE unready
  /\ jr' = [jr EXCEPT !["o"] = "none"]
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

RecordFence ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "created"
  /\ st["n"] = "standby"
  /\ (st["o"] \in {"fenced", "dead"} \/ MUT = "no_fence")
  /\ ph' = "fenced"
  /\ age' = 0
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

ResumeCommitN ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "fenced"
  /\ st["n"] = "standby"
  /\ jr["n"] = "none"
  /\ (MUT = "no_fence" \/ st["o"] # "serving")
  /\ pos' = [pos EXCEPT !["n"] = IF MUT = "no_seek" THEN @ ELSE pos[Src]]
  /\ jr' = [jr EXCEPT !["n"] = "committed"]
  /\ UNCHANGED <<st, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

ResumeGoN ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "fenced"
  /\ st["n"] = "standby"
  /\ jr["n"] = "committed"
  /\ ~gop
  /\ (MUT = "no_fence" \/ st["o"] # "serving")
  /\ gop' = TRUE
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, rm, re, rfs>>

RecordResume ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "fenced"
  /\ st["n"] = "serving"
  /\ jr["n"] = "active"
  /\ ph' = "promoted"
  /\ age' = 0
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

AbortStart ==
  /\ agentUp
  /\ ph \in PrePh
  /\ (MUT = "abort_after_stop" \/ ~(st["o"] = "stopping" \/ (st["o"] = "dead" /\ ~crashed["o"])))
  /\ \/ st["n"] = "dead"
     \/ (age >= Bound /\ ph # "switched")
     \/ (ph = "switched" /\ (fv["n"] # "active" \/ ~Ready("n")))
  /\ (st["n"] # "serving" \/ ~fl["n"])
  /\ st' = IF st["n"] = "serving" THEN [st EXCEPT !["n"] = "fenced"] ELSE st
  /\ jr' = IF MUT = "no_epoch" THEN jr ELSE [jr EXCEPT !["n"] = "none"]
  /\ gop' = IF MUT = "no_epoch" THEN gop ELSE FALSE
  /\ ph' = "aborting"
  /\ age' = 0
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, evSrc, obs, obsAtStop, le, stopped, fwd, rm, re, rfs>>

AbortCommit ==
  /\ agentUp
  /\ ph = "aborting"
  /\ st["o"] = "fenced"
  /\ st["n"] # "serving"
  /\ jr["o"] = "none"
  /\ pos' = [pos EXCEPT !["o"] = IF MUT = "abort_no_seek" THEN @ ELSE pos[Src]]
  /\ jr' = [jr EXCEPT !["o"] = "committed"]
  /\ UNCHANGED <<st, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

AbortGo ==
  /\ agentUp
  /\ ph = "aborting"
  /\ st["o"] = "fenced"
  /\ st["n"] # "serving"
  /\ jr["o"] = "committed"
  /\ st' = [st EXCEPT !["o"] = "serving"]
  /\ jr' = [jr EXCEPT !["o"] = "active"]
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

AbortFinish ==
  /\ agentUp
  /\ ph = "aborting"
  /\ st["o"] \in {"serving", "dead"}
  /\ st' = IF MUT = "abort_stops_old" THEN [st EXCEPT !["o"] = "dead"]
           ELSE IF st["n"] \in {"none", "dead"} THEN st
           ELSE [st EXCEPT !["n"] = "dead"]
  /\ ph' = "failed"
  /\ age' = 0
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

Switch ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "promoted"
  /\ st["n"] = "serving"
  /\ fv["n"] = "active"
  /\ ph' = "switched"
  /\ age' = 0
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

\* the agent reads n's /ready and the front's state: n ready and named, fresh within a tick
Observe ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "switched"
  /\ fv["n"] = "active"
  /\ Ready("n")
  /\ ~obs
  /\ obs' = TRUE
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

StopOld ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ \/ (ph = "switched" /\ fv["n"] = "active" /\ age >= 1 /\ (MUT = "stop_unverified" \/ obs))
     \/ (MUT = "stop_early" /\ ph = "promoted" /\ st["n"] = "serving")
  /\ st["o"] = "fenced"
  /\ st' = [st EXCEPT !["o"] = "stopping"]
  /\ litAtStop' = Ready("n")
  /\ obsAtStop' = obs
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, nStandby, fl, cm, replayed, jr, evSrc, obs, le, stopped, fwd, gop, rm, re, rfs>>

RecordStop ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph \in {"switched", "promoted"}
  /\ st["o"] \in {"stopping", "dead"}
  /\ ph' = "stopping"
  /\ age' = 0
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

DoneRedeploy ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "stopping"
  /\ st["o"] = "dead"
  /\ Ready("n")
  /\ ph' = "done"
  /\ age' = 0
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

LostAfterStop ==
  /\ Mode = "redeploy"
  /\ agentUp
  /\ ph = "stopping"
  /\ st["o"] = "dead"
  /\ ~Ready("n")
  /\ ph' = "lost"
  /\ age' = 0
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

(* ---- the deploy agent: first install ---- *)

\* the window start is recorded durably BEFORE either legacy publisher is stopped
ArmWindow ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph = "created"
  /\ (st["n"] = "standby" \/ MUT = "open_early")
  /\ ph' = "window"
  /\ age' = 0
  /\ dage' = 0
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

StopMain ==
  /\ Mode = "install"
  /\ agentUp
  /\ (ph = "window" \/ (MUT = "stop_before_record" /\ ph = "created"))
  /\ st["o"] = "serving"
  /\ st' = [st EXCEPT !["o"] = "stopping"]
  /\ stopped' = TRUE
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, fwd, gop, rm, re, rfs>>

StopEff ==
  /\ Mode = "install"
  /\ agentUp
  /\ (ph = "window" \/ (MUT = "stop_before_record" /\ ph = "created"))
  /\ le = "up"
  /\ le' = "stopping"
  /\ stopped' = TRUE
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, fwd, gop, rm, re, rfs>>

\* n may resume only when BOTH legacy publishers are verified dead
ResumeOK ==
  \/ (ph \in {"window", "rolling_fwd"} /\ st["o"] = "dead" /\ le = "dead")
  \/ (MUT = "fwd_no_stop" /\ ph = "rolling_fwd")
  \/ (MUT = "install_resume_early" /\ ph = "created" /\ st["n"] = "standby")

ResumeCommitI ==
  /\ Mode = "install"
  /\ agentUp
  /\ ResumeOK
  /\ st["n"] \in {"standby", "fenced"}
  /\ jr["n"] = "none"
  /\ pos' = [pos EXCEPT !["n"] = IF MUT = "no_seek" THEN @ ELSE pos[Src]]
  /\ jr' = [jr EXCEPT !["n"] = "committed"]
  /\ UNCHANGED <<st, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

ResumeGoI ==
  /\ Mode = "install"
  /\ agentUp
  /\ ResumeOK
  /\ st["n"] \in {"standby", "fenced"}
  /\ jr["n"] = "committed"
  /\ ~gop
  /\ gop' = TRUE
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, rm, re, rfs>>

\* both host ports must be free
PortsFree == IF MUT = "front_one_port" THEN st["o"] = "dead" ELSE st["o"] = "dead" /\ le = "dead"

CreateFront ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph \in {"window", "rolling_fwd"}
  /\ ~frontUp
  /\ PortsFree
  /\ frontUp' = TRUE
  /\ fv' = [i \in Insts |-> "down"]
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

Complete ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph \in {"window", "rolling_fwd"}
  /\ frontUp
  /\ fv["n"] = "active"
  /\ Answer = 200
  /\ installed' = TRUE
  /\ ph' = "done"
  /\ age' = 0
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

RollbackStart ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph = "window"
  /\ eage >= WMax
  /\ MUT # "no_rollback"
  /\ (st["n"] # "serving" \/ MUT = "rollback_no_fence" \/ ~fl["n"])
  /\ frontUp' = (MUT = "rollback_keeps_front")
  /\ fv' = [i \in Insts |-> "down"]
  /\ st' = IF st["n"] = "serving" /\ MUT # "rollback_no_fence" THEN [st EXCEPT !["n"] = "fenced"] ELSE st
  /\ jr' = [jr EXCEPT !["n"] = "none"]
  /\ gop' = FALSE
  /\ ph' = "rolling_back"
  /\ UNCHANGED <<pos, pub, eff, lastRan, installed, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, evSrc, obs, obsAtStop, le, stopped, fwd, rm, re, rfs>>

CopyBack ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph = "rolling_back"
  /\ st["o"] = "dead"
  /\ le = "dead"
  /\ ~copied
  /\ (MUT = "rollback_no_fence" \/ st["n"] # "serving")
  /\ MUT # "rollback_no_copy"
  /\ pos' = [pos EXCEPT !["o"] = pos[Src]]
  /\ copied' = TRUE
  /\ UNCHANGED <<st, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

RestartOld ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph = "rolling_back"
  /\ st["o"] = "dead"
  /\ ~rm
  /\ MUT # "rollback_no_restart"
  /\ (copied \/ MUT = "rollback_no_copy")
  /\ (MUT = "rollback_no_fence" \/ st["n"] # "serving")
  /\ st' = [st EXCEPT !["o"] = "serving"]
  /\ rm' = TRUE
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, re, rfs>>

RestartEff ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph = "rolling_back"
  /\ le = "dead"
  /\ ~re
  /\ MUT # "rollback_no_restart"
  /\ (copied \/ MUT = "rollback_no_copy")
  /\ (MUT = "rollback_no_fence" \/ st["n"] # "serving")
  /\ le' = "up"
  /\ re' = TRUE
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, stopped, fwd, gop, rm, rfs>>

RolledBack ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph = "rolling_back"
  /\ Ready("o")
  /\ le = "up"
  /\ st' = IF st["n"] \in {"none", "dead"} THEN st ELSE [st EXCEPT !["n"] = "dead"]
  /\ ph' = "rolled_back"
  /\ age' = 0
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

\* the legacy pair is not ready inside RB: stop and verify exit of BOTH, then go forward
RollForwardStart ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph = "rolling_back"
  /\ eage >= WMax + RB
  /\ ~(Ready("o") /\ le = "up")
  /\ ph' = "rolling_fwd"
  /\ fwd' = TRUE
  /\ st' = IF MUT # "fwd_no_stop" /\ st["o"] = "serving" THEN [st EXCEPT !["o"] = "stopping"] ELSE st
  /\ le' = IF MUT # "fwd_no_stop" /\ le = "up" THEN "stopping" ELSE le
  /\ UNCHANGED <<pos, pub, eff, lastRan, fv, frontUp, installed, age, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, stopped, gop, rm, re, rfs>>

InstallFail ==
  /\ Mode = "install"
  /\ agentUp
  /\ ph = "rolling_fwd"
  /\ \/ eage >= (IF MUT = "no_fwd_reserve" THEN WMax + RB ELSE WMax + RB + RF)
     \/ st["n"] = "dead"
  /\ ph' = "install_failed"
  /\ age' = 0
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, dage, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obs, obsAtStop, le, stopped, fwd, gop, rm, re, rfs>>

AgentStep ==
  \/ StartN \/ Fence \/ RecordFence \/ ResumeCommitN \/ ResumeGoN \/ RecordResume
  \/ AbortStart \/ AbortCommit \/ AbortGo \/ AbortFinish \/ Switch \/ Observe \/ StopOld
  \/ RecordStop \/ DoneRedeploy \/ LostAfterStop
  \/ ArmWindow \/ StopMain \/ StopEff \/ ResumeCommitI \/ ResumeGoI \/ CreateFront \/ Complete
  \/ RollbackStart \/ CopyBack \/ RestartOld \/ RestartEff \/ RolledBack
  \/ RollForwardStart \/ InstallFail

Tick ==
  /\ age < MaxAge
  /\ (agentUp \/ dage < G)
  /\ (MUT = "no_bound" \/ ~WorldUrgent)
  /\ (MUT = "no_bound" \/ ph \notin WinPh \/ age < Limit)
  /\ (MUT = "no_bound" \/ ~(agentUp /\ ENABLED AgentStep))
  /\ age' = age + 1
  /\ dage' = IF agentUp THEN dage ELSE dage + 1
  /\ obs' = FALSE
  /\ rfs' = IF ph = "rolling_fwd" /\ rfs < RF THEN rfs + 1 ELSE rfs
  /\ UNCHANGED <<st, pos, pub, eff, lastRan, fv, frontUp, installed, ph, agentUp, crashed, nCrashes, aCrashes, nServed, unready, degraded, nDeg, copied, litAtStop, nStandby, fl, cm, replayed, jr, evSrc, obsAtStop, le, stopped, fwd, gop, rm, re>>

Next ==
  \/ Publish \/ BootN \/ OldDies \/ EffDies \/ CrashE \/ Tick \/ AgentCrash \/ AgentRestart
  \/ TerminalStutter \/ DeliverGo \/ ResumeReplay
  \/ (\E i \in Insts : Process(i) \/ Commit(i) \/ Crash(i) \/ Probe(i) \/ Degrade(i)
                      \/ Recover(i) \/ PubResult(i))
  \/ AgentStep

Spec == Init /\ [][Next]_vars

Fair == /\ WF_vars(AgentStep) /\ WF_vars(AgentRestart) /\ WF_vars(OldDies) /\ WF_vars(EffDies)
        /\ WF_vars(BootN) /\ WF_vars(Tick) /\ WF_vars(DeliverGo)
        /\ (\A i \in Insts : WF_vars(Probe(i)) /\ WF_vars(Recover(i)) /\ WF_vars(Commit(i)))

FairSpec == Spec /\ Fair

(* ---- properties ---- *)

AtMostOneServing ==
  /\ Cardinality({i \in Insts : st[i] = "serving"}) <= 1
  /\ ~(le = "up" /\ st["n"] = "serving")

Contiguous == \A i \in Insts : st[i] = "serving" => \A e \in 1..pos[i] : eff[e] >= 1

\* the handover repeats no committed event, and repeats any event only after a crash
NoDoubleEffect == ~replayed /\ ((\E e \in 1..N : eff[e] > 1) => nCrashes > 0)

\* after the install, a refusal has a cause: an instance crashed or lost readiness, and no
\* healthy believed-ready instance was left to serve it. The old runtime losing readiness
\* by itself while the replacement boots is such a cause (the guarantee is conditional on
\* the old staying ready until the replacement is ready); the replacement losing readiness
\* after the last observation is another (residual R4).
RefusalHasCause == (installed /\ frontUp /\ Answer = 503) => (CauseFault /\ ~HealthyBelieved)

\* the old is stopped only on a fresh observation of n ready and named by the front
StopOnlyWhenLit == ph \in {"stopping", "done", "lost"} => obsAtStop

FailedLeavesOldServing == ph = "failed" => (st["o"] = "serving" \/ crashed["o"])

WindowStartsWithNewStandby == ph \in WinPh => nStandby

WindowBounded == ph \in WinPh => age <= Limit

RolledBackIsOld ==
  ph = "rolled_back" => (~frontUp /\ (st["o"] = "serving" \/ crashed["o"]))

WindowRecordedBeforeStop == stopped => ph \notin {"idle", "created"}

FrontOnFreePorts == (ph \in {"window", "rolling_fwd"} /\ frontUp) => (st["o"] = "dead" /\ le = "dead")

InstallFailedNeedsTwoFaults == ph = "install_failed" => nCrashes + nDeg >= 2

\* the roll-forward gets its reserved RF ticks before the install gives up (unless n is gone)
RollForwardHasItsTime == (ph = "install_failed" /\ st["n"] # "dead") => rfs >= RF

DoneIsNew == (Mode = "install" /\ ph = "done") => (frontUp /\ installed /\ nServed)

WindowResolves == (ph = "window") ~> (ph \in {"done", "rolled_back", "install_failed"})

Resolves == (ph # "idle") ~> (ph \in TermPh)

(* ---- witnesses: each is violated, which shows its path completes ---- *)

RedeployDoneWitness == ~(Mode = "redeploy" /\ ph = "done")
RedeployFailedWitness == ~(Mode = "redeploy" /\ ph = "failed")
InstallDoneWitness == ~(Mode = "install" /\ ph = "done")
InstallRolledBackWitness == ~(Mode = "install" /\ ph = "rolled_back")
InstallForwardDoneWitness == ~(Mode = "install" /\ ph = "done" /\ fwd)
InstallFailedWitness == ~(Mode = "install" /\ ph = "install_failed")
OverlapWitness == ~(Ready("o") /\ Ready("n") /\ pos["o"] >= 1)
==========================================================================
