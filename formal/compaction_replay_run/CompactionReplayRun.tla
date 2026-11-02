---------------------------- MODULE CompactionReplayRun ----------------------------
(* Compaction replay runs: each case builds one prefix and snapshot; both arms
   restore that snapshot, read through the boundary, and write keyed terminals.
   An independent live switch trips after two distinct incident dedup keys.
   Property (meaning)                         Kind       Mutation that must break it
   UniqueTerminal (one terminal per run key)  invariant  mut_terminal_unique (UniqueTerminalGuard=FALSE)
   NoOnAfterTrip (no on assignment after trip) invariant mut_killswitch (KillGuard=FALSE)
   NoReadAfterBoundary (source time bounded)  invariant  mut_boundary_read (BoundaryGuard=FALSE)
   MatchedRemoval (remove every arm's cell)   invariant  mut_infra_removal (RemovalGuard=FALSE)
   ForkRestore (reruns restore the snapshot)  invariant  mut_fork_restore (RestoreGuard=FALSE)
   KindMatchesCause (distinct terminal kinds) invariant mut_terminal_kind (KindGuard=FALSE)
   OnePrefixOneSnapshot (one build per case)  invariant  mut_prefix_once (OncePrefixGuard=FALSE)
   TypeOK (all variables stay in their types) invariant (checked on Model)
   Terminates (every run key gets a terminal) temporal  mut_no_fairness (SpecNoFair, no WF_vars(Next)) *)
EXTENDS Naturals, FiniteSets
CONSTANTS Cases, Arms, Sessions, Boundary, MaxT, MaxAttempts,
          UniqueTerminalGuard, KillGuard, BoundaryGuard, RemovalGuard,
          RestoreGuard, KindGuard, OncePrefixGuard

RunKeys == Cases \X Arms
Reasons == {"scored", "cap_stop", "infra_fault"}
IncidentKeys == Sessions \X {1, 2}
ASSUME /\ UniqueTerminalGuard \in BOOLEAN /\ KillGuard \in BOOLEAN
       /\ BoundaryGuard \in BOOLEAN /\ RemovalGuard \in BOOLEAN
       /\ RestoreGuard \in BOOLEAN /\ KindGuard \in BOOLEAN
       /\ OncePrefixGuard \in BOOLEAN

VARIABLES prefixState, builds, removed, removalCause, phase, attempt,
          workState, startedFrom, maxRead, termRows, termKind, stopReason,
          analysis, assign, switch, incidents, assignedWhenTripped
replayVars == <<prefixState, builds, removed, removalCause, phase, attempt,
               workState, startedFrom, maxRead, termRows, termKind, stopReason,
               analysis>>
liveVars == <<assign, switch, incidents, assignedWhenTripped>>
vars == <<replayVars, liveVars>>

Init ==
  /\ prefixState = [c \in Cases |-> "unbuilt"]
  /\ builds = [c \in Cases |-> 0]
  /\ removed = {}
  /\ removalCause = [c \in Cases |-> "none"]
  /\ phase = [r \in RunKeys |-> "idle"]
  /\ attempt = [r \in RunKeys |-> 0]
  /\ workState = [r \in RunKeys |-> "none"]
  /\ startedFrom = [r \in RunKeys |-> "none"]
  /\ maxRead = [r \in RunKeys |-> 0]
  /\ termRows = [r \in RunKeys |-> 0]
  /\ termKind = [r \in RunKeys |-> "none"]
  /\ stopReason = [r \in RunKeys |-> "none"]
  /\ analysis = {}
  /\ assign = [s \in Sessions |-> "unassigned"]
  /\ switch = "armed"
  /\ incidents = {}
  /\ assignedWhenTripped = [s \in Sessions |-> FALSE]

\* Build the prefix and snapshot atomically, or remove a prefix-faulted case.
BuildPrefix(c) ==
  /\ prefixState[c] = "unbuilt"
  /\ builds' = [builds EXCEPT ![c] = @ + 1]
  /\ \E outcome \in {"ok", "fault"}:
       /\ prefixState' = [prefixState EXCEPT ![c] = outcome]
       /\ removed' = IF outcome = "fault" THEN removed \cup {c} ELSE removed
       /\ removalCause' = IF outcome = "fault"
                          THEN [removalCause EXCEPT ![c] = "prefix_fault"]
                          ELSE removalCause
  /\ UNCHANGED <<phase, attempt, workState, startedFrom, maxRead, termRows,
                  termKind, stopReason, analysis, liveVars>>

\* Start or retry under the same key, restoring the one snapshot each time.
Start(r) ==
  /\ phase[r] = "idle" /\ attempt[r] < MaxAttempts
  /\ prefixState[r[1]] = "ok" /\ r[1] \notin removed
  /\ attempt' = [attempt EXCEPT ![r] = @ + 1]
  /\ phase' = [phase EXCEPT ![r] = "running"]
  /\ workState' = [workState EXCEPT ![r] =
                     IF attempt[r] = 0 \/ RestoreGuard THEN "snap" ELSE @]
  /\ startedFrom' = [startedFrom EXCEPT ![r] = workState'[r]]
  /\ builds' = IF OncePrefixGuard THEN builds
               ELSE [builds EXCEPT ![r[1]] = @ + 1]
  /\ UNCHANGED <<prefixState, removed, removalCause, maxRead, termRows,
                  termKind, stopReason, analysis, liveVars>>

\* Ordinary terminal writes require a live key with no terminal already written.
WriteTerminal(r, kind) ==
  /\ phase[r] \in {"idle", "running"} /\ termRows[r] = 0
  /\ phase' = [phase EXCEPT ![r] = "done"]
  /\ termRows' = [termRows EXCEPT ![r] = @ + 1]
  /\ termKind' = [termKind EXCEPT ![r] = kind]
  /\ removed' = IF kind = "infra_fault" THEN removed \cup {r[1]} ELSE removed
  /\ removalCause' = IF kind = "infra_fault"
                     THEN [removalCause EXCEPT ![r[1]] = "infra_fault"]
                     ELSE removalCause
  /\ analysis' = IF kind \in {"scored", "cap_stop"} /\ r[1] \notin removed
                  THEN analysis \cup {r}
                  ELSE IF kind = "infra_fault"
                       THEN IF RemovalGuard
                            THEN analysis \ {q \in RunKeys: q[1] = r[1]}
                            ELSE analysis \ {r}
                       ELSE analysis

RecordedKind(r, reason) ==
  IF reason = "cap_stop" THEN IF KindGuard THEN "cap_stop" ELSE "infra_fault"
  ELSE IF reason = "scored" /\ r[1] \in removed THEN "excluded"
  ELSE reason

\* Refuse a source record after the boundary as a terminal failure.
Read(r, t) ==
  /\ phase[r] = "running"
  /\ IF t <= Boundary \/ ~BoundaryGuard
     THEN /\ maxRead' = [maxRead EXCEPT ![r] = IF t > @ THEN t ELSE @]
          /\ UNCHANGED <<removed, removalCause, phase, termRows, termKind,
                          stopReason, analysis>>
     ELSE /\ WriteTerminal(r, "fail")
          /\ stopReason' = [stopReason EXCEPT ![r] = "fail"]
          /\ UNCHANGED maxRead
  /\ UNCHANGED <<prefixState, builds, attempt, workState, startedFrom, liveVars>>

\* Work dirties the fork's working state without changing the snapshot.
Work(r) ==
  /\ phase[r] = "running"
  /\ workState' = [workState EXCEPT ![r] = "dirty"]
  /\ UNCHANGED <<prefixState, builds, removed, removalCause, phase, attempt,
                  startedFrom, maxRead, termRows, termKind, stopReason,
                  analysis, liveVars>>

\* Retry infrastructure faults until the attempt cap, then write a terminal.
Finish(r, reason) ==
  /\ phase[r] = "running"
  /\ stopReason' = [stopReason EXCEPT ![r] = reason]
  /\ IF reason = "infra_fault" /\ attempt[r] < MaxAttempts
     THEN /\ phase' = [phase EXCEPT ![r] = "idle"]
          /\ UNCHANGED <<removed, removalCause, termRows, termKind, analysis>>
     ELSE WriteTerminal(r, RecordedKind(r, reason))
  /\ UNCHANGED <<prefixState, builds, attempt, workState, startedFrom,
                  maxRead, liveVars>>

\* Exclude an idle key after its case is removed, with no run-ending cause.
ExcludeRun(r) ==
  /\ phase[r] = "idle" /\ r[1] \in removed
  /\ WriteTerminal(r, "excluded")
  /\ stopReason' = [stopReason EXCEPT ![r] = "none"]
  /\ UNCHANGED <<prefixState, builds, attempt, workState, startedFrom,
                  maxRead, liveVars>>

\* Mutation: a late writer appends another terminal instead of a keyed refusal.
DuplicateTerminal(r) ==
  /\ phase[r] = "done" /\ termRows[r] >= 1 /\ ~UniqueTerminalGuard
  /\ termRows' = [termRows EXCEPT ![r] = @ + 1]
  /\ UNCHANGED <<prefixState, builds, removed, removalCause, phase, attempt,
                  workState, startedFrom, maxRead, termKind, stopReason,
                  analysis, liveVars>>

ReplayNext ==
  \/ \E c \in Cases: BuildPrefix(c)
  \/ \E r \in RunKeys:
       \/ Start(r) \/ Work(r) \/ ExcludeRun(r) \/ DuplicateTerminal(r)
       \/ \E t \in 0..MaxT: Read(r, t)
       \/ \E reason \in Reasons: Finish(r, reason)

\* Assign sessions in order; the armed switch guards every on assignment.
Assign(s, arm) ==
  /\ assign[s] = "unassigned"
  /\ \A earlier \in Sessions: earlier < s => assign[earlier] # "unassigned"
  /\ (arm = "on" /\ KillGuard) => switch = "armed"
  /\ assign' = [assign EXCEPT ![s] = arm]
  /\ assignedWhenTripped' = [assignedWhenTripped EXCEPT ![s] = (switch = "tripped")]
  /\ UNCHANGED <<switch, incidents, replayVars>>

\* Deduplicate incident keys and trip atomically on the second distinct key.
Incident(k) ==
  /\ assign[k[1]] = "on"
  /\ incidents' = incidents \cup {k}
  /\ switch' = IF Cardinality(incidents') >= 2 THEN "tripped" ELSE switch
  /\ UNCHANGED <<assign, assignedWhenTripped, replayVars>>

LiveNext ==
  \/ \E s \in Sessions, arm \in {"on", "off"}: Assign(s, arm)
  \/ \E k \in IncidentKeys: Incident(k)

Next == ReplayNext \/ LiveNext

\* With all guards on, changing steps consume finite progress; fairness suffices.
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* The mutation that must break Terminates: without fairness a run may stutter forever.
SpecNoFair == Init /\ [][Next]_vars

UniqueTerminal == \A r \in RunKeys: termRows[r] <= 1
NoReadAfterBoundary == \A r \in RunKeys: maxRead[r] <= Boundary
MatchedRemoval == \A c \in removed: \A a \in Arms: <<c, a>> \notin analysis
ForkRestore == \A r \in RunKeys: startedFrom[r] \in {"none", "snap"}
KindMatchesCause ==
  \A r \in RunKeys: termKind[r] # "none" =>
    \/ termKind[r] = stopReason[r]
    \/ /\ stopReason[r] = "scored" /\ r[1] \in removed
       /\ termKind[r] = "excluded"
    \/ /\ stopReason[r] = "none" /\ termKind[r] = "excluded"
OnePrefixOneSnapshot == \A c \in Cases: builds[c] <= 1
NoOnAfterTrip == \A s \in Sessions: assign[s] = "on" => ~assignedWhenTripped[s]
TypeOK ==
  /\ prefixState \in [Cases -> {"unbuilt", "ok", "fault"}]
  /\ builds \in [Cases -> 0..2]
  /\ removed \subseteq Cases
  /\ removalCause \in [Cases -> {"none", "prefix_fault", "infra_fault"}]
  /\ phase \in [RunKeys -> {"idle", "running", "done"}]
  /\ attempt \in [RunKeys -> 0..MaxAttempts]
  /\ workState \in [RunKeys -> {"none", "snap", "dirty"}]
  /\ startedFrom \in [RunKeys -> {"none", "snap", "dirty"}]
  /\ maxRead \in [RunKeys -> 0..MaxT]
  /\ termRows \in [RunKeys -> 0..2]
  /\ termKind \in [RunKeys -> {"none", "scored", "cap_stop", "infra_fault", "excluded", "fail"}]
  /\ stopReason \in [RunKeys -> {"none", "scored", "cap_stop", "infra_fault", "fail"}]
  /\ analysis \subseteq RunKeys
  /\ assign \in [Sessions -> {"unassigned", "on", "off"}]
  /\ switch \in {"armed", "tripped"}
  /\ incidents \subseteq IncidentKeys
  /\ assignedWhenTripped \in [Sessions -> BOOLEAN]
Terminates == \A r \in RunKeys: <>(termRows[r] >= 1)
====================================================================================
