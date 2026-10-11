------------------------------ MODULE LabSelfHeal ------------------------------
(***************************************************************************)
(* OMN-20771 (T2 of the 2026-10-09 process-holes plan, epic OMN-20768).    *)
(*                                                                         *)
(* The lab that heals itself, as section 4 of the plan designs it:         *)
(*   (A) 4.1: the deploy agent on the lab host converges and rolls back    *)
(*       the dev lane from the watched heads, outside the lane;            *)
(*   (B) 4.2: a host-installed guard folds post-merge lab-pass receipts    *)
(*       into one incident per lane, holds runtime-affecting merges        *)
(*       through a required hold job that reads a guard-owned variable     *)
(*       and heartbeat, proves a culprit by a reproduced differential and  *)
(*       opens at most one revert for it.                                 *)
(*                                                                         *)
(* Abstractions (each one is a modelling choice the review can reject):    *)
(*   - One watched repository. A pin is an index into the merged log       *)
(*     (0 = the base commit); pin a is an ancestor of pin b iff a < b.     *)
(*     Every PR in the model is runtime-affecting (the non-runtime path    *)
(*     passes the hold job whatever the variable says and changes no       *)
(*     property here).                                                     *)
(*   - Candidate key = [pin, target migration head]; the overlay digest  *)
(*     is one constant value. An image id adds the build timestamp.  *)
(*   - Composition changes take effect when the job starts; the job's     *)
(*     outcome is its receipt. Receipt production is in-process on the     *)
(*     host (T3), so the agent knows its own verdicts at once; the guard   *)
(*     reads the same receipts later, in any order (delayed, reordered).   *)
(*   - The landing branch is queue-controlled (plan section 2, premise 7): *)
(*     every landing is a merge group built from the queued PRs; its       *)
(*     hold check is computed when the group is built and the group lands *)
(*     in a later step, so a stale green group is representable.           *)
(*   - Time is abstract. The 20-minute heartbeat bound is an action that   *)
(*     marks the heartbeat stale while the guard is down; the 30- and      *)
(*     60-minute guard deadlines are an escalation action enabled while    *)
(*     the incident waits; probe windows are a boolean the environment     *)
(*     opens and closes.                                                   *)
(*                                                                         *)
(* Design switches: every constant below whose name is not a bound is a   *)
(* design rule. TRUE is the design; each mutant configuration sets exactly *)
(* one switch FALSE and must make exactly the property it protects fail.  *)
(* The Fair* switches name the fairness assumptions of the two liveness   *)
(* properties.                                                             *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS
    Feats,            \* ids of feature PRs, e.g. {"f1", "f2"}
    MaxGen,           \* bound on deployment generations
    MaxNoise,         \* bound on flaky, transport-class and invalid-binding receipts
    MaxGuardCrash,    \* bound on guard crashes
    MaxAgentFault,    \* bound on agent self-update failures and container crashes
    InitCut,          \* cutover step the run starts at (0: before T4; 4: watcher enabled)
    InitGuardReady,   \* TRUE: the run starts with the guard installed and the hold job required
    \* 4.1 rules
    UseLaneLock,             \* SingleWriterPerLane: one composition-changing job at a time
    FenceBeforeEnable,       \* SingleWriterPerLane: watcher enabled only after the fence (cutover order)
    DurableAttempts,         \* OneAttemptPerIdentity: the attempt store survives an agent restart
    KeyByCandidate,          \* AttemptsKeyedByCandidateKey: attempts keyed by candidate key, not image id
    BlockAttributedTarget,   \* NoConvergeToAttributedFail
    AncestorCheck,           \* NoBackwardConvergeExceptRollback
    SelfUpdateProvenOnly,    \* AgentRunsOnlyProvenRevision
    RetireAfterVerify,       \* NoRetireHotpatchBeforeVerifiedReplacement
    GenPerCompositionChange, \* EveryCompositionChangeInvalidatesProof
    \* 4.2 rules
    RevertIdempotent,        \* AtMostOneRevertPerCulprit: idempotency key (repo, culprit sha)
    RequireReproduction,     \* NoRevertWithoutReproducedDifferential
    RequireAttribution,      \* NoRevertOnUnattributableFail
    RequireSingleCandidate,  \* NoRevertOnAmbiguousCulprit
    WaitAfterAuthorizedMerge,\* NoFixAndRevertBothLandWithoutCheck
    SingleAuthorization,     \* OneAuthorizationPerIncident
    HoldJobOnMergeGroup,     \* NoRuntimeMergeWhileHeldOrGuardStale
    ReleaseOnCurrentGenOnly, \* NoReleaseExceptOnCurrentGenerationPass
    InvalidateOnHoldOpen,    \* NoPassingCheckSurvivesHoldOpen
    GroupAllMembers,         \* GroupPassesOnlyWithAllRuntimeMembersAuthorized
    RequireAfterHeartbeat,   \* NoEnforcementBeforeHeartbeat
    \* fairness assumptions (liveness)
    FairHostTimer,           \* the agent's host timer evaluates the lane even when its consumer is down
    FairGuardRestart         \* a crashed guard is restarted

PRs   == Feats \cup {"rv", "fx"}       \* rv: the guard's revert; fx: a lane's fix naming the incident
Kinds == {"deploy", "rollback", "recreate", "migrate", "repro"}
IncSt == {"none", "opening", "invalidating", "open", "repro", "confirmed",
          "remediating", "escalated", "released"}
NoJob == [kind |-> "none", tgt |-> 0, who |-> "none"]

VARIABLES
    \* the watched repository and GitHub
    log,        \* Seq of [pr, kind \in {"feat","revert","fix"}, tgt, bad]; pin i = Len prefix i
    pr,         \* [PRs -> [st, chk, bad, tgt]]
    group,      \* the merge group being tested: [mem: SUBSET PRs, chk, cleanAdmit]
    \* the deploy agent (4.1)
    running,    \* pin of the running composition
    migA,       \* applied migration head
    migT,       \* target migration head
    crashed,    \* a container of the current composition is crash-looping (not a code defect)
    gen,        \* deployment generation (persisted in the job store)
    comp,       \* ghost: number of composition changes so far
    job,        \* the job holding the lane lock
    attempts,   \* attempt store (candidate keys, or image ids under the mutation)
    attHist,    \* ghost: every converge attempt, as [ck, ts]
    agentUp,    \* the agent process is running
    consumerUp, \* the agent's command consumer can poll the lane's broker
    rev,        \* the agent's own revision (a pin of this repository)
    prevRev,    \* the revision to restart from when a self-update fails to start
    passPins,   \* pins the agent's own receipts PASSed (proven revisions)
    failPins,   \* pins the agent's own receipts FAILed attributably
    lastPass,   \* newest PASS pin, the rollback target (rule RB is abstracted to "newest PASS")
    hp,         \* hot-patch row: "active" or "retired"
    hpVerified, \* ghost: a replacing build containing the hot-patch source PASSed
    recreated,  \* candidate keys that got their one service recreation
    reproReq,   \* the guard's diagnostic reproduction request waiting in the agent's intake
    cut,        \* cutover step of 4.1: 0 before T4 .. 4 watcher enabled (T7)
    watcherOn,  \* the overlay enables the watcher for the dev lane
    ciQueue,    \* CI dev-lane start commands waiting for the agent
    inWindow,   \* a probe window is open
    \* receipts
    inflight,   \* receipts published and not yet read by the guard
    noise,      \* receipts that were flaky, transport-class or invalid so far
    \* the guard (4.2)
    gUp, gCrashes,
    installed,  \* the guard unit is installed and writing heartbeats
    required,   \* the hold job is a required check on dev
    hbFresh,    \* the variable's heartbeat is younger than 20 minutes
    held,       \* the variable says held
    auth,       \* PRs the variable authorizes
    inc,        \* the lane's incident
    seen,       \* receipts the guard has read (retained by identity and generation)
    revertCount,\* ghost: revert PRs created per culprit pin
    faults,     \* agent faults used so far
    ghost       \* ghost: names of event properties an action broke

vars == <<log, pr, group, running, migA, migT, crashed, gen, comp, job, attempts,
          attHist, agentUp, consumerUp, rev, prevRev, passPins, failPins,
          lastPass, hp, hpVerified, recreated, reproReq, cut, watcherOn, ciQueue,
          inWindow, inflight, noise, gUp, gCrashes, installed, required, hbFresh,
          held, auth, inc, seen, revertCount, faults, ghost>>

agentVars == <<running, migA, migT, crashed, gen, comp, job, attempts, attHist,
               agentUp, consumerUp, rev, prevRev, passPins, failPins, lastPass, hp,
               hpVerified, recreated, reproReq, faults>>
guardVars == <<gUp, gCrashes, installed, required, hbFresh, held, auth, inc, seen,
               revertCount>>
repoVars  == <<log, pr, group>>
cutVars   == <<cut, watcherOn, ciQueue>>

-----------------------------------------------------------------------------
Tip == Len(log)
Pins == 0..Tip

\* A pin is broken when it contains a bad commit that no later commit at or
\* below the pin reverts or fixes.
Broken(p) ==
    \E k \in 1..p : /\ log[k].bad
                    /\ ~\E j \in 1..p : log[j].kind \in {"revert", "fix"} /\ log[j].tgt = k

CKey(p)    == [p |-> p, m |-> migT]
ImageId(p) == [p |-> p, m |-> migT, ts |-> gen]   \* the build timestamp moves with every build
StoreKey(p) == IF KeyByCandidate THEN CKey(p) ELSE ImageId(p)

Healthy == ~Broken(running) /\ ~crashed

\* A receipt: generation, pin, the job that produced it, verdict, class
\* ("assert", "transport") and binding validity.
Rcpt(g, p, k, v, c, ok, cm) ==
    [gen |-> g, pin |-> p, kind |-> k, verdict |-> v, cls |-> c, valid |-> ok, comp |-> cm]

\* 4.2 attribution: valid binding, an assertion-class failing check on a
\* revert_on dimension, no transport- or budget-class failing check.
Attributable(r) == r.verdict = "FAIL" /\ r.valid /\ r.cls = "assert"

NoInc == [st |-> "none", fails |-> {}, culprit |-> 0,
          reported |-> FALSE, reproAsked |-> FALSE, wait |-> 0, merged |-> {}]

Writers == (IF cut < 2 THEN {"ci"} ELSE {}) \cup (IF watcherOn THEN {"watcher"} ELSE {})

ActiveInc == {"opening", "invalidating", "open", "repro", "confirmed", "remediating"}
FailPin   == IF inc.fails = {} THEN 0 ELSE CHOOSE f \in inc.fails : \A g \in inc.fails : g <= f
MaxSeenGen == IF seen = {} THEN 0 ELSE CHOOSE g \in {r.gen : r \in seen} : \A r \in seen : r.gen <= g
SetToSeq(S) == CHOOSE f \in [1..Cardinality(S) -> S] : \A i, j \in DOMAIN f : i # j => f[i] # f[j]
PRKind(x)  == IF x = "rv" THEN "revert" ELSE IF x = "fx" THEN "fix" ELSE "feat"
PosOf(x)   == IF \E k \in 1..Tip : log[k].pr = x THEN CHOOSE k \in 1..Tip : log[k].pr = x ELSE 0

-----------------------------------------------------------------------------
Init ==
    /\ log = <<>>
    /\ pr = [x \in PRs |-> [st |-> "none", chk |-> "none", bad |-> FALSE, tgt |-> 0]]
    /\ group = [mem |-> {}, chk |-> "none", allAuth |-> TRUE, fresh |-> TRUE]
    /\ running = 0 /\ migA = 0 /\ migT = 0 /\ crashed = FALSE
    /\ gen = 0 /\ comp = 0 /\ job = NoJob
    /\ attempts = {} /\ attHist = {}
    /\ agentUp = TRUE /\ consumerUp = TRUE /\ rev = 0 /\ prevRev = 0
    /\ passPins = {0} /\ failPins = {} /\ lastPass = 0
    /\ hp = "active" /\ hpVerified = FALSE
    /\ recreated = {} /\ reproReq = 0
    /\ cut = InitCut /\ watcherOn = (InitCut = 4) /\ ciQueue = 0 /\ inWindow = "before"
    /\ inflight = {} /\ noise = 0
    /\ gUp = InitGuardReady /\ gCrashes = 0 /\ installed = InitGuardReady
    /\ required = InitGuardReady /\ hbFresh = InitGuardReady /\ held = FALSE /\ auth = {} /\ inc = NoInc /\ seen = {}
    /\ revertCount = <<>>
    /\ faults = 0
    /\ ghost = {}

-----------------------------------------------------------------------------
(***************************************************************************)
(* The deploy agent (4.1).                                                 *)
(***************************************************************************)

\* Every composition-changing effect starts under the lane lock and
\* allocates a generation; the migration writer comes under the same policy.
\* tags: names of event properties this start breaks, computed by the caller
\* from state the design rule does not read.
StartJob(k, t, who, tags) ==
    /\ UseLaneLock => job.kind = "none"
    /\ gen < MaxGen
    /\ job' = [kind |-> k, tgt |-> t, who |-> who]
    /\ gen' = IF k = "migrate" /\ ~GenPerCompositionChange THEN gen ELSE gen + 1
    /\ comp' = comp + 1
    /\ ghost' = ghost \cup tags
                      \cup (IF job.kind # "none" THEN {"SingleWriterPerLane"} ELSE {})

\* QueuedFirst: never while a routed command waits. After the fence a CI
\* command is not routed to the agent (it is refused when read).
QueuedFirst == (cut < 2 => ciQueue = 0) /\ reproReq = 0

WatcherTags(t) ==
    (IF t \in failPins THEN {"NoConvergeToAttributedFail"} ELSE {})
    \cup (IF t < running THEN {"NoBackwardConvergeExceptRollback"} ELSE {})

\* CONVERGE: the watcher read a head t with git ls-remote (a read may lag the
\* true head, so t is any merged pin) and the running identity differs.
\* QueuedFirst: no routed command waits. NoConvergeAtProbe. A hot-patch row
\* whose source (commit 1) is not in the target pins makes the decision REFUSE.
Converge ==
    /\ watcherOn /\ inWindow # "open"
    /\ QueuedFirst
    /\ \E t \in Pins :
        /\ t # running
        /\ AncestorCheck => t > running
        /\ StoreKey(t) \notin attempts
        /\ BlockAttributedTarget => t \notin failPins
        /\ ~(hp = "active" /\ t = 0)
        /\ StartJob("deploy", t, "watcher", WatcherTags(t))
        /\ running' = t /\ crashed' = FALSE
        /\ attempts' = attempts \cup {StoreKey(t)}
        /\ attHist' = attHist \cup {[ck |-> CKey(t), gen |-> gen]}
    /\ UNCHANGED <<migA, migT, agentUp, consumerUp, rev, prevRev, passPins, failPins,
                   lastPass, hp, hpVerified, recreated, reproReq, faults>>

\* An unhealthy lane at its watched pins with no attributable FAIL gets one
\* service recreation per candidate key (no rebuild), then escalates.
Recreate ==
    /\ watcherOn /\ inWindow # "open" /\ QueuedFirst
    /\ running = Tip /\ ~Healthy /\ running \notin failPins
    /\ CKey(running) \notin recreated
    /\ StartJob("recreate", running, "watcher", {})
    /\ crashed' = FALSE /\ recreated' = recreated \cup {CKey(running)}
    /\ UNCHANGED <<running, migA, migT, attempts, attHist, agentUp, consumerUp, rev,
                   prevRev, passPins, failPins, lastPass, hp, hpVerified, reproReq, faults>>

\* Migration convergence: the agent's migration writer under the same lock
\* and generation counter.
MigrateConverge ==
    /\ watcherOn /\ inWindow # "open" /\ QueuedFirst
    /\ migA # migT
    /\ StartJob("migrate", running, "watcher", {})
    /\ migA' = migT
    /\ UNCHANGED <<running, migT, crashed, attempts, attHist, agentUp, consumerUp, rev,
                   prevRev, passPins, failPins, lastPass, hp, hpVerified, recreated,
                   reproReq, faults>>

\* The guard's diagnostic reproduction, from the agent's intake: not an
\* ordinary converge attempt; restoring the last PASS identity follows it.
ServeRepro ==
    /\ reproReq # 0
    /\ StartJob("repro", reproReq, "guard", {})
    /\ running' = reproReq /\ reproReq' = 0
    /\ UNCHANGED <<migA, migT, crashed, attempts, attHist, agentUp, consumerUp, rev,
                   prevRev, passPins, failPins, lastPass, hp, hpVerified, recreated, faults>>

\* A CI dev-lane start command (the pre-cutover path through the lane's own
\* orchestrator). Before the fence the agent obeys it; after, it refuses it.
ServeCI ==
    /\ ciQueue > 0
    /\ ciQueue' = 0
    /\ IF cut >= 2
       THEN UNCHANGED <<job, gen, comp, ghost, running, crashed>>
       ELSE \/ /\ Tip # running
               /\ StartJob("deploy", Tip, "ci", {})
               /\ running' = Tip /\ crashed' = FALSE
            \/ /\ Tip = running
               /\ UNCHANGED <<job, gen, comp, ghost, running, crashed>>
    /\ UNCHANGED <<migA, migT, attempts, attHist, agentUp, consumerUp, rev, prevRev,
                   passPins, failPins, lastPass, hp, hpVerified, recreated, reproReq, faults>>

\* One evaluation of the agent's single policy. The consumer path reads the
\* lane's own broker; the host timer runs the same evaluation without it.
Decide == Converge \/ Recreate \/ MigrateConverge \/ ServeRepro
ConsumerEval == agentUp /\ consumerUp /\ ((Decide /\ UNCHANGED ciQueue) \/ ServeCI)
TimerEval    == agentUp /\ Decide

\* Outcomes of the in-process lab pass on the running composition: the true
\* verdict, or (bounded) a flaky result, a transport-class failure or an
\* invalid binding.
TrueOutcome == IF Broken(running) \/ crashed
               THEN [v |-> "FAIL", c |-> "assert", ok |-> TRUE]
               ELSE [v |-> "PASS", c |-> "assert", ok |-> TRUE]
NoisyOutcomes ==
    { [v |-> IF TrueOutcome.v = "PASS" THEN "FAIL" ELSE "PASS", c |-> "assert", ok |-> TRUE],
      [v |-> "FAIL", c |-> "transport", ok |-> TRUE],
      [v |-> "FAIL", c |-> "assert", ok |-> FALSE] }

\* The holder finishes: it publishes the receipt for the job's generation.
\* The existing post-verification rollback (now asking the decision node)
\* restores the newest PASS identity after a FAILed deploy, and a diagnostic
\* reproduction is always followed by restoring the last PASS identity.
FinishJob ==
    /\ job.kind # "none"
    /\ \E o \in {TrueOutcome} \cup (IF noise < MaxNoise THEN NoisyOutcomes ELSE {}) :
        LET r    == Rcpt(gen, running, job.kind, o.v, o.c, o.ok, comp)
            back == (job.kind = "deploy" /\ o.v = "FAIL") \/ job.kind = "repro"
        IN
        /\ inflight' = inflight \cup {r}
        /\ noise' = IF o = TrueOutcome THEN noise ELSE noise + 1
        /\ passPins' = IF o.v = "PASS" THEN passPins \cup {running} ELSE passPins
        /\ failPins' = IF Attributable(r) THEN failPins \cup {running} ELSE failPins
        /\ lastPass' = IF o.v = "PASS" /\ job.kind # "repro" THEN running ELSE lastPass
        /\ hpVerified' = (hpVerified \/ (o.v = "PASS" /\ running >= 1))
        /\ IF back /\ gen < MaxGen
           THEN /\ job' = [kind |-> "rollback", tgt |-> lastPass, who |-> "agent"]
                /\ gen' = gen + 1 /\ comp' = comp + 1
                /\ running' = lastPass /\ crashed' = FALSE
           ELSE /\ job' = NoJob
                /\ UNCHANGED <<gen, comp, running, crashed>>
    /\ UNCHANGED <<migA, migT, attempts, attHist, agentUp, consumerUp, rev, prevRev, hp,
                   recreated, reproReq, faults, ghost>>

\* Self-update: re-executes the agent. The agent moves only to a revision
\* whose receipt PASSed, keeps the previous one, and an update may break its
\* startup. A restart keeps the durable attempt store.
SelfUpdate ==
    /\ agentUp /\ job.kind = "none"
    /\ \E r2 \in (IF SelfUpdateProvenOnly THEN passPins ELSE Pins) :
        /\ r2 > rev
        /\ prevRev' = rev /\ rev' = r2
    /\ attempts' = IF DurableAttempts THEN attempts ELSE {}
    /\ \/ agentUp' = TRUE /\ UNCHANGED faults
       \/ faults < MaxAgentFault /\ agentUp' = FALSE /\ faults' = faults + 1
    /\ UNCHANGED <<running, migA, migT, crashed, gen, comp, job, attHist, consumerUp,
                   passPins, failPins, lastPass, hp, hpVerified, recreated, reproReq,
                   inflight, noise, ghost>>

\* The agent process dies between jobs (OOM, host reboot). The supervisor
\* restarts it on the same revision; only the durable store survives.
AgentCrash ==
    /\ agentUp /\ job.kind = "none" /\ faults < MaxAgentFault
    /\ agentUp' = FALSE /\ prevRev' = rev /\ faults' = faults + 1
    /\ attempts' = IF DurableAttempts THEN attempts ELSE {}
    /\ UNCHANGED <<running, migA, migT, crashed, gen, comp, job, attHist, consumerUp, rev,
                   passPins, failPins, lastPass, hp, hpVerified, recreated, reproReq,
                   inflight, noise, ghost>>

\* The supervisor restarts the agent on its previous revision.
RestartPrevious ==
    /\ ~agentUp
    /\ agentUp' = TRUE /\ rev' = prevRev
    /\ UNCHANGED <<running, migA, migT, crashed, gen, comp, job, attempts, attHist,
                   consumerUp, prevRev, passPins, failPins, lastPass, hp, hpVerified,
                   recreated, reproReq, inflight, noise, faults, ghost>>

AgentEnvUnch == UNCHANGED <<running, migA, gen, comp, job, attempts, attHist, agentUp,
                            rev, prevRev, passPins, failPins, lastPass, hpVerified,
                            recreated, reproReq, inflight, noise, ghost>>

\* The agent's consumer fails (broker down, construction error); its host
\* timer keeps running. It may recover, with no fairness promised.
ConsumerFail    == consumerUp /\ faults < MaxAgentFault /\ consumerUp' = FALSE
                   /\ faults' = faults + 1 /\ UNCHANGED <<migT, crashed, hp>> /\ AgentEnvUnch
ConsumerRecover == ~consumerUp /\ consumerUp' = TRUE
                   /\ UNCHANGED <<migT, crashed, hp, faults>> /\ AgentEnvUnch
ContainerCrash  == ~crashed /\ faults < MaxAgentFault /\ crashed' = TRUE
                   /\ faults' = faults + 1 /\ UNCHANGED <<migT, consumerUp, hp>> /\ AgentEnvUnch
AdvanceMigration == migT = 0 /\ migT' = 1
                   /\ UNCHANGED <<crashed, consumerUp, hp, faults>> /\ AgentEnvUnch

\* hotpatch-retire-5d21's command, under the deployment lock: only after the
\* replacing build is verified (and the tripwire reads clean).
RetireHotpatch ==
    /\ hp = "active"
    /\ IF RetireAfterVerify
       THEN hpVerified /\ job.kind = "none"
       ELSE job.kind = "deploy" /\ job.tgt >= 1
    /\ hp' = "retired"
    /\ UNCHANGED <<migT, crashed, consumerUp, faults>> /\ AgentEnvUnch

\* The cutover of 4.1, in order: T4 ships the watcher disabled; T5 fences
\* foreign dev-lane commands, removes CI's dev-lane publish and drains the
\* queued commands; T6 moves omnimarket's pin; T7 enables the watcher.
CutStep ==
    /\ cut < 4
    /\ cut' = cut + 1
    /\ ciQueue' = IF cut = 1 THEN 0 ELSE ciQueue
    /\ watcherOn' = (watcherOn \/ cut = 3)
EnableEarly ==
    /\ ~FenceBeforeEnable /\ cut >= 1 /\ ~watcherOn
    /\ watcherOn' = TRUE
    /\ UNCHANGED <<cut, ciQueue>>

WindowStep ==
    /\ inWindow # "after"
    /\ inWindow' = IF inWindow = "before" THEN "open" ELSE "after"

-----------------------------------------------------------------------------
(***************************************************************************)
(* The guard (4.2): a host unit on the ledger host, outside every lane.    *)
(* Its incident state is durable; a crash loses only the step in flight.   *)
(***************************************************************************)
GUnch == UNCHANGED <<repoVars, agentVars, cutVars, inWindow, noise>>

EmptyGroup == [mem |-> {}, chk |-> "none", allAuth |-> TRUE, fresh |-> TRUE]

\* Evidence about the FAIL identity p, read from the retained receipts.
\* Attribution is a property of that identity: every retained FAIL receipt
\* of it is attributable.
AttrAt(p)        == \A r \in seen : (r.verdict = "FAIL" /\ r.pin = p) => Attributable(r)
ReproFailedAt(p) == \E r \in seen : r.kind = "repro" /\ r.pin = p /\ Attributable(r)
RbPass(p)        == \E r \in seen : r.kind = "rollback" /\ r.verdict = "PASS" /\ r.valid /\ r.pin = p
RbPassBelow(p)   == \E q \in 0..(p - 1) : RbPass(q)
NewestRbBelow(p) == IF RbPassBelow(p)
                    THEN CHOOSE q \in 0..(p - 1) : RbPass(q) /\ \A q2 \in 0..(p - 1) : RbPass(q2) => q2 <= q
                    ELSE 0
IncAttr     == AttrAt(FailPin)
ReproFailed == ReproFailedAt(FailPin)

GInstall ==
    /\ ~installed
    /\ installed' = TRUE /\ gUp' = TRUE /\ hbFresh' = TRUE
    /\ UNCHANGED <<gCrashes, required, held, auth, inc, seen, revertCount, inflight, ghost>> /\ GUnch

GHeartbeat ==
    /\ gUp /\ ~hbFresh /\ hbFresh' = TRUE
    /\ UNCHANGED <<gUp, gCrashes, installed, required, held, auth, inc, seen, revertCount,
                   inflight, ghost>> /\ GUnch

\* 20 minutes pass with no heartbeat while the guard is down.
HbExpire ==
    /\ installed /\ ~gUp /\ hbFresh /\ hbFresh' = FALSE
    /\ UNCHANGED <<gUp, gCrashes, installed, required, held, auth, inc, seen, revertCount,
                   inflight, ghost>> /\ GUnch

\* T9 makes the hold job required on dev only after reading the heartbeat fresh.
MakeRequired ==
    /\ ~required
    /\ RequireAfterHeartbeat => installed /\ hbFresh
    /\ required' = TRUE
    /\ ghost' = ghost \cup (IF installed /\ hbFresh THEN {} ELSE {"NoEnforcementBeforeHeartbeat"})
    /\ UNCHANGED <<gUp, gCrashes, installed, hbFresh, held, auth, inc, seen, revertCount,
                   inflight>> /\ GUnch

GCrash ==
    /\ gUp /\ gCrashes < MaxGuardCrash
    /\ gUp' = FALSE /\ gCrashes' = gCrashes + 1
    /\ UNCHANGED <<installed, required, hbFresh, held, auth, inc, seen, revertCount,
                   inflight, ghost>> /\ GUnch

GRestart ==
    /\ installed /\ ~gUp
    /\ gUp' = TRUE /\ hbFresh' = TRUE
    /\ UNCHANGED <<gCrashes, installed, required, held, auth, inc, seen, revertCount,
                   inflight, ghost>> /\ GUnch

GUnchG == UNCHANGED <<gUp, gCrashes, installed, required, hbFresh, revertCount>>

\* Read one receipt, in any order. A FAIL opens an incident unless its
\* generation is older than the newest the guard has seen (generation orders
\* current-state updates only); a FAIL during an open incident joins it.
\* Every receipt is retained by identity and generation.
GRead ==
    /\ gUp
    /\ \E r \in inflight :
        /\ inflight' = inflight \ {r}
        /\ seen' = seen \cup {r}
        /\ inc' = IF r.verdict # "FAIL" THEN inc
                  ELSE IF inc.st \in {"none", "released"}
                       THEN IF r.gen >= MaxSeenGen
                            THEN [NoInc EXCEPT !.st = "opening", !.fails = {r.pin}]
                            ELSE inc
                  ELSE IF inc.st \in ActiveInc
                       THEN [inc EXCEPT !.fails = @ \cup {r.pin}]
                  ELSE inc
    /\ UNCHANGED <<held, auth, ghost>> /\ GUnchG /\ GUnch

\* Write held, then invalidate earlier green checks and queued entries, and
\* only then report the hold open.
GHoldOpen ==
    /\ gUp /\ inc.st = "opening"
    /\ held' = TRUE
    /\ inc' = [inc EXCEPT !.st = IF InvalidateOnHoldOpen THEN "invalidating" ELSE "open",
                          !.reported = ~InvalidateOnHoldOpen]
    /\ UNCHANGED <<auth, seen, inflight, ghost>> /\ GUnchG /\ GUnch

GInvalidate ==
    /\ gUp /\ inc.st = "invalidating"
    /\ pr' = [x \in PRs |-> IF pr[x].st \in {"open", "queued"} /\ x \notin auth
                            THEN [pr[x] EXCEPT !.st = "open", !.chk = "none"] ELSE pr[x]]
    /\ group' = IF group.mem \ auth # {} THEN EmptyGroup ELSE group
    /\ inc' = [inc EXCEPT !.st = "open", !.reported = TRUE]
    /\ UNCHANGED <<log, agentVars, cutVars, inWindow, noise, held, auth, seen, inflight, ghost>>
    /\ GUnchG

\* Attribution: an unattributable incident holds and escalates; otherwise
\* the guard asks the agent's intake for one diagnostic reproduction.
GDecide ==
    /\ gUp /\ inc.st = "open"
    /\ IF RequireAttribution /\ ~IncAttr
       THEN inc' = [inc EXCEPT !.st = "escalated"] /\ UNCHANGED reproReq
       ELSE /\ inc' = [inc EXCEPT !.st = "repro", !.reproAsked = TRUE]
            /\ reproReq' = FailPin
    /\ UNCHANGED <<repoVars, running, migA, migT, crashed, gen, comp, job, attempts, attHist,
                   agentUp, consumerUp, rev, prevRev, passPins, failPins, lastPass, hp,
                   hpVerified, recreated, faults, cutVars, inWindow, noise,
                   held, auth, seen, inflight, ghost>> /\ GUnchG

\* Proof of cause: the incident's FAILs are attributable (re-read here: a
\* FAIL that joined after the reproduction request counts), the rollback to
\* the newest eligible PASS identity PASSed, the FAIL identity redeployed
\* once FAILed again attributably, and exactly one runtime-affecting merge
\* separates them.
GConfirm ==
    /\ gUp /\ inc.st = "repro"
    /\ RequireAttribution => IncAttr
    /\ RbPassBelow(FailPin)
    /\ RequireReproduction => ReproFailed
    /\ RequireSingleCandidate => FailPin - NewestRbBelow(FailPin) = 1
    /\ inc' = [inc EXCEPT !.st = "confirmed", !.culprit = FailPin]
    /\ UNCHANGED <<held, auth, seen, inflight, ghost>> /\ GUnchG /\ GUnch

\* The 30- and 60-minute deadlines, an unreproduced FAIL or an ambiguous
\* culprit: the hold stays, no revert, one alarm.
GEscalate ==
    /\ gUp /\ inc.st \in {"open", "repro", "confirmed", "remediating"}
    /\ inc' = [inc EXCEPT !.st = "escalated"]
    /\ UNCHANGED <<held, auth, seen, inflight, ghost>> /\ GUnchG /\ GUnch

\* Open the revert PR (idempotency key: repo, culprit sha). The guard may
\* crash after the PR exists and before its state is persisted.
\* The checks are relative to the culprit's own FAIL identity: a later FAIL
\* that joins the incident after confirmation does not move them.
GOpenRevert ==
    /\ gUp /\ inc.st = "confirmed"
    /\ RequireAttribution => AttrAt(inc.culprit)
    /\ LET c      == inc.culprit
           exists == pr["rv"].st # "none" /\ pr["rv"].tgt = c   \* open, queued, merged or closed
           create == ~(RevertIdempotent /\ exists)
           tags   == (IF AttrAt(c) THEN {} ELSE {"NoRevertOnUnattributableFail"})
                     \cup (IF ReproFailedAt(c) /\ RbPassBelow(c) THEN {}
                           ELSE {"NoRevertWithoutReproducedDifferential"})
                     \cup (IF RbPassBelow(c) /\ c - NewestRbBelow(c) = 1 THEN {}
                           ELSE {"NoRevertOnAmbiguousCulprit"})
       IN /\ pr' = IF create
                   THEN [pr EXCEPT !["rv"] = [st |-> "open", chk |-> "none", bad |-> FALSE, tgt |-> c]]
                   ELSE pr
          /\ revertCount' = IF create THEN Append(revertCount, c) ELSE revertCount
          /\ ghost' = ghost \cup tags
          /\ \/ /\ inc' = [inc EXCEPT !.st = "remediating"]
                /\ UNCHANGED <<gUp, gCrashes>>
             \/ /\ gCrashes < MaxGuardCrash
                /\ gUp' = FALSE /\ gCrashes' = gCrashes + 1
                /\ UNCHANGED inc
    /\ UNCHANGED <<log, group, agentVars, cutVars, inWindow, noise, installed, required,
                   hbFresh, held, auth, seen, inflight>>

\* At most one authorization live per incident; after an authorized PR
\* merges, the next waits for a receipt on an identity containing it.
\* Finding F2 (PR body): the guard authorizes only while its hold job is
\* required; between T12 and T9 an authorization binds nothing, a fix lands
\* unheld, and NoFixAndRevertBothLandWithoutCheck fails.
GAuthorize ==
    /\ gUp /\ inc.st \in {"open", "repro", "confirmed", "remediating"}
    /\ required
    /\ SingleAuthorization => auth = {}
    /\ WaitAfterAuthorizedMerge => inc.wait = 0
    /\ WaitAfterAuthorizedMerge => \A y \in {"rv", "fx"} : pr[y].st = "merged" => y \in inc.merged  \* F1
    /\ \E x \in {"rv", "fx"} :
        /\ pr[x].st \in {"open", "queued"} /\ x \notin auth
        /\ x = "rv" => inc.st = "remediating"
        /\ auth' = auth \cup {x}
    /\ UNCHANGED <<held, inc, seen, inflight, ghost>> /\ GUnchG /\ GUnch

\* Finding F1 (PR body): a fix or revert can land unauthorized in the window
\* between the FAIL and the end of the hold-open invalidation. The wait is
\* therefore applied to every fix or revert merge the guard observes during
\* an incident, not only to authorized ones; with the wait applied to
\* authorized merges only, NoFixAndRevertBothLandWithoutCheck fails.
GObserveMerge ==
    /\ gUp /\ inc.st \in ActiveInc
    /\ \E x \in {"rv", "fx"} :
        /\ pr[x].st = "merged" /\ x \notin inc.merged
        /\ auth' = auth \ {x}
        /\ inc' = [inc EXCEPT !.wait = PosOf(x), !.merged = @ \cup {x}]
    /\ UNCHANGED <<held, seen, inflight, ghost>> /\ GUnchG /\ GUnch

GClearWait ==
    /\ gUp /\ inc.wait # 0
    /\ \E r \in seen : r.pin >= inc.wait
    /\ inc' = [inc EXCEPT !.wait = 0]
    /\ UNCHANGED <<held, auth, seen, inflight, ghost>> /\ GUnchG /\ GUnch

\* If a fix PASSes first, the guard closes the revert.
GCloseRevert ==
    /\ gUp /\ pr["rv"].st = "open" /\ pr["fx"].st = "merged"
    /\ \E r \in seen : r.verdict = "PASS" /\ r.pin >= PosOf("fx")
    /\ pr' = [pr EXCEPT !["rv"].st = "closed"]
    /\ auth' = auth \ {"rv"}
    /\ UNCHANGED <<log, group, agentVars, cutVars, inWindow, noise, held, inc, seen,
                   inflight, ghost>> /\ GUnchG

\* Release only on a PASS for the agent's current generation, with no
\* deployment in flight, a fresh cursor, and pins descending from every FAIL.
GRelease ==
    /\ gUp /\ inc.st \in {"open", "repro", "confirmed", "remediating"}
    /\ \E r \in seen :
        /\ r.verdict = "PASS"
        /\ LET ok == /\ r.gen = gen /\ job.kind = "none" /\ inflight = {} /\ r.valid
                     /\ \A f \in inc.fails : r.pin >= f
           IN /\ ReleaseOnCurrentGenOnly => ok
              /\ ghost' = ghost \cup (IF ok THEN {} ELSE {"NoReleaseExceptOnCurrentGenerationPass"})
    /\ held' = FALSE /\ auth' = {}
    /\ inc' = [inc EXCEPT !.st = "released", !.reported = FALSE]
    /\ UNCHANGED <<seen, inflight>> /\ GUnchG /\ GUnch

-----------------------------------------------------------------------------
(***************************************************************************)
(* GitHub: PRs, the runtime-merge-hold job and the merge queue.            *)
(***************************************************************************)
RUnch == UNCHANGED <<agentVars, guardVars, cut, watcherOn, inWindow, inflight, noise>>

\* The hold job: a runtime-affecting change fails while the variable says
\* held unless it is authorized, and fails on a stale heartbeat. The job
\* runs from T9 on; whether its red blocks depends on required.
HoldJob(x) ==
    IF ~hbFresh THEN "red"
    ELSE IF held /\ x \notin auth THEN "red"
    ELSE "green"

GroupCheck(Q) ==
    IF ~HoldJobOnMergeGroup THEN "green"
    ELSE IF ~hbFresh THEN "red"
    ELSE IF ~held THEN "green"
    ELSE IF GroupAllMembers THEN (IF Q \subseteq auth THEN "green" ELSE "red")
    ELSE IF Q \cap auth # {} THEN "green" ELSE "red"

OpenFeat ==
    \E f \in Feats, b \in BOOLEAN :
        /\ pr[f].st = "none"
        /\ pr' = [pr EXCEPT ![f] = [st |-> "open", chk |-> "none", bad |-> b, tgt |-> 0]]
        /\ UNCHANGED <<log, group, ciQueue, ghost>> /\ RUnch

\* A lane's fix PR naming the open incident; the fix itself may fail.
OpenFix ==
    /\ inc.st \in ActiveInc /\ pr["fx"].st = "none"
    /\ \E k \in 1..Tip, b \in BOOLEAN :
        /\ log[k].bad
        /\ pr' = [pr EXCEPT !["fx"] = [st |-> "open", chk |-> "none", bad |-> b, tgt |-> k]]
    /\ UNCHANGED <<log, group, ciQueue, ghost>> /\ RUnch

\* The hold job runs once per push or rerun; a finished result stays until
\* the next rerun (the guard's invalidation, or a rerun of a red check).
RunCheck ==
    \E x \in PRs :
        /\ pr[x].st = "open" /\ pr[x].chk = "none"
        /\ pr' = [pr EXCEPT ![x].chk = HoldJob(x)]
        /\ UNCHANGED <<log, group, ciQueue, ghost>> /\ RUnch

RerunRed ==
    \E x \in PRs :
        /\ pr[x].st = "open" /\ pr[x].chk = "red"
        /\ pr' = [pr EXCEPT ![x].chk = "none"]
        /\ UNCHANGED <<log, group, ciQueue, ghost>> /\ RUnch

\* The landing controller enqueues a PR whose checks are green.
Enqueue ==
    \E x \in PRs :
        /\ pr[x].st = "open" /\ (pr[x].chk = "green" \/ (~required /\ pr[x].chk # "none"))
        /\ pr' = [pr EXCEPT ![x].st = "queued"]
        /\ UNCHANGED <<log, group, ciQueue, ghost>> /\ RUnch

\* GitHub builds a merge group from the queued entries and runs the hold job
\* on it (merge_group). allAuth and fresh record what the property demands.
BuildGroup ==
    /\ group.mem = {}
    /\ LET Q == {x \in PRs : pr[x].st = "queued"}
       IN /\ Q # {}
          /\ group' = [mem |-> Q, chk |-> GroupCheck(Q),
                       allAuth |-> (~held \/ Q \subseteq auth), fresh |-> hbFresh]
    /\ UNCHANGED <<log, pr, ciQueue, ghost>> /\ RUnch

\* The group lands. Before T5, CI publishes a dev-lane start command per
\* runtime merge (omnimarket's caller until T6). A red hold job blocks the
\* landing only once it is required.
LandGroup ==
    /\ group.mem # {} /\ (group.chk = "green" \/ ~required)
    /\ LET s   == SetToSeq(group.mem)
           rf  == group.mem \cap {"rv", "fx"}
           prf == {y \in {"rv", "fx"} : pr[y].st = "merged"}
           unchecked == Cardinality(rf) = 2
                        \/ (rf # {} /\ \E y \in prf : ~\E r \in seen : r.pin >= PosOf(y))
           tags == (IF required /\ (~group.fresh \/ ~group.allAuth
                                     \/ (held /\ inc.reported /\ ~(group.mem \subseteq auth)))
                    THEN {"NoRuntimeMergeWhileHeldOrGuardStale"} ELSE {})
                   \cup (IF required /\ unchecked THEN {"NoFixAndRevertBothLandWithoutCheck"} ELSE {})
       IN /\ log' = log \o [i \in 1..Len(s) |->
                       [pr |-> s[i], kind |-> PRKind(s[i]), tgt |-> pr[s[i]].tgt, bad |-> pr[s[i]].bad]]
          /\ ghost' = ghost \cup tags
    /\ pr' = [x \in PRs |-> IF x \in group.mem THEN [pr[x] EXCEPT !.st = "merged"] ELSE pr[x]]
    /\ group' = EmptyGroup
    /\ ciQueue' = IF cut < 3 THEN 1 ELSE ciQueue
    /\ RUnch

GroupFail ==
    /\ group.mem # {} /\ group.chk = "red" /\ required
    /\ pr' = [x \in PRs |-> IF x \in group.mem THEN [pr[x] EXCEPT !.st = "open", !.chk = "none"]
                            ELSE pr[x]]
    /\ group' = EmptyGroup
    /\ UNCHANGED <<log, ciQueue, ghost>> /\ RUnch

-----------------------------------------------------------------------------
OtherUnch == UNCHANGED <<repoVars, guardVars, inflight, noise>>

\* Each step in full (every variable specified), so it can carry fairness.
ConsumerA == ConsumerEval /\ UNCHANGED <<guardVars, repoVars, cut, watcherOn, inWindow, inflight, noise>>
TimerA    == TimerEval /\ UNCHANGED <<guardVars, repoVars, cut, watcherOn, ciQueue, inWindow, inflight, noise>>
FinishA   == FinishJob /\ UNCHANGED <<guardVars, repoVars, cutVars, inWindow>>
RestartA  == RestartPrevious /\ UNCHANGED <<guardVars, repoVars, cutVars, inWindow>>
CutA      == CutStep /\ UNCHANGED <<agentVars, guardVars, repoVars, inWindow, inflight, noise, ghost>>
WindowA   == WindowStep /\ UNCHANGED <<agentVars, guardVars, repoVars, cutVars, inflight, noise, ghost>>

AgentNext ==
    \/ ConsumerA \/ TimerA \/ FinishA \/ RestartA
    \/ (SelfUpdate \/ AgentCrash) /\ UNCHANGED <<guardVars, repoVars, cutVars, inWindow>>
    \/ (ConsumerFail \/ ConsumerRecover \/ ContainerCrash \/ AdvanceMigration \/ RetireHotpatch)
       /\ UNCHANGED <<guardVars, repoVars, cutVars, inWindow>>

EnvNext ==
    \/ CutA \/ WindowA
    \/ EnableEarly /\ UNCHANGED <<agentVars, guardVars, repoVars, inWindow, inflight, noise, ghost>>

GuardNext ==
    \/ GInstall \/ GHeartbeat \/ HbExpire \/ MakeRequired \/ GCrash \/ GRestart
    \/ GRead \/ GHoldOpen \/ GInvalidate \/ GDecide \/ GConfirm \/ GEscalate
    \/ GOpenRevert \/ GAuthorize \/ GObserveMerge \/ GClearWait \/ GCloseRevert \/ GRelease

RepoNext == OpenFeat \/ OpenFix \/ RunCheck \/ RerunRed \/ Enqueue \/ BuildGroup \/ LandGroup \/ GroupFail

Next == AgentNext \/ EnvNext \/ GuardNext \/ RepoNext

\* Declared fairness. Jobs finish (every job is killed at its ceiling), the
\* probe window closes, a failed agent start restarts on its previous
\* revision, the cutover completes, the guard reads, decides and releases
\* while up, and the queue moves. FairHostTimer and FairGuardRestart are the
\* two assumptions the liveness properties name.
Fairness ==
    /\ WF_vars(FinishA) /\ WF_vars(WindowA) /\ WF_vars(RestartA) /\ WF_vars(CutA)
    /\ WF_vars(ConsumerA)
    /\ (FairHostTimer => WF_vars(TimerA))
    /\ (FairGuardRestart => WF_vars(GRestart))
    /\ WF_vars(GInstall) /\ WF_vars(GHeartbeat) /\ WF_vars(MakeRequired)
    /\ WF_vars(GRead) /\ WF_vars(GHoldOpen) /\ WF_vars(GInvalidate) /\ WF_vars(GDecide)
    /\ WF_vars(GConfirm) /\ WF_vars(GOpenRevert) /\ WF_vars(GAuthorize) /\ WF_vars(GObserveMerge)
    /\ WF_vars(GClearWait) /\ WF_vars(GEscalate) /\ WF_vars(GRelease)

Spec == Init /\ [][Next]_vars /\ Fairness

-----------------------------------------------------------------------------
(***************************************************************************)
(* Safety. State properties read state the design rule does not compute;  *)
(* event properties are checked by the action that would break them, from *)
(* state the guarded rule does not read, and recorded in ghost.            *)
(***************************************************************************)
TypeOK ==
    /\ job.kind \in Kinds \cup {"none"}
    /\ inc.st \in IncSt
    /\ cut \in 0..4
    /\ gen \in 0..MaxGen

\* One writer of the dev lane at every cutover step: the lane lock admits one
\* composition-changing job, and the CI path and the watcher are never both
\* able to start a dev-lane job.
SingleWriterPerLane ==
    /\ "SingleWriterPerLane" \notin ghost
    /\ Cardinality(Writers) <= 1

OneAttemptPerIdentity ==
    \A h1, h2 \in attHist : h1.ck = h2.ck => h1 = h2

AttemptsKeyedByCandidateKey ==
    /\ \A a \in attempts : DOMAIN a = {"p", "m"}
    /\ \A h \in attHist : h.ck \in attempts \/ ~DurableAttempts

NoConvergeToAttributedFail       == "NoConvergeToAttributedFail" \notin ghost
NoBackwardConvergeExceptRollback == "NoBackwardConvergeExceptRollback" \notin ghost
AgentRunsOnlyProvenRevision      == agentUp => rev \in passPins
NoRetireHotpatchBeforeVerifiedReplacement == hp = "retired" => hpVerified

AtMostOneRevertPerCulprit ==
    \A i, j \in DOMAIN revertCount : i # j => revertCount[i] # revertCount[j]

NoRevertWithoutReproducedDifferential == "NoRevertWithoutReproducedDifferential" \notin ghost
NoRevertOnUnattributableFail          == "NoRevertOnUnattributableFail" \notin ghost
NoRevertOnAmbiguousCulprit            == "NoRevertOnAmbiguousCulprit" \notin ghost
NoFixAndRevertBothLandWithoutCheck    == "NoFixAndRevertBothLandWithoutCheck" \notin ghost
OneAuthorizationPerIncident           == Cardinality(auth) <= 1
NoRuntimeMergeWhileHeldOrGuardStale   == "NoRuntimeMergeWhileHeldOrGuardStale" \notin ghost
NoReleaseExceptOnCurrentGenerationPass == "NoReleaseExceptOnCurrentGenerationPass" \notin ghost

\* Once the hold is reported open, no unauthorized open PR keeps a green hold
\* check from before, and no queued group keeps a green result.
NoPassingCheckSurvivesHoldOpen ==
    (required /\ held /\ inc.reported) =>
        /\ \A x \in PRs : (pr[x].st \in {"open", "queued"} /\ pr[x].chk = "green") => x \in auth
        /\ (group.mem # {} /\ group.chk = "green") => group.mem \subseteq auth

GroupPassesOnlyWithAllRuntimeMembersAuthorized ==
    (required /\ group.mem # {} /\ group.chk = "green") => group.allAuth

NoEnforcementBeforeHeartbeat == "NoEnforcementBeforeHeartbeat" \notin ghost

\* A PASS counts as proof only for the generation it names; every
\* composition change allocates a new generation, so no PASS of the current
\* generation predates a composition change.
EveryCompositionChangeInvalidatesProof ==
    \A r \in seen \cup inflight : (r.verdict = "PASS" /\ r.gen = gen) => r.comp = comp

-----------------------------------------------------------------------------
(***************************************************************************)
(* Liveness.                                                               *)
(***************************************************************************)
PinRange == 0..Cardinality(PRs)

\* Checked with the liveness properties: the generation bound never disables
\* a step (StartJob and the rollback continuation need gen < MaxGen), so no
\* liveness verdict is an artifact of the bound.
GenBoundNotReached == gen < MaxGen

\* A lane with a newer watched head converges to it (or past it) or rolls
\* back off it: an attempt for it ran, or its FAIL bars it.
NewerHeadConvergesOrRollsBack ==
    \A t \in PinRange :
        (watcherOn /\ Tip >= t /\ running < t)
            ~> (running >= t \/ \E h \in attHist : h.ck.p = t \/ t \in failPins)

\* An incident ends released or escalated.
IncidentEndsReleasedOrEscalated ==
    (inc.st \in ActiveInc) ~> (inc.st \in {"released", "escalated"})
=============================================================================
