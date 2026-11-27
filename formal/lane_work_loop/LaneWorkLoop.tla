---------------------------- MODULE LaneWorkLoop ----------------------------
(* The lane work loop (OMN-26700, plan sections 3.4 to 3.6): one orchestrator
   instance folds inputs serially. Each input runs fold -> receipt publish ->
   output publishes -> offset commit; a crash may fall between any two steps and
   is followed by the recovery procedure (readback, replay of the uncommitted input).
   Environment: two start inputs for one change, worker inputs, effect completions,
   PR observations (fresh and stale), ticks, help answers, the head moving at any
   step, the worker vanishing, a caller resending with a new correlation id.
   States are the plan's sixteen collapsed to the ones the properties distinguish:
     working   = claimed, red, diagnosing, fixing, waiting_release, pushing (worker-owned)
     proving, proven, handing_off, confirming (confirming_handoff), help (help_requested)
     awaiting_worker, handed_off, blocked.
   Properties (invariants) and the mutation constant that must break each:
     OneTerminal               a workflow publishes at most one terminal      TermGuard=FALSE
     OneLiveWorkflowPerChange  at most one live workflow for the change       UniqueId=FALSE
     HandoffAtProvenHead       handed_off names request R at the last proven head
                               and a post-answer observation showed it        Confirm=FALSE
     NoLostOutput              a committed receipt's outputs were all published CommitAfter=FALSE
     NoDuplicateEffect         an effect command is published once per action Readback=FALSE
     RetryBound                only admissible failures are counted           CountAdmissible=FALSE
     AnswerInContract          a help answer folds only in help               AnswerGuard=FALSE
   Temporal (fair): DeadlineFires (a workflow whose worker vanished ends in a
   terminal; TickOn=FALSE) and Terminates (every workflow with a worker ends). *)
EXTENDS Naturals, Integers, FiniteSets
CONSTANTS MaxIn, MaxHead, MaxCrash, HelpBudget, RetryN,
          TermGuard, UniqueId, Confirm, CommitAfter, Readback,
          CountAdmissible, AnswerGuard, TickOn

W == {1, 2}                         \* attempt numbers of the one change
Term == {"handed_off", "blocked"}
Live == {"awaiting_worker", "working", "proving", "proven",
         "handing_off", "confirming", "help"}
WorkerOwned == {"awaiting_worker", "working"}
NodeOwned == {"proving", "proven", "handing_off", "confirming"}
EffectStates == {"proving", "proven", "handing_off", "help"}
TierTop == 0
NoIn == [k |-> "none", w |-> 0, h |-> 0, f |-> FALSE, a |-> "none"]

VARIABLES st, eh, ph, rq, ob, rc, rcG, hb, tier, org, gone, termPub, ansIn,
          head, hmoves, crashes, startsLeft, inCnt,
          pc, cur, rP, cm, rOuts, rW, pubd, effCnt, lost
wf == <<st, eh, ph, rq, ob, rc, rcG, hb, tier, org, gone, termPub, ansIn>>
pipe == <<pc, cur, rP, cm, rOuts, rW, pubd, effCnt, lost>>
vars == <<wf, head, hmoves, crashes, startsLeft, inCnt, pipe>>

Min(a, b) == IF a < b THEN a ELSE b
LiveSet == {w \in W : st[w] \in Live}

Init ==
  /\ st = [w \in W |-> "none"] /\ eh = [w \in W |-> 0] /\ ph = [w \in W |-> -1]
  /\ rq = [w \in W |-> -1] /\ ob = [w \in W |-> -1]
  /\ rc = [w \in W |-> 0] /\ rcG = [w \in W |-> 0]
  /\ hb = [w \in W |-> 0] /\ tier = [w \in W |-> 0]
  /\ org = [w \in W |-> "working"] /\ gone = [w \in W |-> FALSE]
  /\ termPub = [w \in W |-> 0] /\ ansIn = {}
  /\ head = 0 /\ hmoves = 0 /\ crashes = 0 /\ startsLeft = 2 /\ inCnt = 0
  /\ pc = "idle" /\ cur = NoIn /\ rP = FALSE /\ cm = FALSE
  /\ rOuts = {} /\ rW = 1 /\ pubd = {} /\ effCnt = 0 /\ lost = FALSE

(* ---- the fold: what one input does to its workflow (pure, from durable state) ---- *)
\* Result record: ok (an edge exists), new state and fields for workflow c.w.
Refuse(c) == [ok |-> FALSE, ns |-> st[c.w], neh |-> eh[c.w], nph |-> ph[c.w],
              nrq |-> rq[c.w], nob |-> ob[c.w],
              nrc |-> rc[c.w], nrcG |-> rcG[c.w], nhb |-> hb[c.w],
              ntier |-> tier[c.w], norg |-> org[c.w], ans |-> FALSE]
To(c, s) == [Refuse(c) EXCEPT !.ok = TRUE, !.ns = s]
\* Entering help from another state consumes the help budget (plan 3.4).
ToHelp(c) ==
  IF hb[c.w] < HelpBudget
    THEN [To(c, "help") EXCEPT !.nhb = hb[c.w] + 1, !.ntier = 0, !.norg = st[c.w]]
    ELSE To(c, "blocked")

Res(c) ==
  LET s == st[c.w] IN
  CASE c.k = "start" ->
         IF s = "none" /\ (c.w = 1 \/ st[1] # "none")
            /\ (~UniqueId \/ LiveSet = {})
           THEN [To(c, "awaiting_worker") EXCEPT !.neh = head]
           ELSE Refuse(c)
    [] c.k = "resend" -> Refuse(c)
    [] c.k = "worker_started" ->
         IF s = "awaiting_worker" THEN To(c, "working") ELSE Refuse(c)
    [] c.k = "cause_unknown" ->
         IF s = "working" THEN ToHelp(c) ELSE Refuse(c)
    [] c.k = "obs" ->
         IF ~c.f THEN Refuse(c)               \* stale: dropped by the classifier
         ELSE IF s = "working" THEN
                IF c.h = eh[c.w] THEN To(c, "proving")
                ELSE [To(c, "working") EXCEPT !.neh = c.h]
         ELSE IF s \in {"proving", "proven", "handing_off"} /\ c.h # eh[c.w] THEN
                [To(c, "proving") EXCEPT !.neh = c.h, !.nrq = -1]
         ELSE IF s = "confirming" THEN
                IF c.h = ph[c.w] THEN [To(c, "handed_off") EXCEPT !.nob = c.h]
                ELSE [To(c, "proving") EXCEPT !.neh = c.h, !.nrq = -1]
         ELSE Refuse(c)
    [] c.k = "proof_passed" ->
         IF s = "proving" /\ c.h = eh[c.w]
           THEN [To(c, "proven") EXCEPT !.nph = c.h, !.nrc = 0, !.nrcG = 0]
           ELSE Refuse(c)
    [] c.k = "proof_failed" ->
         IF s = "proving"
           THEN LET n == Min(rc[c.w] + 1, RetryN) IN
                IF n >= RetryN
                  THEN [ToHelp(c) EXCEPT !.nrc = n, !.nrcG = n]
                  ELSE [To(c, "working") EXCEPT !.nrc = n, !.nrcG = n]
           ELSE IF CountAdmissible THEN Refuse(c)
                ELSE [Refuse(c) EXCEPT !.nrc = Min(rc[c.w] + 1, RetryN)]
    [] c.k = "proof_posted" ->
         IF s = "proven" THEN [To(c, "handing_off") EXCEPT !.nrq = ph[c.w]]
         ELSE Refuse(c)
    [] c.k = "handoff_written" ->
         IF s = "handing_off" /\ c.h = rq[c.w]
           THEN IF Confirm THEN To(c, "confirming")
                ELSE To(c, "handed_off")
           ELSE Refuse(c)
    [] c.k = "handoff_failed" ->
         IF s = "handing_off" THEN ToHelp(c) ELSE Refuse(c)
    [] c.k = "deadline" ->
         IF s \in WorkerOwned THEN To(c, "blocked")
         ELSE IF s \in NodeOwned THEN ToHelp(c)
         ELSE IF s = "help"
           THEN IF tier[c.w] < TierTop
                  THEN [To(c, "help") EXCEPT !.ntier = tier[c.w] + 1]
                  ELSE To(c, "blocked")
         ELSE IF ~TermGuard /\ s \in Term THEN To(c, "blocked")
         ELSE Refuse(c)
    [] c.k = "help_answer" ->
         IF s \in Term \/ s = "none" THEN Refuse(c)
         ELSE IF s # "help" /\ AnswerGuard THEN Refuse(c)
         ELSE [CASE c.a = "found" -> To(c, "working")
                 [] c.a = "retry" -> To(c, org[c.w])
                 [] c.a = "stop" -> To(c, "blocked")
                 [] OTHER ->
                      IF tier[c.w] < TierTop
                        THEN [To(c, "help") EXCEPT !.ntier = tier[c.w] + 1]
                        ELSE To(c, "blocked")
               EXCEPT !.ans = TRUE]

(* A receipt's outputs: the step reply always; an effect command on entering an
   effect state; the terminal on entering a terminal state. *)
OutsOf(r) ==
  {"reply"} \cup (IF r.ok /\ r.ns \in EffectStates THEN {"effect"} ELSE {})
            \cup (IF r.ok /\ r.ns \in Term THEN {"term"} ELSE {})

(* ---- environment ---- *)
WorkerKind(c) == c.k \in {"worker_started", "cause_unknown"}
          \/ (c.k = "obs" /\ st[c.w] = "working")
Arrivals ==
  {[NoIn EXCEPT !.k = "start", !.w = w] : w \in W}
  \cup {[NoIn EXCEPT !.k = k, !.w = w] : k \in {"worker_started", "cause_unknown",
          "proof_posted", "proof_failed", "handoff_failed", "resend"}, w \in W}
  \cup {[NoIn EXCEPT !.k = k, !.w = w, !.h = h] : k \in {"proof_passed", "handoff_written"},
          w \in W, h \in 0..MaxHead}
  \cup {[NoIn EXCEPT !.k = "obs", !.w = w, !.h = h, !.f = f] : w \in W, h \in 0..MaxHead, f \in BOOLEAN}
  \cup {[NoIn EXCEPT !.k = "help_answer", !.w = w, !.a = a] : w \in W,
          a \in {"found", "retry", "stop", "failed"}}
ValidArrival(c) ==
  /\ (c.k = "start" => startsLeft > 0)
  /\ (c.k # "start" => st[c.w] # "none")
  /\ (gone[c.w] => ~WorkerKind(c))
  /\ (c.k = "obs" /\ c.f => c.h >= eh[c.w] /\ c.h <= head)

Take(c) ==
  /\ pc = "idle"
  /\ cur' = c /\ pc' = "fold" /\ rP' = FALSE
  /\ UNCHANGED <<wf, head, hmoves, crashes, startsLeft, inCnt, cm, rOuts, rW, pubd, effCnt, lost>>

Arrive ==       \* any input, bounded per state
  \E c \in Arrivals : ValidArrival(c) /\ inCnt < MaxIn /\ Take(c)

TickArrive ==   \* a runtime tick: deadline_passed for a live workflow
  \E w \in W : TickOn /\ (st[w] \in Live \/ (~TermGuard /\ st[w] \in Term)) /\
    Take([NoIn EXCEPT !.k = "deadline", !.w = w])

Forward(w) ==   \* the one input that moves workflow w forward, if its owner is alive
  LET s == st[w] IN
  CASE s = "awaiting_worker" /\ ~gone[w] -> {[NoIn EXCEPT !.k = "worker_started", !.w = w]}
    [] s = "working" /\ ~gone[w] ->
         {[NoIn EXCEPT !.k = "obs", !.w = w, !.h = head, !.f = TRUE]}
    [] s = "proving" -> {[NoIn EXCEPT !.k = "proof_passed", !.w = w, !.h = eh[w]]}
    [] s = "proven" -> {[NoIn EXCEPT !.k = "proof_posted", !.w = w]}
    [] s = "handing_off" -> {[NoIn EXCEPT !.k = "handoff_written", !.w = w, !.h = rq[w]]}
    [] s = "confirming" -> {[NoIn EXCEPT !.k = "obs", !.w = w, !.h = head, !.f = TRUE]}
    [] s = "help" -> {[NoIn EXCEPT !.k = "help_answer", !.w = w, !.a = "stop"]}
    [] OTHER -> {}
DriveArrive ==  \* fair: the worker, effects and observers do their part
  \/ \E w \in W : \E c \in Forward(w) : ValidArrival(c) /\ Take(c)
  \/ \E w \in W : st[w] = "none" /\ startsLeft > 0 /\ (w = 1 \/ st[1] # "none")
       /\ Take([NoIn EXCEPT !.k = "start", !.w = w])

HeadMove ==
  /\ hmoves < MaxHead /\ head' = head + 1 /\ hmoves' = hmoves + 1
  /\ UNCHANGED <<wf, crashes, startsLeft, inCnt, pipe>>

Vanish ==
  /\ \E w \in W : st[w] \in WorkerOwned /\ ~gone[w] /\ gone' = [gone EXCEPT ![w] = TRUE]
  /\ UNCHANGED <<st, eh, ph, rq, ob, rc, rcG, hb, tier, org, termPub, ansIn,
                 head, hmoves, crashes, startsLeft, inCnt, pipe>>

\* A terminal workflow keeps only what the properties read (proven head, handoff, observation).
(* ---- the pipeline: fold, receipt, outputs, commit; crash; recovery ---- *)
Receipt ==      \* fold applied and the receipt published: the commit point
  /\ pc = "fold" /\ ~rP
  /\ LET r == Res(cur) w == cur.w IN
     /\ st' = [st EXCEPT ![w] = r.ns] /\ eh' = [eh EXCEPT ![w] = IF r.ns \in Term THEN 0 ELSE r.neh]
     /\ ph' = [ph EXCEPT ![w] = IF r.ns = "blocked" THEN -1 ELSE r.nph] /\ rq' = [rq EXCEPT ![w] = IF r.ns = "blocked" THEN -1 ELSE r.nrq]
     /\ ob' = [ob EXCEPT ![w] = IF r.ns = "blocked" THEN -1 ELSE r.nob] /\ rc' = [rc EXCEPT ![w] = IF r.ns \in Term THEN 0 ELSE r.nrc]
     /\ rcG' = [rcG EXCEPT ![w] = IF r.ns \in Term THEN 0 ELSE r.nrcG] /\ hb' = [hb EXCEPT ![w] = IF r.ns \in Term THEN 0 ELSE r.nhb]
     /\ tier' = [tier EXCEPT ![w] = r.ntier] /\ org' = [org EXCEPT ![w] = IF r.ns \in Term THEN "working" ELSE r.norg]
     /\ ansIn' = IF r.ans THEN ansIn \cup {st[w]} ELSE ansIn
     /\ rOuts' = OutsOf(r) /\ rW' = w
     /\ inCnt' = IF r.ns # st[w] THEN 0 ELSE inCnt + 1
  /\ startsLeft' = IF cur.k = "start" THEN startsLeft - 1 ELSE startsLeft
  /\ rP' = TRUE /\ pc' = "pub" /\ cm' = FALSE /\ pubd' = {} /\ effCnt' = 0
  /\ cur' = NoIn
  /\ UNCHANGED <<gone, termPub, head, hmoves, crashes, lost>>

PubOut ==
  /\ pc = "pub" /\ \E o \in rOuts \ pubd :
       /\ pubd' = pubd \cup {o}
       /\ effCnt' = IF o = "effect" THEN effCnt + 1 ELSE effCnt
       /\ termPub' = IF o = "term" THEN [termPub EXCEPT ![rW] = @ + 1] ELSE termPub
  /\ UNCHANGED <<st, eh, ph, rq, ob, rc, rcG, hb, tier, org, gone, ansIn,
                 head, hmoves, crashes, startsLeft, inCnt, pc, cur, rP, cm, rOuts, rW, lost>>

Commit ==       \* the input's offset
  /\ pc = "pub" /\ ~cm /\ (~CommitAfter \/ rOuts \subseteq pubd)
  /\ cm' = TRUE
  /\ UNCHANGED <<wf, head, hmoves, crashes, startsLeft, inCnt, pc, cur, rP, rOuts, rW, pubd, effCnt, lost>>

Finish ==
  /\ pc = "pub" /\ cm /\ rOuts \subseteq pubd
  /\ pc' = "idle" /\ cur' = NoIn /\ rP' = FALSE /\ cm' = FALSE /\ rOuts' = {} /\ pubd' = {}
  /\ UNCHANGED <<wf, head, hmoves, crashes, startsLeft, inCnt, rW, effCnt, lost>>

Crash ==        \* volatile state lost between any two steps
  /\ crashes < MaxCrash /\ pc # "down"
  /\ pc' = "down" /\ crashes' = crashes + 1
  /\ UNCHANGED <<wf, head, hmoves, startsLeft, inCnt, cur, rP, cm, rOuts, rW, pubd, effCnt, lost>>

Recover ==      \* section 3.6: read the topics back, replay the uncommitted input
  /\ pc = "down"
  /\ IF cm THEN /\ lost' = (lost \/ ~(rOuts \subseteq pubd))     \* offset committed: no replay
                /\ pc' = "idle" /\ cur' = NoIn /\ rP' = FALSE /\ cm' = FALSE
                /\ rOuts' = {} /\ pubd' = {}
     ELSE IF rP THEN /\ pc' = "pub"
                     /\ pubd' = IF Readback THEN pubd ELSE pubd \ {"effect"}
                     /\ UNCHANGED <<cur, rP, cm, rOuts, lost>>
     ELSE IF cur = NoIn THEN /\ pc' = "idle" /\ UNCHANGED <<cur, rP, cm, rOuts, pubd, lost>>
     ELSE /\ pc' = "fold" /\ UNCHANGED <<cur, rP, cm, rOuts, pubd, lost>>
  /\ UNCHANGED <<wf, head, hmoves, crashes, startsLeft, inCnt, rW, effCnt>>

Next == Arrive \/ TickArrive \/ DriveArrive \/ HeadMove \/ Vanish
        \/ Receipt \/ PubOut \/ Commit \/ Finish \/ Crash \/ Recover

Spec == Init /\ [][Next]_vars
        /\ SF_vars(DriveArrive) /\ SF_vars(TickArrive)
        /\ WF_vars(Receipt) /\ WF_vars(PubOut) /\ WF_vars(Commit)
        /\ WF_vars(Finish) /\ WF_vars(Recover)

(* ---- properties ---- *)
OneTerminal == \A w \in W : termPub[w] <= 1
OneLiveWorkflowPerChange == Cardinality(LiveSet) <= 1
HandoffAtProvenHead ==
  \A w \in W : st[w] = "handed_off" => (ph[w] >= 0 /\ rq[w] = ph[w] /\ ob[w] = ph[w])
NoLostOutput == ~lost
NoDuplicateEffect == effCnt <= 1
RetryBound == \A w \in W : rc[w] = rcG[w] /\ rc[w] <= RetryN
AnswerInContract == ansIn \subseteq {"help"}
DeadlineFires == \A w \in W : gone[w] ~> (st[w] \in Term)
Terminates == <>[](startsLeft = 0 /\ pc = "idle" /\ \A w \in W : st[w] \in Live => gone[w])
=============================================================================
