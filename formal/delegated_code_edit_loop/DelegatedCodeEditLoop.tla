---------------------------- MODULE DelegatedCodeEditLoop ----------------------------
(* The delegated code edit loop (OMN-20290): two runners race on one correlation
   id; each turn may write an allowed or a forbidden path, run a declared or an
   undeclared check, or finish. Properties:
     OneReceipt        at most one loop receipt per correlation id
     NoForbiddenWrite  a path outside the writable globs is never written
     NoUndeclaredCheck an undeclared check never runs
     TurnBound         a runner never takes more than MaxTurns turns
     Terminates        every runner that claimed ends in a terminal status
   Mutations (cfg constants) that must break them:
     AtomicClaim=FALSE  claim is check-then-set in two steps    -> OneReceipt
     GlobGuard=FALSE    writes skip the writable-glob check     -> NoForbiddenWrite
     CheckGuard=FALSE   run_check skips the declared-name check -> NoUndeclaredCheck
     CapGuard=FALSE     the turn loop has no cap                -> TurnBound *)
EXTENDS Naturals, FiniteSets, Sequences
CONSTANTS Runners, MaxTurns, AtomicClaim, GlobGuard, CheckGuard, CapGuard
Terminal == {"accepted", "no_progress", "budget_exhausted", "refused"}
VARIABLES pc, turns, claimed, receipts, written, ran, lastFail
vars == <<pc, turns, claimed, receipts, written, ran, lastFail>>

Init ==
  /\ pc = [r \in Runners |-> "start"]
  /\ turns = [r \in Runners |-> 0]
  /\ claimed = FALSE
  /\ receipts = 0
  /\ written = {}
  /\ ran = {}
  /\ lastFail = [r \in Runners |-> "none"]

\* Atomic claim: exclusive create.
ClaimAtomic(r) ==
  /\ AtomicClaim /\ pc[r] = "start"
  /\ IF claimed THEN pc' = [pc EXCEPT ![r] = "refused"]
               ELSE /\ pc' = [pc EXCEPT ![r] = "turn"]
  /\ claimed' = TRUE
  /\ UNCHANGED <<turns, receipts, written, ran, lastFail>>

\* Mutation: check, then set in a later step.
ClaimCheck(r) ==
  /\ ~AtomicClaim /\ pc[r] = "start"
  /\ IF claimed THEN pc' = [pc EXCEPT ![r] = "refused"]
               ELSE pc' = [pc EXCEPT ![r] = "claiming"]
  /\ UNCHANGED <<turns, claimed, receipts, written, ran, lastFail>>
ClaimSet(r) ==
  /\ pc[r] = "claiming"
  /\ claimed' = TRUE /\ pc' = [pc EXCEPT ![r] = "turn"]
  /\ UNCHANGED <<turns, receipts, written, ran, lastFail>>

CanTurn(r) == pc[r] = "turn" /\ (~CapGuard \/ turns[r] < MaxTurns)

Write(r, path) ==
  /\ CanTurn(r) /\ turns[r] < MaxTurns + 1
  /\ turns' = [turns EXCEPT ![r] = @ + 1]
  /\ written' = IF path = "allowed" \/ ~GlobGuard THEN written \cup {path} ELSE written
  /\ UNCHANGED <<pc, claimed, receipts, ran, lastFail>>

RunCheck(r, name) ==
  /\ CanTurn(r) /\ turns[r] < MaxTurns + 1
  /\ turns' = [turns EXCEPT ![r] = @ + 1]
  /\ ran' = IF name = "declared" \/ ~CheckGuard THEN ran \cup {name} ELSE ran
  /\ UNCHANGED <<pc, claimed, receipts, written, lastFail>>

\* finish runs the declared checks; they pass or fail with a fingerprint.
Finish(r, outcome) ==
  /\ CanTurn(r) /\ turns[r] < MaxTurns + 1
  /\ turns' = [turns EXCEPT ![r] = @ + 1]
  /\ ran' = ran \cup {"declared"}
  /\ IF outcome = "pass" THEN /\ pc' = [pc EXCEPT ![r] = "accepted"]
                              /\ lastFail' = lastFail
     ELSE IF lastFail[r] = outcome THEN /\ pc' = [pc EXCEPT ![r] = "no_progress"]
                                         /\ lastFail' = lastFail
     ELSE /\ pc' = pc /\ lastFail' = [lastFail EXCEPT ![r] = outcome]
  /\ UNCHANGED <<claimed, receipts, written>>

Cap(r) ==
  /\ CapGuard /\ pc[r] = "turn" /\ turns[r] >= MaxTurns
  /\ pc' = [pc EXCEPT ![r] = "budget_exhausted"]
  /\ UNCHANGED <<turns, claimed, receipts, written, ran, lastFail>>

\* The receipt is written once, by the runner that ends after claiming.
Receipt(r) ==
  /\ pc[r] \in {"accepted", "no_progress", "budget_exhausted"}
  /\ receipts' = receipts + 1
  /\ pc' = [pc EXCEPT ![r] = "done_" \o pc[r]]
  /\ UNCHANGED <<turns, claimed, written, ran, lastFail>>

RunnerNext(r) ==
  \/ ClaimAtomic(r) \/ ClaimCheck(r) \/ ClaimSet(r) \/ Cap(r) \/ Receipt(r)
  \/ \E p \in {"allowed", "forbidden"}: Write(r, p)
  \/ \E n \in {"declared", "undeclared"}: RunCheck(r, n)
  \/ \E o \in {"pass", "failA", "failB"}: Finish(r, o)

Next == \E r \in Runners: RunnerNext(r)

\* Each runner keeps taking steps: every turn is one delegate call that returns.
Spec == Init /\ [][Next]_vars /\ \A r \in Runners: WF_vars(RunnerNext(r))

OneReceipt == receipts <= 1
NoForbiddenWrite == "forbidden" \notin written
NoUndeclaredCheck == "undeclared" \notin ran
TurnBound == \A r \in Runners: turns[r] <= MaxTurns
Done(r) == pc[r] \in {"refused", "done_accepted", "done_no_progress", "done_budget_exhausted"}
Terminates == \A r \in Runners: <>Done(r)
=======================================================================================
