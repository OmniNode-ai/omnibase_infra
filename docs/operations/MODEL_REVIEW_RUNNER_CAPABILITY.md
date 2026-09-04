# Model-review runner capability contract

OMN-17855 records the source-controlled contract for a runner that may execute
the hostile-review workload. It does not provision a runner, apply a label,
resolve a credential, resolve an endpoint, or make a network request.

`config/runner_fleet.yaml` keeps the `model_review` capability inactive. The
three reference fields there are opaque logical identifiers, not secret values,
endpoint addresses, or environment-variable names. A separate, authorized
control-plane rollout must supply the corresponding overlay and facts.

The pure preflight accepts only non-sensitive facts: runner labels, the set of
present reference identifiers, and the set of healthy assertion identifiers. A
runner is eligible only when all of the following are true:

- the capability record is active;
- the runner has the `model-review` label;
- every required reference is present; and
- the configured health assertion is present and healthy.

An absent capability record, missing/unknown facts, or an unhealthy assertion
return a non-ready verdict. The
preflight never treats a generic `omnibase-ci` runner as model-review capable.
Enabling the record is intentionally out of scope for this contract and must
follow a separately authorized overlay rollout and same-repository/fork canary.
