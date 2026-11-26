# The deploy agent's lane lag bound and probe-deferral cap

Moved from the module docstrings of `deploy_agent/idle_converge.py` and
`deploy_agent/routing.py` (OMN-21146, OMN-26703).

## A lagging lane has a declared bound, and is loud (OMN-21146)

The probe windows above cost more than they protect. C15 runs every two hours
for a declared 65 minutes, so with the 40-minute lead each run excludes 105
minutes of every 120, and a merge that lands in one waits for the 15 minutes
between. On 2026-10-15 omnimarket#3964 merged at 17:07Z, inside the 17:41Z
run's exclusion; the default instance's agent accepted its converge at 18:46:52Z, the
exclusion's end, and dod-verify refused merged_pr fleet-wide on the old image
until the lane was rebuilt, with nothing saying the lane was behind.

So an instance may declare ``flags.max_dev_lag_minutes`` in the routing table
(``ModelLaneFlags.max_dev_lag``). The agent records when it first saw the lane
behind a followed repository's dev head (:func:`next_behind_since`); once that
is as old as the bound the lane is OVERDUE, and :func:`decide` no longer
waits out the idle period (``recently_active``) or an unreadable probe schedule.
It still never starts inside a reserved window of the guarded M4 probes C15
(the chain canary) and C16. OMN-19175: deploys inside those windows made C16
and the chain canary reach their terminals after their probe budgets. Every
other bound holds: no job in flight, the FROZEN flag, the load gate, and one
converge per head. And the lag is loud:
:func:`dev_lane_drift` is the ``ModelDevLaneDrift`` the agent publishes and
serves at ``GET /dev-lane-drift``. An instance with no bound is unchanged.

## The probe-deferral cap (OMN-26703)

The lag bound overrides the idle wait and an unreadable schedule, never the
guarded C15 and C16 windows. Those chain: omnimarket 8a5a95e5 merged at 11:49Z
on 2026-11-26 and waited until 13:00Z. `flags.max_probe_deferral_minutes`
(`ModelLaneFlags.max_probe_deferral`, 45 on both dev instances in
`config/deploy_lane_routing.yaml`) is the age, measured from when the agent
first saw the lane behind, past which `decide` no longer refuses
`probe_window`. A job in flight, the FROZEN flag, an unread head or running
revision and one converge per head still refuse. An instance with no cap is
unchanged. The decision row names the overridden window.

## Routing: residual, stated

A sha the clone cannot resolve even after a fetch takes the default, with a
warning line. Refusing would stall the one dev-lane agent on every such command
today; the cost, once a second instance is live, is that an instance that can
read the table and one that cannot may disagree. Both fetch the same origin, so
this needs a fetch failure on one host at the moment of the command.
