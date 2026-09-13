# Cloud pre-push capacity host (OMN-16634)

A single AWS EC2 instance that serves as an **overflow / off-LAN capacity
target** for the governed pre-push picker (`scripts/hooks/prepush_dispatch.sh`,
OMN-16991). It joins the existing host table as the `hcloud` row — it does not
add any new dispatch seam, and it must never displace lab-first placement: its
default state is **stopped**, and a stopped host probes `unreachable` and is
skipped by the picker with fail-closed semantics untouched.

## Where the instance identity lives

**Not here.** The concrete instance id, address, security-group id and account
are held in the private placement overlay, exactly as the `hcloud` row in
`scripts/hooks/prepush_hosts.tsv` already states. This repository is public;
everything in this directory is written to take the host as an argument so
that no deployment identity is ever committed.

Read the identity from the overlay and export it once per session:

```bash
CLOUD_SSH_TARGET="<user>@<address-from-the-private-placement-overlay>"
CLOUD_INSTANCE_ID="<instance-id-from-the-private-placement-overlay>"
```

Shape of the instance, which is not identifying and is worth recording: a
16-vCPU / 32-GiB compute instance, **on-demand**, 100 GiB gp3 root volume with
`DeleteOnTermination=true`, `InstanceInitiatedShutdownBehavior=stop`, and a
security group permitting inbound TCP/22 only from allowlisted CIDRs.

**On-demand, not spot — deliberate.** Stop-on-idle is the cost mechanism and a
one-time spot instance *terminates* on OS shutdown, destroying the warm caches;
a spot interruption also kills a governed suite mid-run. At the observed volume
(~20–40 heavy escalations/month, ≲1 h each) the spot saving is a few dollars
and the failure modes are not worth it.

## Connectivity model (three legs, no tailnet, no VPN)

1. **Dispatch transport** — plain SSH to the instance address. Pushing machines
   connect *out* to this host; nothing connects inward to the lab network. The
   security group is locked to the lab egress address plus individually
   authorized collaborator addresses.
2. **Test-time dependencies** — suites run **hermetically** on this host. The
   remote leg runs `pytest … --ignore=tests/integration`; the in-memory local
   bus is the house default. A test that reaches for a lab service on the
   private LAN from here is a hermeticity **defect to ticket**, never a reason
   to punch a network path.
3. **Platform participation** — any platform-facing traffic (event emission,
   receipts, the future OMN-17226 per-host runner EFFECT) goes through the
   platform **gateway** as a verified external actor per the
   unified-external-ingress doctrine. This host is not a bus-attached LAN peer
   and never speaks Kafka directly.

## Provisioning (committed, idempotent)

Everything on the host comes from `bootstrap.sh` in this directory — no
hand-built installs. From a lab host, with `CLOUD_SSH_TARGET` set as above:

```bash
ssh "$CLOUD_SSH_TARGET" 'mkdir -p /tmp/prepush-cloud'
scp deploy/prepush-cloud/bootstrap.sh \
    deploy/prepush-cloud/onex-prepush-idle-stop.sh \
    deploy/prepush-cloud/onex-prepush-idle-stop.service \
    deploy/prepush-cloud/onex-prepush-idle-stop.timer \
    "$CLOUD_SSH_TARGET:/tmp/prepush-cloud/"
ssh "$CLOUD_SSH_TARGET" 'bash /tmp/prepush-cloud/bootstrap.sh'
deploy/prepush-cloud/warm_cache.sh "$CLOUD_SSH_TARGET" --prove
```

Deliberately absent from the host: **docker** (unit+ci suites are hermetic;
integration is excluded by the remote leg), **tailscale** (operator
directive), **GitHub credentials** (bundle transport only), **AWS
credentials** (idle-stop is an OS halt; no IAM role attached).

The bootstrap also makes the box a **runtime-capable ONEX target** (git +
python3.12 + uv): the OMN-17226 per-host runner EFFECT node can execute here
with no further provisioning. This script family is the named provisioning
seam that the OMN-17226 runner manifest retires.

`bootstrap.sh` substitutes the installing user's workroot into the idle-stop
unit at install time. The committed unit file carries the
`@ONEX_PREPUSH_WORKROOT@` placeholder and the script fails fast if the
environment variable is unset, so no machine-specific path is checked in and a
mis-installed unit cannot silently read "no activity".

## Lifecycle / cost controls

* **Idle auto-stop**: `onex-prepush-idle-stop.timer` (10-min cadence) stops
  the instance after two consecutive idle observations (~20–30 min idle).
  Idle = no heavy-suite LOCK, no pytest/`uv sync`/wrapper process, no
  interactive session, no run-dir activity in 30 min, uptime > 30 min.
  Fail-ACTIVE: an unreadable signal keeps the box up (never kill a governed
  run to save cents).
* **Restart on demand** (lab side, needs AWS credentials):
  ```bash
  aws ec2 start-instances --instance-ids "$CLOUD_INSTANCE_ID"
  # ~40s to SSH-reachable; the address is static so the host table needs no edit
  ```
* **Picker behavior while stopped**: the 3-second SSH probe fails, the row
  logs `hcloud=unreachable`, and placement falls through to the lab rows.
  Degradation is graceful and fail-closed semantics are untouched — this is
  also what preserves lab-first placement: the cloud row only competes when
  someone has deliberately started it.
* **Cost at observed volume**: ~20–40 runs/month × ≲1 h × $0.8211/h ≈
  **$16–33/month compute** + $8/month EBS (100 GiB gp3) + ~$3.6/month static
  address while stopped ⇒ **≈ $28–45/month worst case, ≪ that when idle-stop
  keeps runs batched**; a stopped month costs ~$12.

## Collaborator access

An authorized collaborator's public key is installed for the same user the
picker dispatches as. Interactive use is an ordinary `ssh "$CLOUD_SSH_TARGET"`;
a governed pre-push needs nothing special, because once their checkout carries
the `hcloud` table row their hook's picker probes and dispatches to this host
exactly as it does for lab hosts.

Collaborators who reach the lab over the tailnet do not reach this host that
way — it deliberately does not join it — so their public egress address must be
added to the security group once:

```bash
aws ec2 authorize-security-group-ingress \
  --group-id "<security-group-id-from-the-private-placement-overlay>" \
  --ip-permissions 'IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=<address>/32,Description=<label>}]'
```

## Host-table row

See `scripts/hooks/prepush_hosts.tsv` (`hcloud`). The row is memory+load
probed like every other row; ranking is unchanged (ascending load ratio among
slot-free hosts). Per-repo tables in `omnibase_core` and `omnimarket` carry
the row only with that repo's own transport proof, per the OMN-17159/OMN-17435
discipline.
