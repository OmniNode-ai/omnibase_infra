# Public Surface Uptime Alerting — Runbook

**Ticket:** OMN-13751
**Severity scope:** `PublicSurfaceDown` (critical), `PublicSurfaceSlowResponse` (warning),
`OnexPodCrashLoopBackOff` (critical), `OnexPodImagePullBackOff` (critical)

---

## Why this exists

The 2026-06-29 omniweb outage (HTTP 502) was latent for approximately 34 days before
an operator noticed it manually. No alert fired. This runbook documents the alerting
stack added in OMN-13751 to close that gap.

---

## Alert conditions

### PublicSurfaceDown

**Fires when:** the Blackbox Exporter HTTP probe returns non-2xx for ≥ 3 consecutive
30-second intervals (~90 seconds) for any of:

- `https://omninode.ai/`
- `https://www.omninode.ai/`
- `https://app.omninode.ai/`
- `https://api.omninode.ai/health`

**Triage steps:**

1. Check the surface directly: `curl -o /dev/null -s -w "%{http_code}" <URL>`
2. Check omniweb logs on the AWS cluster:
   `kubectl -n onex-dev logs -l app=omniweb --tail=100`
3. Check the ALB/ingress for upstream errors.
4. If the surface is a `502 Bad Gateway`, the backend pod is likely unhealthy —
   check for CrashLoopBackOff or ImagePullBackOff (see below).

**Silence:** do NOT silence this alert without first confirming the surface is live.
A silence during an outage hides the problem from the next operator.

### PublicSurfaceSlowResponse

**Fires when:** probe duration > 5 seconds for ≥ 3 minutes on any probed surface.

**Triage steps:**

1. Check ALB latency metrics in the AWS console.
2. Check CPU/memory on the backing pods: `kubectl -n onex-dev top pods`
3. If the backing service is a container, review recent deploys:
   `kubectl -n onex-dev rollout history deployment/<name>`

### OnexPodCrashLoopBackOff

**Fires when:** a container in namespace `onex-dev` has been in `CrashLoopBackOff`
state for ≥ 5 minutes.

**Triage steps:**

```bash
# Get logs from the crashing container
kubectl -n onex-dev logs <pod-name> -c <container-name> --previous

# Check pod events for OOM / startup failures
kubectl -n onex-dev describe pod <pod-name>

# Check if it is an entrypoint / init-container issue
kubectl -n onex-dev get pod <pod-name> -o jsonpath='{.status.initContainerStatuses}'
```

If the crash is the runtime entrypoint stamp failure (schema fingerprint):
see `docs/runbooks/apply-migrations.md` and the OMN-13666 fix.

### OnexPodImagePullBackOff

**Fires when:** a container in namespace `onex-dev` cannot pull its image for ≥ 5 minutes.

**Triage steps:**

```bash
# Check which image is failing and why
kubectl -n onex-dev describe pod <pod-name>

# Verify ECR credentials / imagePullSecrets
kubectl -n onex-dev get secret -o name | grep ecr

# Force a re-pull by deleting and letting the deployment recreate
kubectl -n onex-dev rollout restart deployment/<deployment-name>
```

---

## Activation sequence

### Part A — Prometheus uptime checking (Docker / .201)

The Blackbox Exporter + Alertmanager services must be added to the observability
Docker Compose profile on the .201 server.

1. **Set required environment variables** (via Infisical / `.omnibase/.env`):

   ```bash
   ALERTMANAGER_SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
   ALERTMANAGER_SLACK_CHANNEL=#alerts-platform
   ```

2. **Add Blackbox Exporter and Alertmanager** to the observability Docker Compose
   profile in `docker/docker-compose.infra.yml`:

   ```yaml
   # Under the observability profile:
   blackbox-exporter:
     image: prom/blackbox-exporter:latest
     profiles: [observability]
     volumes:
       - ./observability/prometheus/blackbox.yml:/etc/blackbox_exporter/config.yml:ro
     ports:
       - "9115:9115"

   alertmanager:
     image: prom/alertmanager:latest
     profiles: [observability]
     command:
       - --config.file=/etc/alertmanager/alertmanager.yml
       - --config.expand-env=true
     volumes:
       - ./observability/prometheus/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
     environment:
       ALERTMANAGER_SLACK_WEBHOOK_URL: "${ALERTMANAGER_SLACK_WEBHOOK_URL}"
       ALERTMANAGER_SLACK_CHANNEL: "${ALERTMANAGER_SLACK_CHANNEL}"
     ports:
       - "9093:9093"
   ```

3. **Start the observability profile**:

   ```bash
   docker compose -f docker/docker-compose.infra.yml --profile observability up -d
   ```

4. **Verify** Prometheus has picked up the uptime-check targets:
   Open `http://<.201-host>:9090/targets` and confirm `uptime-check` shows 4 targets.

5. **Verify alerts are evaluable**:
   Open `http://<.201-host>:9090/alerts` and confirm `PublicSurfaceDown` and
   `PublicSurfaceSlowResponse` appear in the list.

### Part B — k8s pod crash alerting (onex-dev namespace)

**Option 1 — PrometheusRule CRD** (if kube-prometheus-stack is installed):

```bash
kubectl apply -f observability/k8s/alerting/pod-crash-prometheusrule.yaml

# Verify rule pickup (~2 m after apply)
kubectl -n onex-dev get prometheusrule onex-pod-crash-alerts
```

**Option 2 — CronJob** (portable fallback):

```bash
# 1. Create the Slack webhook secret
kubectl -n onex-dev create secret generic alerting-credentials \
  --from-literal=slack_webhook_url=https://hooks.slack.com/services/...

# 2. Apply all manifests (ServiceAccount, Role, RoleBinding, CronJob)
kubectl apply -f observability/k8s/alerting/pod-crash-cronjob.yaml

# 3. Test immediately (bypasses the 5-minute schedule)
kubectl -n onex-dev create job --from=cronjob/pod-crash-alerter pod-crash-manual-test
kubectl -n onex-dev logs -l job-name=pod-crash-manual-test
```

---

## Test procedure (DoD verification)

To verify a deliberately-broken surface fires the alert and clean state is quiet:

```bash
# Temporarily point a test target at an unreachable URL by manually checking
# the Prometheus expression browser for: probe_success{instance="https://omninode.ai/"}
# If the probe is working, the value should be 1. Block the URL at the firewall
# or use a test target that returns 503 to verify the alert fires after 90 s.

# Confirm clean state is quiet (all probes green):
curl -s http://localhost:9090/api/v1/query?query=probe_success | \
  python3 -c "import sys,json; d=json.load(sys.stdin); [print(r) for r in d['data']['result']]"
```

---

## Related

- `observability/prometheus/rules/public-surface-uptime.yml` — Alert rule definitions
- `observability/prometheus/alertmanager.yml` — Alertmanager Slack routing config
- `observability/k8s/alerting/pod-crash-prometheusrule.yaml` — PrometheusRule CRD
- `observability/k8s/alerting/pod-crash-cronjob.yaml` — CronJob fallback
- `docs/runbooks/runtime-crash-loop.md` — Runtime container crash-loop runbook
