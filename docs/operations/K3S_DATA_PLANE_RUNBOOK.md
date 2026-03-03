# K3s Data-Plane Runbook

**Ticket**: OMN-3488
**Host**: 192.168.86.201 (Linux GPU server, Ubuntu 24.04.3 LTS)
**Prerequisite for**: OMN-3478 (cloud tunnel script rewrite)
**Last updated**: 2026-03-03

---

## Overview

This runbook covers the single-node k3s cluster on `192.168.86.201` that hosts the cloud data-plane bus. The cloud bus exposes Redpanda via NodePort so the Mac can reach it through an SSH tunnel.

```text
Mac (localhost:29092)
  ↕ SSH tunnel (launchd: ai.omninode.redpanda-tunnel)
192.168.86.201:29092  (NodePort → omninode-redpanda-external svc)
  ↕ k3s kube-proxy
data-plane/omninode-redpanda pod  (Redpanda external listener, port 9093)
```

**Bus selection**: this is the `cloud` bus. Activate with `bus-cloud` in your shell.
See `~/.claude/CLAUDE.md` "Bus Selection Policy" for the full two-bus architecture.

---

## Installation (first-time / OMN-3488)

### Prerequisites

- SSH key access: `ssh jonah@192.168.86.201`
- sudo rights on the remote host
- Manifests in `k8s/data-plane/` (checked into omnibase_infra)

### Automated (recommended)

```bash
# From your Mac, in the omnibase_infra worktree:
./scripts/provision-k3s-data-plane.sh

# With explicit args:
./scripts/provision-k3s-data-plane.sh --host 192.168.86.201 --user jonah

# Dry run (shows what would happen):
./scripts/provision-k3s-data-plane.sh --dry-run
```

### Manual steps

1. **Install k3s**:
   ```bash
   ssh jonah@192.168.86.201
   # Configure NodePort range to allow 29092 (below k3s default 30000-32767):
   sudo mkdir -p /etc/rancher/k3s
   sudo tee /etc/rancher/k3s/config.yaml <<'EOF'
   kube-apiserver-arg:
     - service-node-port-range=1024-65535
   EOF
   # Install with pinned version — INSTALL_K3S_VERSION must be set for sh (right of pipe):
   curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=v1.31.4+k3s1 sh -
   sudo systemctl enable k3s --now
   sudo systemctl status k3s
   ```

2. **Record bind address** (for OMN-3478):
   ```bash
   sudo ss -lntp | grep ':6443'
   # Expected: LISTEN 0 4096 *:6443 → K3S_BIND_ADDR=192.168.86.201
   ```

3. **Apply manifests**:
   ```bash
   # Copy from Mac:
   scp k8s/data-plane/*.yaml jonah@192.168.86.201:/tmp/omninode-data-plane-manifests/

   # On the remote host:
   sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl apply -f /tmp/omninode-data-plane-manifests/
   ```

4. **Wait for Redpanda**:
   ```bash
   sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl -n data-plane rollout status statefulset/omninode-redpanda --timeout=180s
   ```

---

## Verification

### Service health

```bash
# k3s running?
ssh jonah@192.168.86.201 sudo systemctl is-active k3s

# Namespace exists?
ssh jonah@192.168.86.201 sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl get ns data-plane

# Services present?
ssh jonah@192.168.86.201 sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl -n data-plane get svc

# Pod running?
ssh jonah@192.168.86.201 sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl -n data-plane get pods

# NodePort listening?
ssh jonah@192.168.86.201 sudo ss -lntp | grep ':29092'
```

### Tunnel reachability (from Mac)

```bash
# Activate cloud bus + verify tunnel
bus-cloud
kcat -L -b localhost:29092 -t test 2>&1 | head -5
```

If the tunnel daemon is not running, start it:
```bash
launchctl start ai.omninode.redpanda-tunnel
cat /tmp/omninode-redpanda-tunnel.log
```

---

## Operations

### View Redpanda logs

```bash
ssh jonah@192.168.86.201 \
  "sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl -n data-plane logs statefulset/omninode-redpanda --tail=100"
```

### Restart Redpanda pod

```bash
ssh jonah@192.168.86.201 \
  "sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl -n data-plane rollout restart statefulset/omninode-redpanda"
```

### List topics

```bash
# Via Redpanda admin API (NodePort 30644):
ssh jonah@192.168.86.201 \
  "sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl -n data-plane exec statefulset/omninode-redpanda -- rpk topic list"
```

### Create topics

```bash
ssh jonah@192.168.86.201 \
  "sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl -n data-plane exec statefulset/omninode-redpanda -- \
   rpk topic create <topic-name> --partitions 3 --replicas 1"
```

### Update manifests (apply changes)

```bash
# Re-apply after manifest changes
scp k8s/data-plane/*.yaml jonah@192.168.86.201:/tmp/omninode-data-plane-manifests/
ssh jonah@192.168.86.201 \
  "sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl apply -f /tmp/omninode-data-plane-manifests/"
```

---

## K3s Management

### k3s kubeconfig location on host

```text
/etc/rancher/k3s/k3s.yaml
```

Always prefix kubectl commands with `sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml` on the remote host.

### Copy kubeconfig to Mac (for OMN-3478 tunnel)

After OMN-3478 tunnel is active (`bus-cloud`), you can use a local kubeconfig:

```bash
# On Mac — retrieve kubeconfig (already points to 127.0.0.1:6443 for tunnel use)
ssh jonah@192.168.86.201 "sudo cat /etc/rancher/k3s/k3s.yaml" > ~/.kube/omninode-mvp1

chmod 600 ~/.kube/omninode-mvp1
export KUBECONFIG=~/.kube/omninode-mvp1
kubectl get nodes
```

Note: `~/.kube/omninode-mvp1` already points to `https://127.0.0.1:6443` (SSH tunnel endpoint). The tunnel forwards Mac:6443 → 192.168.86.201:6443.

### Upgrade k3s

```bash
# Pin version in provision-k3s-data-plane.sh → K3S_VERSION var, then re-run:
# INSTALL_K3S_VERSION must be set for sh (right of pipe), not for curl:
ssh jonah@192.168.86.201 \
  "curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=<new-version> sh -"
```

### View k3s service logs

```bash
ssh jonah@192.168.86.201 sudo journalctl -u k3s -n 100 -f
```

---

## Architecture Notes

### Why k3s (not Docker Compose)?

- The cloud bus requires a stable `NodePort` that survives container restarts.
- Kubernetes `NodePort` services provide a fixed port on the host (29092) regardless of pod restarts.
- The SSH tunnel (OMN-3478) targets this fixed NodePort.
- k3s is the lightest single-node Kubernetes distribution — minimal overhead on a GPU host.

### Why port 29092?

- Convention: local Docker bus = `19092`, cloud bus = `29092`.
- The Mac `~/.zshrc` `bus-cloud` function sets `KAFKA_BOOTSTRAP_SERVERS=localhost:29092`.
- The launchd tunnel daemon (`ai.omninode.redpanda-tunnel`) forwards Mac:29092 → host:29092.

### NodePort range

k3s default NodePort range: `30000–32767`.
Port 29092 is **below** the default range, so k3s must be configured to allow it:

```bash
# Verify k3s was started with --service-node-port-range flag:
ssh jonah@192.168.86.201 sudo systemctl cat k3s | grep service-node-port-range

# If missing, add to /etc/rancher/k3s/config.yaml:
ssh jonah@192.168.86.201 "sudo tee -a /etc/rancher/k3s/config.yaml <<'EOF'
kube-apiserver-arg:
  - service-node-port-range=1024-65535
EOF
sudo systemctl restart k3s"
```

See "NodePort range fix" section below for details.

---

## NodePort Range Fix

k3s default NodePort range is `30000–32767`. Port 29092 falls below this.

The provision script and manifests account for this by configuring k3s with an extended range. If you install k3s manually, you must set this:

```bash
# Create k3s config before installing (or restart k3s after):
ssh jonah@192.168.86.201 "sudo mkdir -p /etc/rancher/k3s && sudo tee /etc/rancher/k3s/config.yaml <<'EOF'
kube-apiserver-arg:
  - service-node-port-range=1024-65535
EOF"
```

Then install / restart:
```bash
ssh jonah@192.168.86.201 "curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=v1.31.4+k3s1 sh -"
```

---

## Manifest Reference

| File | Description |
|------|-------------|
| `k8s/data-plane/namespace.yaml` | Creates `data-plane` namespace |
| `k8s/data-plane/redpanda-statefulset.yaml` | Single-broker Redpanda StatefulSet |
| `k8s/data-plane/redpanda-service.yaml` | ClusterIP + NodePort services |
| `scripts/provision-k3s-data-plane.sh` | Idempotent provisioning script |

---

## Cross-Reference

| Ticket | Description |
|--------|-------------|
| OMN-3474 | Preflight — confirmed 192.168.86.201 had no k3s |
| OMN-3488 | **This ticket** — provision k3s + data-plane namespace |
| OMN-3478 | Cloud tunnel script rewrite (uses K3S_BIND_ADDR from this ticket) |
| OMN-3431 | Two-bus policy (local Docker vs cloud k3s) |
