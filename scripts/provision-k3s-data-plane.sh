#!/usr/bin/env bash
# provision-k3s-data-plane.sh
# Idempotent provisioning of k3s + data-plane namespace on 192.168.86.201.
#
# Ticket: OMN-3488
# Prerequisite for: OMN-3478 (cloud tunnel script rewrite)
#
# Usage (run from Mac or directly on the host):
#   ./scripts/provision-k3s-data-plane.sh [--host 192.168.86.201] [--user jonah] [--dry-run]
#
# IMPORTANT — sudo requires interactive TTY on 192.168.86.201 (use_pty sudoers restriction).
# This script must be run interactively (not via non-interactive SSH from an agent):
#
#   Option A (recommended): SSH to the host, then run:
#     ssh jonah@192.168.86.201
#     bash /tmp/provision-k3s-data-plane.sh --local
#
#   Option B: Run remotely with SSH pseudo-TTY allocation:
#     ssh -t jonah@192.168.86.201 'bash -s' < ./scripts/provision-k3s-data-plane.sh
#
# What this script does:
#   1. Installs k3s single-node on the target host (if not already installed)
#   2. Enables and starts the k3s systemd service
#   3. Records the k3s API bind address (needed by OMN-3478 tunnel script)
#   4. Creates the data-plane namespace
#   5. Applies all manifests from k8s/data-plane/ (namespace, StatefulSet, Services)
#   6. Waits for the Redpanda pod to become ready
#   7. Prints verification summary
#
# Dependencies (on the remote host):
#   - systemd (Ubuntu 24.04.3 LTS confirmed)
#   - curl (pre-installed on Ubuntu)
#   - sudo access for the SSH user
#
# After running this script, proceed to OMN-3478:
#   - K3S_BIND_ADDR will be printed by this script (step 3)
#   - The SSH tunnel: ssh -L 29092:<K3S_BIND_ADDR>:29092 -L 6443:<K3S_BIND_ADDR>:6443 jonah@192.168.86.201

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────────
TARGET_HOST="${TARGET_HOST:-192.168.86.201}"
SSH_USER="${SSH_USER:-jonah}"
DRY_RUN=false
LOCAL_MODE=false  # run directly on the host (no SSH wrapping)
MANIFESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/k8s/data-plane"
K3S_VERSION="v1.31.4+k3s1"  # pin for reproducibility

# ── arg parsing ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)    TARGET_HOST="$2"; shift 2 ;;
    --user)    SSH_USER="$2";    shift 2 ;;
    --dry-run) DRY_RUN=true;     shift   ;;
    --local)   LOCAL_MODE=true;  shift   ;;  # run directly on the host, skip SSH
    *)         echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── helpers ──────────────────────────────────────────────────────────────────
info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
error() { echo "[ERROR] $*" >&2; exit 1; }
dry()   { [[ "$DRY_RUN" == "true" ]] && echo "[DRY]   $*" && return 0; return 1; }

ssh_run() {
  # Run a command on the remote host via SSH (skipped in --local mode).
  if [[ "$LOCAL_MODE" == "true" ]]; then
    bash -c "$*"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${SSH_USER}@${TARGET_HOST}" "$@"
  fi
}

ssh_sudo() {
  # Run a sudo command on the remote host via SSH.
  # NOTE: 192.168.86.201 uses 'use_pty' sudoers restriction — requires interactive TTY.
  # In --local mode (running directly on host), sudo works normally.
  # In remote mode, this may fail. Use: ssh -t jonah@192.168.86.201 'bash -s' < script.sh
  if [[ "$LOCAL_MODE" == "true" ]]; then
    sudo bash -c "$1"
  else
    ssh -t -o ConnectTimeout=10 "${SSH_USER}@${TARGET_HOST}" sudo bash -c "$1"
  fi
}

# ── step 0: verify connectivity ──────────────────────────────────────────────
info "Verifying SSH connectivity to ${SSH_USER}@${TARGET_HOST}..."
ssh_run "echo 'SSH OK'" || error "Cannot SSH to ${TARGET_HOST}. Check connectivity and SSH keys."

# ── step 1: install k3s (idempotent) ─────────────────────────────────────────
info "Step 1: Checking k3s installation..."
if ssh_run "which k3s > /dev/null 2>&1"; then
  K3S_INSTALLED_VER=$(ssh_run "k3s --version 2>/dev/null | head -1 || echo unknown")
  info "k3s already installed: ${K3S_INSTALLED_VER}"
else
  info "k3s not found — configuring NodePort range and installing ${K3S_VERSION}..."

  # Configure k3s to allow NodePort 29092 (below default 30000-32767 range).
  # This must be done before k3s starts so the API server picks up the flag.
  K3S_CONFIG="kube-apiserver-arg:\n  - service-node-port-range=1024-65535\n"
  if dry "mkdir -p /etc/rancher/k3s && echo '${K3S_CONFIG}' > /etc/rancher/k3s/config.yaml"; then
    :  # dry run, skip
  else
    ssh_sudo "mkdir -p /etc/rancher/k3s && printf '${K3S_CONFIG}' > /etc/rancher/k3s/config.yaml"
    info "k3s config written: service-node-port-range=1024-65535"
  fi

  # Install k3s — INSTALL_K3S_VERSION must be set for the sh invocation (right side of pipe),
  # not for curl. Placing it before the pipe only exports it to curl, not to the installer.
  if dry "curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=${K3S_VERSION} sh -"; then
    :  # dry run, skip
  else
    ssh_sudo "curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=${K3S_VERSION} sh -"
  fi
fi

# ── step 2: enable + start k3s ───────────────────────────────────────────────
info "Step 2: Enabling and starting k3s service..."
if dry "systemctl enable k3s --now"; then
  :
else
  ssh_sudo "systemctl enable k3s --now"
  # Wait up to 60s for k3s to become active
  for i in $(seq 1 12); do
    STATUS=$(ssh_sudo "systemctl is-active k3s 2>/dev/null || echo inactive")
    if [[ "$STATUS" == "active" ]]; then
      info "k3s is active."
      break
    fi
    info "  Waiting for k3s to start... (${i}/12)"
    sleep 5
  done
  STATUS=$(ssh_sudo "systemctl is-active k3s 2>/dev/null || echo inactive")
  [[ "$STATUS" == "active" ]] || error "k3s failed to start. Check: ssh ${SSH_USER}@${TARGET_HOST} sudo systemctl status k3s"
fi

# ── step 3: record k3s API bind address ──────────────────────────────────────
info "Step 3: Recording k3s API bind address..."
if ! dry "ss -lntp | grep ':6443'"; then
  # k3s binds the API server to 0.0.0.0:6443 by default.
  # The tunnel target is the host's primary IP (not 0.0.0.0).
  BIND_ADDR=$(ssh_run "ss -lntp | grep ':6443' | awk '{print \$4}' | head -1 || echo '0.0.0.0:6443'")
  K3S_BIND_IP="${TARGET_HOST}"  # Use actual host IP — not 0.0.0.0
  info "k3s API listening at: ${BIND_ADDR}"
  info "K3S_BIND_ADDR (for OMN-3478 tunnel): ${K3S_BIND_IP}"
  echo ""
  echo "=== OMN-3478 update ==="
  echo "K3S_BIND_ADDR=${K3S_BIND_IP}"
  echo "Tunnel command:  ssh -L 6443:${K3S_BIND_IP}:6443 -L 29092:${K3S_BIND_IP}:29092 ${SSH_USER}@${TARGET_HOST}"
  echo "======================="
  echo ""
fi

# ── step 4: copy manifests + apply ────────────────────────────────────────────
info "Step 4: Applying k8s/data-plane manifests..."

if [[ ! -d "$MANIFESTS_DIR" ]]; then
  error "Manifests directory not found: ${MANIFESTS_DIR}"
fi

if dry "kubectl apply -f ${MANIFESTS_DIR}/"; then
  :
elif [[ "$LOCAL_MODE" == "true" ]]; then
  # Running directly on host — manifests path must be provided or script is in-repo
  if [[ ! -d "$MANIFESTS_DIR" ]]; then
    error "Manifests directory not found: ${MANIFESTS_DIR}. Copy k8s/data-plane/*.yaml to the host first."
  fi
  sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl apply -f "${MANIFESTS_DIR}/"
else
  # Copy manifests to the remote host
  REMOTE_MANIFESTS_DIR="/tmp/omninode-data-plane-manifests"
  ssh_run "mkdir -p ${REMOTE_MANIFESTS_DIR}"
  scp -o BatchMode=yes "${MANIFESTS_DIR}"/*.yaml "${SSH_USER}@${TARGET_HOST}:${REMOTE_MANIFESTS_DIR}/"

  # Apply using k3s kubectl
  ssh_sudo "KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl apply -f ${REMOTE_MANIFESTS_DIR}/"
fi

# ── step 5: wait for Redpanda pod ────────────────────────────────────────────
info "Step 5: Waiting for omninode-redpanda pod to become ready (up to 3 minutes)..."
if ! dry "kubectl -n data-plane rollout status statefulset/omninode-redpanda"; then
  ssh_sudo "KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl -n data-plane rollout status statefulset/omninode-redpanda --timeout=180s" || \
    warn "Pod not ready within timeout — check: ssh ${SSH_USER}@${TARGET_HOST} sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl -n data-plane get pods"
fi

# ── step 6: verification summary ────────────────────────────────────────────
info "Step 6: Verification summary"
echo ""
echo "=== VERIFICATION ==="
if ! dry "systemctl is-active k3s"; then
  K3S_STATUS=$(ssh_sudo "systemctl is-active k3s 2>/dev/null || echo unknown")
  echo "k3s systemd status:     ${K3S_STATUS}"
fi
if ! dry "kubectl -n data-plane get svc omninode-redpanda"; then
  SVC_OUTPUT=$(ssh_sudo "KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl -n data-plane get svc omninode-redpanda 2>/dev/null || echo 'NOT FOUND'")
  echo "omninode-redpanda svc:  ${SVC_OUTPUT}"
  EXT_SVC=$(ssh_sudo "KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl -n data-plane get svc omninode-redpanda-external 2>/dev/null || echo 'NOT FOUND'")
  echo "omninode-redpanda-ext:  ${EXT_SVC}"
fi
echo "=== END ==="
echo ""
info "Done. OMN-3488 provisioning complete."
info "Next: Update OMN-3478 with K3S_BIND_ADDR=${TARGET_HOST} and rewrite tunnel script."
