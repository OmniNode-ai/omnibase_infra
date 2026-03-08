#!/usr/bin/env bash
# cloud-bus-tunnel.sh — Persistent cloud bus tunnel (SSM + kubectl port-forward)
# Managed by launchd plist ai.omninode.cloud-bus-tunnel
#
# Architecture:
#   1. SSM session → localhost:6443 → k8s API (i-0e596e8b557e27785:6443)
#   2. kubectl port-forward → localhost:29092 → svc/omninode-redpanda:9092
#   3. kubectl port-forward → localhost:9092 → svc/omninode-redpanda:9092
#
# The script runs as a long-lived process. launchd restarts it on exit.

set -euo pipefail

KUBECONFIG="${HOME}/.kube/omninode-mvp1"
SSM_INSTANCE="i-0e596e8b557e27785"
REGION="us-east-1"
LOG_DIR="/tmp"

cleanup() {
    echo "[cloud-bus-tunnel] Shutting down..."
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Ensure AWS SSO session is valid
if ! aws sts get-caller-identity --region "$REGION" >/dev/null 2>&1; then
    echo "[cloud-bus-tunnel] ERROR: AWS credentials expired. Run 'aws sso login' first." >&2
    exit 1
fi

# 1. Start SSM tunnel for k8s API
echo "[cloud-bus-tunnel] Starting SSM tunnel to ${SSM_INSTANCE}:6443..."
aws ssm start-session \
    --target "$SSM_INSTANCE" \
    --region "$REGION" \
    --document-name AWS-StartPortForwardingSession \
    --parameters '{"portNumber":["6443"],"localPortNumber":["6443"]}' \
    > "${LOG_DIR}/ssm-k8s-tunnel.log" 2>&1 &
SSM_PID=$!

# Wait for SSM to establish
sleep 8

# Verify SSM tunnel is alive
if ! kill -0 "$SSM_PID" 2>/dev/null; then
    echo "[cloud-bus-tunnel] ERROR: SSM tunnel failed to start" >&2
    exit 1
fi

# 2. Start kubectl port-forwards
echo "[cloud-bus-tunnel] Starting kubectl port-forward 29092->9092..."
kubectl --kubeconfig "$KUBECONFIG" -n data-plane \
    port-forward svc/omninode-redpanda 29092:9092 \
    > "${LOG_DIR}/omninode-redpanda-tunnel.log" 2>&1 &

echo "[cloud-bus-tunnel] Starting kubectl port-forward 9092->9092..."
kubectl --kubeconfig "$KUBECONFIG" -n data-plane \
    port-forward svc/omninode-redpanda 9092:9092 \
    > "${LOG_DIR}/omninode-redpanda-tunnel-9092.log" 2>&1 &

echo "[cloud-bus-tunnel] All tunnels started. Waiting..."

# Wait for any child to exit (triggers launchd restart)
# macOS ships bash 3.2 which lacks `wait -n`, so poll background PIDs
while true; do
    for pid in $SSM_PID $(jobs -p); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[cloud-bus-tunnel] Process $pid exited. Cleaning up for restart..."
            exit 1
        fi
    done
    sleep 5
done
