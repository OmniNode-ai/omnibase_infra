#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# k8s-pod-readiness-check.sh -- SSM-based k8s pod READY gate for /redeploy K8S_VERIFY phase
#
# Checks that the 4 required deployments in onex-dev are all READY via AWS SSM.
# Uses AWS SSM send-command to run kubectl on the k8s node — no KUBECONFIG or
# ~/.kube/ required on the local machine.
#
# Required deployments:
#   - omninode-runtime
#   - omninode-runtime-effects
#   - omnibase-intelligence-api
#   - omninode-agent-actions-consumer
#
# Usage:
#   bash scripts/k8s-pod-readiness-check.sh
#   bash scripts/k8s-pod-readiness-check.sh --namespace onex-prod
#
# Exit codes:
#   0 -- All required deployments READY
#   1 -- One or more deployments not READY
#   2 -- SSM instance not reachable (advisory — does not block deploy)

set -euo pipefail

NAMESPACE="${NAMESPACE:-onex-dev}"
SSM_INSTANCE_ID="${SSM_INSTANCE_ID:-i-0e596e8b557e27785}"
AWS_REGION="${AWS_REGION:-us-east-1}"
SSM_TIMEOUT="${SSM_TIMEOUT:-60}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --instance-id) SSM_INSTANCE_ID="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

REQUIRED_DEPLOYMENTS=(
  "omninode-runtime"
  "omninode-runtime-effects"
  "omnibase-intelligence-api"
  "omninode-agent-actions-consumer"
)

# ── Check AWS CLI available ────────────────────────────────────────────────

if ! command -v aws >/dev/null 2>&1; then
  echo "K8S_VERIFY WARNING: aws CLI not found — skipping k8s readiness check (advisory)"
  exit 2
fi

# ── Send SSM command ───────────────────────────────────────────────────────

KUBECTL_CMD="kubectl get deployments -n ${NAMESPACE} -o json"

echo "K8S_VERIFY: checking deployments in namespace ${NAMESPACE} via SSM ${SSM_INSTANCE_ID}"

COMMAND_OUTPUT=$(aws ssm send-command \
  --instance-ids "${SSM_INSTANCE_ID}" \
  --region "${AWS_REGION}" \
  --document-name "AWS-RunShellScript" \
  --parameters "commands=[\"${KUBECTL_CMD}\"]" \
  --output json 2>&1) || {
    echo "K8S_VERIFY WARNING: SSM send-command failed — instance may be offline (advisory)"
    echo "  Error: ${COMMAND_OUTPUT}"
    exit 2
  }

COMMAND_ID=$(echo "${COMMAND_OUTPUT}" | python3 -c "import sys,json; print(json.load(sys.stdin)['Command']['CommandId'])" 2>/dev/null) || {
  echo "K8S_VERIFY WARNING: could not parse SSM command ID (advisory)"
  exit 2
}

# Wait for command to complete
sleep 5
RESULT=$(aws ssm get-command-invocation \
  --command-id "${COMMAND_ID}" \
  --instance-id "${SSM_INSTANCE_ID}" \
  --region "${AWS_REGION}" \
  --output json 2>&1) || {
    echo "K8S_VERIFY WARNING: could not retrieve SSM command result (advisory)"
    exit 2
  }

STATUS=$(echo "${RESULT}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('Status','Unknown'))" 2>/dev/null || echo "Unknown")
if [[ "${STATUS}" != "Success" ]]; then
  echo "K8S_VERIFY WARNING: SSM command status=${STATUS} (advisory)"
  exit 2
fi

DEPLOYMENTS_JSON=$(echo "${RESULT}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('StandardOutputContent','{}'))" 2>/dev/null || echo "{}")

# ── Parse readiness ────────────────────────────────────────────────────────

NOT_READY=()

for deployment in "${REQUIRED_DEPLOYMENTS[@]}"; do
  READY=$(echo "${DEPLOYMENTS_JSON}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
items = data.get('items', [])
for item in items:
    name = item.get('metadata', {}).get('name', '')
    if name == '${deployment}':
        status = item.get('status', {})
        desired = status.get('replicas', 0)
        ready = status.get('readyReplicas', 0)
        print(f'{ready}/{desired}')
        sys.exit(0)
print('0/0')
" 2>/dev/null || echo "0/0")

  READY_COUNT="${READY%%/*}"
  TOTAL_COUNT="${READY##*/}"

  if [[ "${READY_COUNT}" == "${TOTAL_COUNT}" && "${TOTAL_COUNT}" != "0" ]]; then
    echo "K8S_VERIFY OK: ${deployment} ${READY} READY"
  else
    echo "K8S_VERIFY FAIL: ${deployment} ${READY} not READY"
    NOT_READY+=("${deployment} (${READY})")
  fi
done

# ── Summary ────────────────────────────────────────────────────────────────

if [[ ${#NOT_READY[@]} -gt 0 ]]; then
  echo ""
  echo "K8S_VERIFY FAILED: ${#NOT_READY[@]} deployment(s) not READY:"
  for d in "${NOT_READY[@]}"; do
    echo "  - ${d}"
  done
  exit 1
fi

echo ""
echo "K8S_VERIFY OK: all ${#REQUIRED_DEPLOYMENTS[@]} deployments READY in ${NAMESPACE}"
exit 0
