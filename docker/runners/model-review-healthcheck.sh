#!/usr/bin/env bash
# OMN-17876 — fail-closed candidate capability health assertion.
#
# This check reads only the sanitized observation emitted by the private
# capability probe. It never resolves or prints a credential, endpoint, or
# runner identity. The typed Python preflight remains the contract authority;
# this container check enforces the same activation, provenance, freshness,
# label, and reference-presence boundary before the candidate is healthy.
set -euo pipefail

fail() {
    echo "model-review unhealthy: $1" >&2
    exit 1
}

[[ "${MODEL_REVIEW_CAPABILITY_ACTIVE:-0}" == "1" ]] || exit 0
[[ "${MODEL_REVIEW_CONFIG_ACTIVE:-0}" == "1" ]] || fail "capability config is inactive"
[[ "${RUNNER_GROUP:-}" == "omnibase-ci" ]] || fail "required runner group is missing"
case ",${RUNNER_LABELS:-}," in
    *,model-review,*) ;;
    *) fail "required runner label is missing" ;;
esac

observation_path="${MODEL_REVIEW_OBSERVATION_PATH:-/etc/omnibase/model-review-observation.json}"
[[ -r "${observation_path}" ]] || fail "sanitized capability observation is unavailable"
command -v jq >/dev/null 2>&1 || fail "jq is unavailable"

# OMN-17876 intentionally ships no verifier. Active state remains unhealthy
# until the separately authorized operator rollout installs this fixed-path,
# sanctioned receipt verifier. An env flag or UUID cannot satisfy this gate.
verifier_path=/usr/local/bin/model-review-attestation-verifier
[[ -x "${verifier_path}" ]] || fail "sanctioned live attestation verifier is unavailable"
"${verifier_path}" "${observation_path}" >/dev/null || fail "live attestation verifier rejected observation"

# Keep this shell projection byte-for-byte aligned with config/runner_fleet.yaml.
# These are opaque IDs only; the probe never resolves or prints their values.
required_reference_ids=(
    "dc9565c8-7f13-46dc-bd89-9694c13e1d2f"
    "b2a8287b-0a9f-4cbc-b2e8-cf954f9a71f7"
    "2672472a-bac9-4344-8c8c-79da6cb604ae"
)

provenance=$(jq -r '.provenance // empty' "${observation_path}")
[[ "${provenance}" == "runner-local-model-review-preflight" ]] || fail "observation provenance is untrusted"
attestation_id=$(jq -r '.attestation_id // empty' "${observation_path}")
[[ "${attestation_id}" =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]] || fail "attestation identity is missing"

observed_at=$(jq -r '.observed_at // empty' "${observation_path}")
if ! observed_epoch=$(date -u -d "${observed_at}" +%s 2>/dev/null); then
    # The production image uses GNU date; local validation also runs on macOS,
    # whose BSD date accepts an offset without the colon.
    bsd_observed_at="${observed_at}"
    if [[ "${bsd_observed_at}" =~ ^(.+T[0-9]{2}:[0-9]{2}:[0-9]{2})(\.[0-9]+)?([+-][0-9]{2}):([0-9]{2})$ ]]; then
        bsd_observed_at="${BASH_REMATCH[1]}${BASH_REMATCH[3]}${BASH_REMATCH[4]}"
    fi
    observed_epoch=$(date -u -j -f "%Y-%m-%dT%H:%M:%S%z" "${bsd_observed_at}" +%s 2>/dev/null) || fail "observation timestamp is invalid"
fi
now_epoch=$(date -u +%s)
age_seconds=$((now_epoch - observed_epoch))
max_age="${MODEL_REVIEW_MAX_OBSERVATION_AGE_SECONDS:-300}"
[[ "${max_age}" =~ ^[0-9]+$ ]] || fail "observation age budget is invalid"
(( age_seconds >= 0 && age_seconds <= max_age )) || fail "capability observation is stale"

jq -e --arg credential_ref "${required_reference_ids[0]}" \
    --arg endpoint_ref "${required_reference_ids[1]}" \
    --arg healthcheck_ref "${required_reference_ids[2]}" '
    (.reviewer_cli_available == true) and
    ((.present_reference_ids | type) == "array") and
    ((.present_reference_ids | length) == 3) and
    ((.present_reference_ids | unique | length) == 3) and
    (([ $credential_ref, $endpoint_ref, $healthcheck_ref ] - .present_reference_ids) | length == 0) and
    ((.healthy_reference_ids | type) == "array") and
    ((.healthy_reference_ids | length) == 3) and
    ((.healthy_reference_ids | unique | length) == 3) and
    (([ $credential_ref, $endpoint_ref, $healthcheck_ref ] - .healthy_reference_ids) | length == 0) and
    ((.healthy_reference_ids - .present_reference_ids) | length == 0)
' "${observation_path}" >/dev/null || fail "capability references or CLI are not healthy"

exit 0
