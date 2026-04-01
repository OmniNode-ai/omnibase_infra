#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# check-stale-images.sh — detect when local Docker images are older than
# the source code that produced them.
#
# Usage:
#   scripts/check-stale-images.sh [compose-file] [profile]
#
# Prints image names that are stale (code is newer than image).
# Exit 0 with output  = stale images found (rebuild recommended)
# Exit 0 with no output = all images are current
# Exit 1 = error

set -euo pipefail

COMPOSE_FILE="${1:-docker/docker-compose.generated.yml}"
PROFILE="${2:-runtime}"
INFRA_DIR="${OMNIBASE_INFRA_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# Get the latest commit timestamp in the repo (epoch seconds)
code_ts=$(git -C "${INFRA_DIR}" log -1 --format='%ct' 2>/dev/null || echo "0")
if [[ "${code_ts}" == "0" ]]; then
    echo "[check-stale-images] ERROR: Cannot determine git commit timestamp" >&2
    exit 1
fi

# Get runtime-profiled service names from the compose file
compose_path="${INFRA_DIR}/${COMPOSE_FILE}"
if [[ ! -f "${compose_path}" ]]; then
    # Try as absolute path
    compose_path="${COMPOSE_FILE}"
fi

if [[ ! -f "${compose_path}" ]]; then
    echo "[check-stale-images] ERROR: Compose file not found: ${COMPOSE_FILE}" >&2
    exit 1
fi

# Extract images for services with the given profile that have a build context
# (services using pre-built images like postgres/redpanda don't need rebuilding)
service_images=$(docker compose -f "${compose_path}" --profile "${PROFILE}" config --format json 2>/dev/null \
    | jq -r '.services | to_entries[]
        | select(.value.profiles // [] | index("'"${PROFILE}"'"))
        | select(.value.build != null)
        | .value.image // .key' \
    | sort -u)

if [[ -z "${service_images}" ]]; then
    # No buildable runtime services found — nothing to check
    exit 0
fi

stale_found=0
while IFS= read -r image; do
    [[ -z "${image}" ]] && continue

    # Get image creation timestamp (epoch seconds)
    image_created=$(docker inspect --format '{{.Created}}' "${image}" 2>/dev/null || echo "")

    if [[ -z "${image_created}" ]]; then
        # Image doesn't exist locally — definitely needs building
        echo "${image}"
        stale_found=1
        continue
    fi

    # Convert ISO 8601 timestamp to epoch seconds
    # macOS date and GNU date handle this differently
    if date --version &>/dev/null 2>&1; then
        # GNU date
        image_ts=$(date -d "${image_created}" +%s 2>/dev/null || echo "0")
    else
        # macOS date — parse ISO 8601
        image_ts=$(date -j -f "%Y-%m-%dT%H:%M:%S" "$(echo "${image_created}" | cut -c1-19)" +%s 2>/dev/null || echo "0")
    fi

    if [[ "${code_ts}" -gt "${image_ts}" ]]; then
        echo "${image}"
        stale_found=1
    fi
done <<< "${service_images}"

exit 0
