#!/usr/bin/env bash
# Hardened curl wrapper for the runner image build (OMN-13915).
#
# COPY'd into the image as /usr/local/bin/omni-curl rather than materialised via
# a `RUN cat <<'EOF'` heredoc: heredoc-in-RUN is BuildKit-only syntax, and the
# self-hosted omnibase-ci runner lost its buildx component and fell back to the
# legacy builder — which silently wrote an empty omni-curl, so every download
# no-oped and the uv/gosu/runner fetch steps failed with a missing tarball. A
# COPY'd script builds correctly under either builder.
set -euo pipefail
output_file=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    -o|--output)
      if ((i + 1 < ${#args[@]})); then
        output_file="${args[$((i + 1))]}"
      fi
      ;;
    -o*)
      output_file="${args[$i]#-o}"
      ;;
    --output=*)
      output_file="${args[$i]#--output=}"
      ;;
  esac
done

for attempt in 1 2 3; do
  if curl \
    --http1.1 \
    --fail \
    --show-error \
    --location \
    --connect-timeout 15 \
    --max-time 300 \
    --retry 5 \
    --retry-delay 5 \
    --retry-max-time 300 \
    --retry-connrefused \
    --retry-all-errors \
    "$@"; then
    if [[ -z "${output_file}" || -s "${output_file}" ]]; then
      exit 0
    fi
    echo "omni-curl: curl succeeded but output file is missing or empty: ${output_file}" >&2
  fi
  if [[ -n "${output_file}" ]]; then
    rm -f "${output_file}"
  fi
  sleep $((attempt * 5))
done

echo "omni-curl: failed after retries: $*" >&2
exit 1
