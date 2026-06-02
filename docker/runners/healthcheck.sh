#!/usr/bin/env bash
# Docker healthcheck for the GitHub Actions runner container (OMN-12433).
#
# The old healthcheck only ran `pgrep -f Runner.Listener`. That passes even when
# the listener has silently lost its connection to github.com - the exact failure
# that left ~9 runners "Up (healthy)" in Docker while offline in the GitHub pool,
# wedging the merge queue. This check additionally proves github.com egress so a
# runner that cannot reach GitHub is marked unhealthy and removed from rotation.
set -u

# 1. Listener process must be alive (runner agent running at all).
if ! pgrep -f Runner.Listener >/dev/null 2>&1; then
  echo "unhealthy: Runner.Listener not running"
  exit 1
fi

# 2. github.com must be reachable. A connected listener with no in-flight job is
#    expected; an egress fault that drops the GitHub connection is what we catch.
#    Use a short-timeout unauthenticated request. This proves network egress
#    without requiring an API token.
if ! curl -fsS --max-time 8 -o /dev/null https://github.com/; then
  echo "unhealthy: github.com egress unreachable"
  exit 1
fi

echo "healthy: listener up, github.com reachable"
exit 0
