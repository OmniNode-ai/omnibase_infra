#!/usr/bin/env bash
# Delegated code edit loop model (OMN-20290): two runners race on one
# correlation id; each turn writes an allowed or forbidden path, runs a
# declared or undeclared check, or finishes.
# Property (meaning)                         Kind       Mutation that must break it
# OneReceipt (one receipt per correlation)   invariant  mut_claim (check-then-set claim)
# NoForbiddenWrite (globs confine writes)    invariant  mut_glob (no writable-glob check)
# NoUndeclaredCheck (only declared checks)   invariant  mut_check (no declared-name check)
# TurnBound (at most MaxTurns turns)         invariant  mut_cap (no turn cap)
# Terminates (every runner ends)             temporal   (checked on Model)
# Run with TLA2TOOLS_JAR set:
#   TLA2TOOLS_JAR=/path/tla2tools.jar ./run_checks.sh results
# results/ holds committed TLC output for Model.cfg and each mut_*.cfg,
# plus model.sha256, the digest of the spec and cfg files used for them.
# Bounds: 2 runners, 3 turns.
# Needs TLA2TOOLS_JAR (path to tla2tools.jar) and either `java` or docker (lab: .201).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: run_checks.sh <outdir>}"
JAR="${TLA2TOOLS_JAR:?set TLA2TOOLS_JAR to tla2tools.jar}"
mkdir -p "$OUT"
run_tlc() {
  local cfg="$1"
  if command -v java >/dev/null 2>&1; then
    (cd "$HERE" && java -cp "$JAR" tlc2.TLC -config "$cfg" -workers 2 -metadir "$(mktemp -d)" DelegatedCodeEditLoop.tla)
  else
    docker run --rm -v "$HERE":/m:ro -v "$JAR":/tla2tools.jar:ro -w /m eclipse-temurin:17-jre-alpine \
      java -cp /tla2tools.jar tlc2.TLC -config "$cfg" -workers 2 -metadir /tmp/tlcmeta DelegatedCodeEditLoop.tla
  fi
}
for cfg in Model.cfg mut_*.cfg; do
  run_tlc "$cfg" > "$OUT/${cfg%.cfg}.out" 2>&1
  echo "$cfg exit=$?"
done
( cd "$HERE" && cat DelegatedCodeEditLoop.tla *.cfg | shasum -a 256 | cut -d' ' -f1 ) > "$OUT/model.sha256"
