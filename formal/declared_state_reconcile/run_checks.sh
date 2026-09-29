#!/usr/bin/env bash
# Declared-state reconcile model: one reconcile of surface grants against a
# declared set, with concurrent hand edits, two reconcilers, and an
# observability flag.
# Property (meaning)                    Kind       Mutation that must break it
# Converges (apply and hand edit)       temporal   leak_lock (refusal keeps lock)
# NoInterleave (reconciles never apply together) invariant  no_lock
# NoStaleApply (stale plan is refused)   invariant  no_fresh
# NoNeededRemoved (declared grant kept)  invariant  no_filter
# NoApplyOnUnobservable                  invariant  no_observable_guard
# Run with TLA2TOOLS_JAR set:
#   TLA2TOOLS_JAR=/path/tla2tools.jar ./run_checks.sh results
# results/ holds committed TLC output for Model.cfg and each mut_*.cfg,
# plus model.sha256, the digest of the spec and cfg files used for them.
# The unit test checks the digest, passing model, and mutation violations.
# Bounds: 3 grants, 2 reconcilers, 2 hand edits, 1 observability flip.
# Run TLC on the model and every mutation. Usage: run_checks.sh <outdir>
# Needs TLA2TOOLS_JAR (path to tla2tools.jar) and either `java` or docker (lab: .201).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: run_checks.sh <outdir>}"
JAR="${TLA2TOOLS_JAR:?set TLA2TOOLS_JAR to tla2tools.jar}"
mkdir -p "$OUT"
run_tlc() {
  local cfg="$1"
  if command -v java >/dev/null 2>&1; then
    (cd "$HERE" && java -cp "$JAR" tlc2.TLC -config "$cfg" -workers 2 -metadir "$(mktemp -d)" DeclaredStateReconcile.tla)
  else
    docker run --rm -v "$HERE":/m:ro -v "$JAR":/tla2tools.jar:ro -w /m eclipse-temurin:17-jre-alpine \
      java -cp /tla2tools.jar tlc2.TLC -config "$cfg" -workers 2 -metadir /tmp/tlcmeta DeclaredStateReconcile.tla
  fi
}
for cfg in Model.cfg mut_*.cfg; do
  run_tlc "$cfg" > "$OUT/${cfg%.cfg}.out" 2>&1
  echo "$cfg exit=$?"
done
( cd "$HERE" && cat DeclaredStateReconcile.tla *.cfg | shasum -a 256 | cut -d' ' -f1 ) > "$OUT/model.sha256"
