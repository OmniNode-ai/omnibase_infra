#!/bin/bash
# Run all codegen node tests

cd /Volumes/PRO-G40/Code/omninode_bridge

echo "==================== CODEGEN NODES TEST SUMMARY ===================="
echo ""

total_passed=0
total_failed=0
total_tests=0

for node_dir in \
  src/omninode_bridge/nodes/codegen_stub_extractor_effect/v1_0_0 \
  src/omninode_bridge/nodes/codegen_code_validator_effect/v1_0_0 \
  src/omninode_bridge/nodes/codegen_code_injector_effect/v1_0_0 \
  src/omninode_bridge/nodes/codegen_store_effect/v1_0_0 \
  src/omninode_bridge/nodes/codegen_metrics_reducer/v1_0_0; do

  node_name=$(basename $(dirname "$node_dir"))
  echo "Testing: $node_name"
  echo "----------------------------------------"

  cd "$node_dir"
  output=$(poetry run pytest tests/test_node.py -v --tb=no -q 2>&1)

  # Extract test counts from output
  passed=$(echo "$output" | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+" || echo "0")
  failed=$(echo "$output" | grep -oE "[0-9]+ failed" | grep -oE "[0-9]+" || echo "0")

  total_passed=$((total_passed + passed))
  total_failed=$((total_failed + failed))
  total_tests=$((total_tests + passed + failed))

  echo "  Passed: $passed"
  echo "  Failed: $failed"
  echo ""

  cd /Volumes/PRO-G40/Code/omninode_bridge
done

echo "==================== OVERALL SUMMARY ===================="
echo "Total Tests: $total_tests"
echo "Passed: $total_passed"
echo "Failed: $total_failed"
if [ $total_tests -gt 0 ]; then
  pass_rate=$((total_passed * 100 / total_tests))
  echo "Pass Rate: ${pass_rate}%"
fi
echo "========================================================="
