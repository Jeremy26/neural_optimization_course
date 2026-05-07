#!/bin/bash
# Lock Jetson Orin into a reproducible benchmark state.
# Run with sudo. One-time per boot.
set -euo pipefail

echo "==> Setting power mode to MAXN (mode 0, no power cap)..."
nvpmodel -m 0
nvpmodel -q

echo "==> Locking all clocks to max..."
jetson_clocks

echo "==> Verifying clock state..."
jetson_clocks --show | head -20

echo ""
echo "Orin is locked into max-performance mode."
echo "Run benchmarks now. State resets on reboot."
