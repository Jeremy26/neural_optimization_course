#!/bin/bash
# Lock Jetson Orin into a reproducible benchmark state.
# Run with sudo. One-time per boot.
#
# Picks the highest power mode the board supports:
#   - Orin Nano Super → MAXN_SUPER  (25W)
#   - Orin Nano       → MAXN        (15W, mode 1 — note: mode 0 is 7W LOW)
#   - AGX Orin        → MAXN        (mode 0)
set -euo pipefail

NVPCONF=/etc/nvpmodel.conf

pick_mode() {
    # Prefer MAXN_SUPER, then MAXN, then highest-numbered mode in nvpmodel.conf.
    local id
    for name in MAXN_SUPER MAXN; do
        id=$(awk -v n="$name" '
            /^< POWER_MODEL/ {
                for (i=1;i<=NF;i++) {
                    if ($i ~ /^ID=/)   { split($i,a,"="); cur_id=a[2] }
                    if ($i ~ /^NAME=/) { split($i,a,"="); cur_name=a[2] }
                }
                if (cur_name == n) { print cur_id; exit }
            }' "$NVPCONF")
        if [ -n "$id" ]; then
            echo "$id $name"
            return
        fi
    done
    # Fallback: highest ID listed
    id=$(awk '
        /^< POWER_MODEL/ {
            for (i=1;i<=NF;i++) if ($i ~ /^ID=/) { split($i,a,"="); print a[2] }
        }' "$NVPCONF" | sort -n | tail -1)
    echo "$id UNKNOWN"
}

read -r MODE_ID MODE_NAME < <(pick_mode)
if [ -z "${MODE_ID:-}" ]; then
    echo "ERROR: could not determine a power mode from $NVPCONF" >&2
    exit 1
fi

echo "==> Setting power mode $MODE_ID ($MODE_NAME)..."
nvpmodel -m "$MODE_ID"
nvpmodel -q

echo "==> Locking all clocks to max..."
jetson_clocks

echo "==> Verifying clock state..."
jetson_clocks --show | head -20

echo ""
echo "Orin is locked into max-performance mode ($MODE_NAME)."
echo "Run benchmarks now. State resets on reboot."
