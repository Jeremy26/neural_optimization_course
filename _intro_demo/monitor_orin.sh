#!/bin/bash
# Stream tegrastats every 200ms while the demo runs, log to file.
# Useful for grabbing GPU/CPU/temp/power overlay numbers for the video.
set -euo pipefail

LOG="${1:-orin_tegrastats.log}"

echo "==> Logging tegrastats to $LOG (Ctrl-C to stop)"
echo "    Format: RAM | CPU | GPU | EMC | temps | power"
echo ""

tegrastats --interval 200 --logfile "$LOG" &
PID=$!

trap "kill $PID 2>/dev/null; echo; echo 'Stopped.'" INT
wait $PID
