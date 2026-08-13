#!/usr/bin/env bash
# monitor_generation.sh — LIVE dashboard for high-throughput generation
#
# Usage:
#   watch -n 5 ./scripts/monitor_generation.sh outputs/billiards
#
# Or standalone:
#   ./scripts/monitor_generation.sh outputs/billiards

set -euo pipefail

OUTPUT_DIR="${1:-outputs/billiards}"
PHYSICS_DIR="$OUTPUT_DIR/physics"
TOTAL_TARGET="${2:-36000}"   # default: 50h / 5s = 36000 episodes

echo "═══════════════════════════════════════════════════"
echo "  PHY-DATA-GEN  MONITOR  $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════"

# ── Count completed episodes ──────────────────────────
if [ -d "$PHYSICS_DIR" ]; then
    COMPLETED=$(find "$PHYSICS_DIR" -name "validation.json" -maxdepth 2 2>/dev/null | wc -l)
    PASSED=$(find "$PHYSICS_DIR" -name "validation.json" -maxdepth 2 -exec grep -l '"passed": true' {} \; 2>/dev/null | wc -l)
else
    COMPLETED=0
    PASSED=0
fi

PCT=$(( COMPLETED * 100 / TOTAL_TARGET 2>/dev/null || 0 ))
FAILED=$(( COMPLETED - PASSED ))

echo "  Episodes:   $COMPLETED / $TOTAL_TARGET  ($PCT%)"
echo "  Valid:      $PASSED"
echo "  Failed:     $FAILED"

# ── Data volume ───────────────────────────────────────
DATA_HOURS=$(echo "scale=2; $COMPLETED * 5 / 3600" | bc 2>/dev/null || echo "?")
echo "  Data h:     ${DATA_HOURS}h"

# ── Disk usage ────────────────────────────────────────
if [ -d "$OUTPUT_DIR" ]; then
    DISK_USAGE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    echo "  Disk:       $DISK_USAGE"
fi

# ── Throughput estimate ───────────────────────────────
# Look at the last 10 validation timestamps to estimate rate
if [ -d "$PHYSICS_DIR" ]; then
    LAST_MOD=$(find "$PHYSICS_DIR" -name "validation.json" -maxdepth 2 \
        -exec stat --format='%Y' {} \; 2>/dev/null | sort -n | tail -1)

    if [ -n "$LAST_MOD" ] && [ "$COMPLETED" -gt 0 ]; then
        NOW=$(date +%s)
        ELAPSED=$(( NOW - LAST_MOD + 5 ))  # +5s grace
        if [ "$ELAPSED" -lt 0 ]; then
            ELAPSED=1
        fi

        # Get earliest validation file for total elapsed
        FIRST_MOD=$(find "$PHYSICS_DIR" -name "validation.json" -maxdepth 2 \
            -exec stat --format='%Y' {} \; 2>/dev/null | sort -n | head -1)
        if [ -n "$FIRST_MOD" ]; then
            TOTAL_ELAPSED=$(( NOW - FIRST_MOD ))
            if [ "$TOTAL_ELAPSED" -gt 0 ]; then
                AVG_EPS=$(echo "scale=2; $COMPLETED / $TOTAL_ELAPSED * 60" | bc 2>/dev/null || echo "?")
                echo "  Avg rate:   ${AVG_EPS} eps/min"

                if [ "$COMPLETED" -lt "$TOTAL_TARGET" ]; then
                    REMAINING=$(( TOTAL_TARGET - COMPLETED ))
                    SPEED=$(echo "scale=4; $COMPLETED / $TOTAL_ELAPSED" | bc 2>/dev/null || echo "0")
                    if [ "$(echo "$SPEED > 0" | bc 2>/dev/null)" = "1" ]; then
                        ETA_SEC=$(echo "scale=0; $REMAINING / $SPEED" | bc 2>/dev/null || echo "0")
                        if [ "$ETA_SEC" -gt 0 ]; then
                            ETA_H=$(( ETA_SEC / 3600 ))
                            ETA_M=$(( (ETA_SEC % 3600) / 60 ))
                            echo "  ETA @ rate: ${ETA_H}h${ETA_M}m"
                        fi
                    fi
                fi
            fi
        fi
    fi
fi

# ── GPU status ────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r idx name gpu_util mem_used mem_total temp; do
        mem_pct=$(( mem_used * 100 / mem_total 2>/dev/null || 0 ))
        echo "  GPU $idx ($name): ${gpu_util}% util, ${mem_used}M/${mem_total}M (${mem_pct}%), ${temp}°C"
    done
fi

# ── Active Isaac Sim processes ────────────────────────
NUM_SIMS=$(pgrep -f "isaacsim" 2>/dev/null | wc -l || echo "0")
NUM_PYTHON=$(pgrep -f "generate_high_throughput" 2>/dev/null | wc -l || echo "0")
echo "  Isaac sim processes: $NUM_SIMS"
echo "  Generator processes: $NUM_PYTHON"

echo "───────────────────────────────────────────────────"
