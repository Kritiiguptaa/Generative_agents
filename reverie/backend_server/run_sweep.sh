#!/usr/bin/env bash
# Run 06 -- multi-seed sweep for the trading-agent hallucination study.
#
# WHY SEEDS, NOT LONGER RUNS: every report so far has been n=1, and a single
# seed cannot support a rate claim no matter how many steps it covers. On the
# DGX a 200-step run costs ~15-25 min wall clock (measured off run 05: median
# 1.0s per LLM call, 779 calls for 200 steps), so the budget buys seeds.
#
# ORDERING: a seed is finished completely -- all four configs -- before the
# next begins. Kill this at any point and every seed it finished is a complete,
# analysable matched set. Interleaving configs would leave a half-populated
# grid that answers nothing.
#
# BUDGET: stops launching new work once MAX_HOURS is reached. It never kills a
# config mid-run, so the real ceiling is MAX_HOURS plus one config (~50 min).
#
# Usage, from reverie/backend_server:
#   chmod +x run_sweep.sh
#   nohup ./run_sweep.sh > /workspace/sweep.log 2>&1 &
#   tail -f /workspace/sweep.log
#
# Re-running it later skips whatever already completed, so an interrupted
# sweep resumes rather than restarting.

set -u   # NOT -e: one crashed config must not kill the remaining seeds

STEPS=200
SEEDS=(42 43 44 45)
MAX_HOURS=10
OUT=/workspace/run06

MAX_SECONDS=$(( MAX_HOURS * 3600 ))
mkdir -p "$OUT"

# --- preflight: the two things that silently ruin an overnight run ---------
if ! curl -sf http://localhost:11434/api/tags > /dev/null; then
    echo "FATAL: Ollama not responding on :11434. Start it with 'ollama serve'."
    exit 1
fi
for m in mistral nomic-embed-text; do
    if ! curl -s http://localhost:11434/api/tags | grep -q "\"$m"; then
        echo "FATAL: model '$m' not pulled. Run: ollama pull $m"
        exit 1
    fi
done
echo "[$(date +%T)] preflight OK -- ollama up, mistral + nomic-embed-text present"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null \
    || echo "WARNING: nvidia-smi unavailable -- check Ollama is not running CPU-only"
echo "[$(date +%T)] plan: ${#SEEDS[@]} seeds x 4 configs, ${STEPS} steps, cap ${MAX_HOURS}h"

START=$SECONDS

run () {   # run <name> <script> <args...>
    local name="$1"; shift
    if [ -f "$OUT/$name.done" ]; then
        echo "[$(date +%T)] SKIP  $name (already complete)"
        return 0
    fi
    local elapsed=$(( SECONDS - START ))
    if [ "$elapsed" -ge "$MAX_SECONDS" ]; then
        echo "[$(date +%T)] BUDGET reached ($(( elapsed / 60 )) min) -- not starting $name"
        return 1
    fi
    echo "[$(date +%T)] START $name"
    local t0=$SECONDS
    # -u: without it Python block-buffers into the file and a live tail shows
    # nothing for minutes, which reads as a hang.
    if python -u "$@" > "$OUT/$name.log" 2>&1; then
        touch "$OUT/$name.done"
        echo "[$(date +%T)] OK    $name  ($(( SECONDS - t0 ))s)"
    else
        echo "[$(date +%T)] FAIL  $name  ($(( SECONDS - t0 ))s) -- see $OUT/$name.log"
        tail -5 "$OUT/$name.log" | sed 's/^/        /'
    fi
    return 0
}

for s in "${SEEDS[@]}"; do
    echo ""
    echo "=================== SEED $s ==================="

    run "mw_s$s"     trading_reverie.py --fork base_trading --sim run06_mw_s$s \
                     --steps $STEPS --seed $s --fresh                       || break
    run "base_s$s"   trading_reverie.py --fork base_trading --sim run06_base_s$s \
                     --steps $STEPS --seed $s --fresh --no-middleware       || break
    run "pmw_s$s"    paired_eval.py --fork base_trading --sim run06_pmw_s$s \
                     --steps $STEPS --seed $s --fresh --advance-with middleware || break
    run "pbase_s$s"  paired_eval.py --fork base_trading --sim run06_pbase_s$s \
                     --steps $STEPS --seed $s --fresh --advance-with baseline   || break

    echo "---- seed $s complete | $(( (SECONDS - START) / 60 )) min elapsed ----"
done

echo ""
echo "[$(date +%T)] SWEEP ENDED after $(( (SECONDS - START) / 60 )) min"
echo "configs completed: $(ls "$OUT"/*.done 2>/dev/null | wc -l) / $(( ${#SEEDS[@]} * 4 ))"
echo "complete seeds (all 4 configs):"
for s in "${SEEDS[@]}"; do
    n=$(ls "$OUT"/{mw,base,pmw,pbase}_s$s.done 2>/dev/null | wc -l)
    [ "$n" -eq 4 ] && echo "  seed $s  COMPLETE" || echo "  seed $s  partial ($n/4)"
done
