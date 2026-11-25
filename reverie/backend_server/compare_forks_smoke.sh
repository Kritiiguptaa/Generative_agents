#!/usr/bin/env bash
# Compare rosters on short runs: does a persona wording change make agents trade more?
#
#   FORKS="base_trading base_trading_active" SEEDS="42 43" STEPS=30 ./compare_forks_smoke.sh
#
# Runs the middleware arm only, for each fork and seed, one after another, then
# prints each fork's per-agent table (hold %, buys, persona coverage) from
# analyze_sweep.py. Same seed, same temperature, same code for every fork, so the
# only difference between the tables is the fork's persona files.
#
# REFUSES TO START if a trading_reverie process is already running. Pasting this
# twice once started two jobs writing the same sim and corrupted both logs.
#
# Logs go to $OUT/<fork>/mw_s<seed>.log (the layout analyze_sweep.py reads), and
# the sims are named ${TAG}_<fork>_mw_s<seed>, so nothing touches a real sweep.

set -u
FORKS=${FORKS:-"base_trading base_trading_active"}
SEEDS=${SEEDS:-"42 43"}
STEPS=${STEPS:-30}
TEMPERATURE=${TEMPERATURE:-0.7}
OUT=${OUT:-/workspace/cmp}
TAG=${TAG:-cmp}
PY=${PY:-python}

if pgrep -f "trading_reverie.py" > /dev/null 2>&1; then
    echo "FATAL: a trading_reverie.py process is already running. Wait for it or stop it."
    pgrep -af "trading_reverie.py" | cut -c1-120
    exit 1
fi

for f in $FORKS; do
    mkdir -p "$OUT/$f"
    for s in $SEEDS; do
        log="$OUT/$f/mw_s$s.log"
        echo "[$(date +%T)] RUN  fork=$f seed=$s steps=$STEPS  (live log: $log)"
        if ! $PY -u trading_reverie.py --fork "$f" --sim "${TAG}_${f}_mw_s$s" \
                --steps "$STEPS" --seed "$s" --temperature "$TEMPERATURE" --fresh \
                > "$log" 2>&1; then
            echo "[$(date +%T)] FAIL fork=$f seed=$s -- last lines of $log:"
            tail -5 "$log" | sed 's/^/        /'
        fi
    done
done

for f in $FORKS; do
    echo ""
    echo "================ fork: $f ================"
    python3 analyze_sweep.py --out "$OUT/$f" --fork "$f" | sed -n '/^## Per agent/,/^Caveats/p'
done
