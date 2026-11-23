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

# Overridable from the environment so the script can be exercised against a
# scratch directory without editing it. Editing a sweep script to test it is
# how a test edit reaches a real run.
STEPS=${STEPS:-200}
read -r -a SEEDS <<< "${SEEDS:-42 43 44 45}"
MAX_HOURS=${MAX_HOURS:-10}
OUT=${OUT:-/workspace/run07}
# Sampling temperature for the decision call. MUST stay > 0: run 06 swept four
# seeds at temperature=0, where decoding is greedy and the seed reaches only
# the market generator. The middleware arm consequently replayed one
# trajectory on every seed (Marcus buy=7/hold=193, PnL +$608.62 on all four;
# s44 and s45 identical for all three agents), so four seeds bought n~=1 and
# the sweep failed at its only purpose. Passed explicitly rather than left to
# the CLI default so the sweep records the setting it ran under.
TEMPERATURE=0.7

# Optional extra arms, off by default so the sweep's cost and shape are what
# the four core configs have always been. Each adds work in proportion to the
# seed count: ablations add 3 configs per seed, basecap adds 1, so with 4 seeds
# turning both on takes the grid from 16 to 32 configs and roughly doubles the
# wall clock. Enable from the environment:
#   RUN_ABLATIONS=1 RUN_BASECAP=1 ./run_sweep.sh
RUN_ABLATIONS=${RUN_ABLATIONS:-0}
RUN_BASECAP=${RUN_BASECAP:-0}

# Minimum free space required before the sweep will start, in MiB, on BOTH the
# output directory and the simulation storage directory. 16 configs x 200 steps
# produce logs in the tens of MiB plus a persona checkpoint per sim, so this is
# generous; the point is to refuse obviously-doomed runs, not to predict usage.
MIN_FREE_MB=${MIN_FREE_MB:-4096}
# Size of the real write test, in MiB. df is unreliable on this NFS export, so
# free space is also verified by actually writing.
WRITE_TEST_MB=${WRITE_TEST_MB:-256}
# Staging directory on LOCAL disk. Logs are written here and copied to $OUT
# only once a config finishes, so a failing copy cannot truncate a live log.
STAGE=${STAGE:-/tmp/sweep_stage}

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown argument: $arg"; echo "usage: $0 [--dry-run]"; exit 2 ;;
    esac
done

MAX_SECONDS=$(( MAX_HOURS * 3600 ))
mkdir -p "$OUT" "$STAGE"

# Where the simulations themselves are written. This fills up faster than $OUT
# and lives on a different filesystem, so both are checked: run 07 died on
# ENOSPC and checking only the log directory would not have caught it.
SIM_STORAGE=${SIM_STORAGE:-../../environment/frontend_server/storage}

# --- preflight --------------------------------------------------------------
# free_mb <dir> -- available MiB, or empty if df cannot answer.
free_mb () {
    df -Pm "$1" 2>/dev/null | awk 'NR==2 {print $4}'
}

# check_space <dir> <label> -- refuse to start if df says there is not enough.
check_space () {
    local dir="$1" label="$2"
    mkdir -p "$dir" 2>/dev/null
    local avail
    avail=$(free_mb "$dir")
    if [ -z "$avail" ]; then
        echo "WARNING: df could not report free space for $label ($dir)."
        echo "         Continuing: the write test below is the real check."
        return 0
    fi
    echo "[$(date +%T)] $label: ${avail} MiB free ($dir)"
    if [ "$avail" -lt "$MIN_FREE_MB" ]; then
        echo "FATAL: only ${avail} MiB free on $label, need ${MIN_FREE_MB}."
        echo "       Free space or lower MIN_FREE_MB, then rerun."
        exit 1
    fi
    return 0
}

# write_test <dir> -- actually write, fsync and delete a file of WRITE_TEST_MB.
#
# THIS IS THE CHECK THAT MATTERS. Run 07 died when the NFS export returned
# ENOSPC while the client still reported 99 GB free, so df alone cannot be
# trusted here. conv=fsync forces the data out rather than leaving it in the
# page cache, which is what makes a client-side quota or a full export show up
# now instead of eight hours in.
write_test () {
    local dir="$1"
    local probe="$dir/.sweep_write_test.$$"
    if ! dd if=/dev/zero of="$probe" bs=1M count="$WRITE_TEST_MB" \
         conv=fsync status=none 2>/dev/null; then
        rm -f "$probe"
        echo "FATAL: could not write ${WRITE_TEST_MB} MiB to $dir."
        echo "       df may still report free space: run 07 died exactly this"
        echo "       way, with the client showing 99 GB available."
        exit 1
    fi
    rm -f "$probe"
    echo "[$(date +%T)] write test OK -- ${WRITE_TEST_MB} MiB written and fsynced to $dir"
}

# Disk is checked FIRST, before Ollama. It is what actually killed run 07, and
# checking it first means a space problem is reported even when the server is
# also down, rather than being hidden behind an unrelated fatal error.
check_space "$OUT" "output dir"
check_space "$SIM_STORAGE" "sim storage"
check_space "$STAGE" "local stage"
# Skipped on a dry run: writing and fsyncing hundreds of MiB to check a plan
# that will not run is a cost with no answer attached.
if [ "$DRY_RUN" -eq 0 ]; then
    write_test "$OUT"
    write_test "$STAGE"
else
    echo "[$(date +%T)] write test skipped (--dry-run)"
fi

# A dry run calls no model, so a missing server is reported and not fatal:
# requiring Ollama in order to print the plan would make --dry-run useless for
# checking a sweep before the GPU is free.
if ! curl -sf http://localhost:11434/api/tags > /dev/null; then
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "WARNING: Ollama not responding on :11434. Fine for --dry-run;"
        echo "         a real run will refuse to start until it is up."
    else
        echo "FATAL: Ollama not responding on :11434. Start it with 'ollama serve'."
        exit 1
    fi
else
    for m in mistral nomic-embed-text; do
        if ! curl -s http://localhost:11434/api/tags | grep -q "\"$m"; then
            echo "FATAL: model '$m' not pulled. Run: ollama pull $m"
            exit 1
        fi
    done
    echo "[$(date +%T)] preflight OK -- ollama up, mistral + nomic-embed-text present"
fi
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null \
    || echo "WARNING: nvidia-smi unavailable -- check Ollama is not running CPU-only"
CONFIGS_PER_SEED=4
[ "$RUN_ABLATIONS" -eq 1 ] && CONFIGS_PER_SEED=$(( CONFIGS_PER_SEED + 3 ))
[ "$RUN_BASECAP" -eq 1 ] && CONFIGS_PER_SEED=$(( CONFIGS_PER_SEED + 1 ))
echo "[$(date +%T)] plan: ${#SEEDS[@]} seeds x ${CONFIGS_PER_SEED} configs, ${STEPS} steps, temp ${TEMPERATURE}, cap ${MAX_HOURS}h"
[ "$RUN_ABLATIONS" -eq 1 ] && echo "           + ablation arms (compression / idrag / filter)"
[ "$RUN_BASECAP" -eq 1 ] && echo "           + baseline-states-risk-cap arm"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[$(date +%T)] DRY RUN -- printing the plan, calling no model"
fi

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
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] would run: $name"
        echo "          python -u $*"
        return 0
    fi
    local t0=$SECONDS
    # Written to LOCAL disk first. Writing straight to the NFS export is how a
    # failing export truncates a log that is still being appended to; staging
    # locally means the only network write is one copy of a finished file.
    local stage_log="$STAGE/$name.log"
    # The per-step output goes to stage_log, not here, so say where to watch it.
    echo "[$(date +%T)] START $name   (live: tail -f $stage_log)"
    # -u: without it Python block-buffers into the file and a live tail shows
    # nothing for minutes, which reads as a hang.
    if python -u "$@" > "$stage_log" 2>&1; then
        # Exit 0 is not proof the log survived. Run 07 died mid-write when the
        # NFS export returned ENOSPC while the client still saw 99GB free, and
        # a write that fails during the final flush can truncate the report
        # while Python exits clean. A .done on a truncated log makes the next
        # resume SKIP a config that was never analysable. Both harnesses end
        # their header with "REPORT", so requiring it means .done implies a
        # readable report.
        if grep -q "REPORT$" "$stage_log"; then
            # Copy, verify the copy, and only then mark done. A .done written
            # before the copy is checked would claim a config whose log never
            # arrived.
            if cp "$stage_log" "$OUT/$name.log" \
               && grep -q "REPORT$" "$OUT/$name.log"; then
                touch "$OUT/$name.done"
                echo "[$(date +%T)] OK    $name  ($(( SECONDS - t0 ))s)"
                rm -f "$stage_log"
            else
                echo "[$(date +%T)] COPY FAILED $name -- the run itself was fine."
                echo "        Complete log kept at $stage_log; not marking done."
                echo "        Free space on $OUT and rerun to copy it."
                return 1
            fi
        else
            echo "[$(date +%T)] TRUNCATED $name -- exit 0 but no report in the log."
            echo "        Not marking done; a rerun will redo this config."
            cp "$stage_log" "$OUT/$name.log" 2>/dev/null
            return 1
        fi
    else
        echo "[$(date +%T)] FAIL  $name  ($(( SECONDS - t0 ))s)"
        cp "$stage_log" "$OUT/$name.log" 2>/dev/null \
            && echo "        log copied to $OUT/$name.log" \
            || echo "        log kept at $stage_log (copy failed too)"
        tail -5 "$stage_log" | sed 's/^/        /'
    fi
    return 0
}

for s in "${SEEDS[@]}"; do
    echo ""
    echo "=================== SEED $s ==================="

    run "mw_s$s"     trading_reverie.py --fork base_trading --sim run07_mw_s$s \
                     --steps $STEPS --seed $s --temperature $TEMPERATURE --fresh   || break
    run "base_s$s"   trading_reverie.py --fork base_trading --sim run07_base_s$s \
                     --steps $STEPS --seed $s --temperature $TEMPERATURE --fresh --no-middleware || break
    run "pmw_s$s"    paired_eval.py --fork base_trading --sim run07_pmw_s$s \
                     --steps $STEPS --seed $s --temperature $TEMPERATURE --fresh --advance-with middleware || break
    run "pbase_s$s"  paired_eval.py --fork base_trading --sim run07_pbase_s$s \
                     --steps $STEPS --seed $s --temperature $TEMPERATURE --fresh --advance-with baseline || break

    # --- optional arms -----------------------------------------------------
    # Appended after the four core configs so that killing the sweep early
    # still leaves complete matched sets of the configs the headline claim
    # needs. These answer follow-up questions; the four above answer the paper.
    if [ "$RUN_ABLATIONS" -eq 1 ]; then
        # One stage each. The middleware changes context length, identity
        # information and the action interface together, so no effect can be
        # attributed to one of them without these. NOT the middleware: never
        # report an ablation arm's numbers as the system's.
        run "abl_comp_s$s"   trading_reverie.py --fork base_trading --sim run07_abl_comp_s$s \
                             --steps $STEPS --seed $s --temperature $TEMPERATURE --fresh \
                             --components compression || break
        run "abl_idrag_s$s"  trading_reverie.py --fork base_trading --sim run07_abl_idrag_s$s \
                             --steps $STEPS --seed $s --temperature $TEMPERATURE --fresh \
                             --components idrag || break
        run "abl_filter_s$s" trading_reverie.py --fork base_trading --sim run07_abl_filter_s$s \
                             --steps $STEPS --seed $s --temperature $TEMPERATURE --fresh \
                             --components filter || break
    fi

    if [ "$RUN_BASECAP" -eq 1 ]; then
        # Baseline told its numeric risk limit in prose. In run 06, 10 of 18
        # resized baseline orders were clamped for the risk cap, so some of the
        # baseline's resizing may be it breaking a limit it was never told.
        # This arm separates "did not know" from "ignored it".
        run "basecap_s$s"    trading_reverie.py --fork base_trading --sim run07_basecap_s$s \
                             --steps $STEPS --seed $s --temperature $TEMPERATURE --fresh \
                             --no-middleware --baseline-states-risk-cap || break
    fi

    echo "---- seed $s complete | $(( (SECONDS - START) / 60 )) min elapsed ----"
done

echo ""
echo "[$(date +%T)] SWEEP ENDED after $(( (SECONDS - START) / 60 )) min"
echo "configs completed: $(ls "$OUT"/*.done 2>/dev/null | wc -l) / $(( ${#SEEDS[@]} * CONFIGS_PER_SEED ))"
echo "complete seeds (the 4 core configs):"
for s in "${SEEDS[@]}"; do
    n=$(ls "$OUT"/{mw,base,pmw,pbase}_s$s.done 2>/dev/null | wc -l)
    [ "$n" -eq 4 ] && echo "  seed $s  COMPLETE" || echo "  seed $s  partial ($n/4)"
done
# Any log left in the stage never made it across. Said plainly, because a
# finished run whose log did not copy is recoverable and easy to miss.
remaining=$(ls "$STAGE"/*.log 2>/dev/null | wc -l)
if [ "$remaining" -gt 0 ]; then
    echo ""
    echo "WARNING: $remaining finished log(s) still in $STAGE -- the runs"
    echo "         completed but the copy to $OUT failed. They are NOT lost."
    ls "$STAGE"/*.log | sed 's/^/           /'
fi
echo ""
echo "Per-config detail: ./sweep_status.sh"
