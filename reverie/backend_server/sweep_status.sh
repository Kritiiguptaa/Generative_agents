#!/usr/bin/env bash
# What state is every sweep config actually in?
#
# WHY THIS EXISTS: after run 07 there was no way to tell which configs had
# finished. `ls *.done` answers only one of the three questions that matter,
# because a config can be done, or truncated (the run exited 0 but the log
# holds no report, usually a write that failed during the final flush), or
# missing entirely (never started, or started and killed). Those need different
# responses, and a count of .done files cannot distinguish the second from the
# first.
#
# Reads only. It never writes a .done, because a tool that can mark a config
# complete can mark a truncated one complete.
#
# Usage, from reverie/backend_server:
#   ./sweep_status.sh                 # defaults to /workspace/run07
#   OUT=/workspace/run06 ./sweep_status.sh
#   ./sweep_status.sh --quiet         # summary counts only

set -u

OUT=${OUT:-/workspace/run07}
STAGE=${STAGE:-/tmp/sweep_stage}
SEEDS=${SEEDS:-"42 43 44 45"}

# Core configs first, then the optional arms. Listed rather than globbed so a
# config that was never started is reported as missing instead of being absent
# from the output, which is the whole point.
CORE_CONFIGS="mw base pmw pbase"
OPTIONAL_CONFIGS="abl_comp abl_idrag abl_filter basecap"

QUIET=0
for arg in "$@"; do
    case "$arg" in
        --quiet) QUIET=1 ;;
        *) echo "unknown argument: $arg"; echo "usage: $0 [--quiet]"; exit 2 ;;
    esac
done

if [ ! -d "$OUT" ]; then
    echo "No sweep output directory at $OUT."
    echo "Either the sweep never ran, or OUT points somewhere else."
    exit 1
fi

n_done=0
n_trunc=0
n_missing=0
n_stranded=0
n_running=0

# classify <name> -- print one config's state and bump the matching counter.
#
# The order of these checks matters. A .done is only trusted when the log also
# contains a report: run_sweep.sh will not write a .done without one, but an
# older sweep could have, and trusting the marker over the log would hide
# exactly the failure this script exists to surface.
classify () {
    local name="$1"
    local log="$OUT/$name.log"
    local done_marker="$OUT/$name.done"
    local state detail=""

    if [ -f "$done_marker" ] && [ -f "$log" ] && grep -q "REPORT$" "$log"; then
        state="done"
        n_done=$(( n_done + 1 ))
    elif [ -f "$done_marker" ] && { [ ! -f "$log" ] || ! grep -q "REPORT$" "$log"; }; then
        # A marker claiming a log that is absent or has no report. Worse than a
        # plain truncation, because a resume will SKIP this config.
        state="BAD-DONE"
        detail="marker present but log has no report: a resume will skip it"
        n_trunc=$(( n_trunc + 1 ))
    elif [ -f "$log" ] && grep -q "REPORT$" "$log"; then
        state="uncopied"
        detail="log has a report but no .done: rerun to mark it"
        n_trunc=$(( n_trunc + 1 ))
    elif [ -f "$log" ]; then
        state="truncated"
        detail="log present, no report ($(wc -l < "$log") lines)"
        n_trunc=$(( n_trunc + 1 ))
    elif [ -f "$STAGE/$name.log" ] && grep -q "REPORT$" "$STAGE/$name.log"; then
        state="stranded"
        detail="finished log still in $STAGE: the run completed, the copy failed"
        n_stranded=$(( n_stranded + 1 ))
    elif [ -f "$STAGE/$name.log" ] && pgrep -f -- "--sim [^ ]*_${name}( |\$)" >/dev/null; then
        # A staged log with no report is the normal state of the config in
        # flight. This used to be reported as "stranded ... the run completed",
        # which sent a healthy sweep's operator looking for a failed copy.
        state="running"
        detail="tail -f $STAGE/$name.log"
        n_running=$(( n_running + 1 ))
    elif [ -f "$STAGE/$name.log" ]; then
        state="killed"
        detail="no report and no live process: interrupted mid-run, a rerun redoes it"
        n_trunc=$(( n_trunc + 1 ))
    else
        state="missing"
        n_missing=$(( n_missing + 1 ))
    fi

    if [ "$QUIET" -eq 0 ]; then
        printf "  %-20s %-10s %s\n" "$name" "$state" "$detail"
    fi
}

echo "Sweep status: $OUT"
echo ""

for s in $SEEDS; do
    [ "$QUIET" -eq 0 ] && echo "seed $s"
    for c in $CORE_CONFIGS; do
        classify "${c}_s${s}"
    done
    # Only reported when something for them exists, so an unrequested optional
    # arm is not listed as 16 missing configs and does not drown the core ones.
    for c in $OPTIONAL_CONFIGS; do
        if [ -f "$OUT/${c}_s${s}.log" ] || [ -f "$OUT/${c}_s${s}.done" ] \
           || [ -f "$STAGE/${c}_s${s}.log" ]; then
            classify "${c}_s${s}"
        fi
    done
    [ "$QUIET" -eq 0 ] && echo ""
done

echo "done      : $n_done"
echo "running   : $n_running"
echo "incomplete: $n_trunc   (truncated, uncopied, killed, or a .done with no report)"
echo "stranded  : $n_stranded   (finished, log never copied off local disk)"
echo "missing   : $n_missing"
echo ""

if [ "$n_trunc" -gt 0 ] || [ "$n_stranded" -gt 0 ]; then
    echo "Rerunning run_sweep.sh will redo the incomplete configs and skip the"
    echo "done ones. Check free space first: an incomplete config is usually a"
    echo "write that failed, not a model that failed."
    echo "A config marked BAD-DONE will be SKIPPED by a resume; delete its"
    echo ".done file to force it to rerun."
fi
if [ "$n_done" -gt 0 ] && [ "$n_trunc" -eq 0 ] && [ "$n_missing" -eq 0 ] \
   && [ "$n_stranded" -eq 0 ]; then
    echo "All requested configs are complete."
fi
