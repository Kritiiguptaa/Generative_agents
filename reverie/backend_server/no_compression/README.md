# No-Compression 2-Agent Simulation

This folder provides a standalone runner that:

1. Forks a simulation (default: `base_test_hallucination` with 2 agents).
2. Runs agent simulation steps directly.
3. Computes hallucination each step using existing backend logic.
4. Does **not** call any compression pipeline.

## Run

From project root:

```bash
python reverie/backend_server/no_compression/run_no_compression_sim.py \
  --origin base_test_hallucination \
  --target no_compression_demo \
  --steps 20 \
  --expected-agents 2
```

## Outputs

Created under:

`environment/frontend_server/storage/<target>/`

Important files:

- `movement/<step>.json`: per-step movement and `hallucination_score` per agent
- `reverie/hallucination_analysis.json`: full hallucination history
- `reverie/hallucination_summary_no_compression.json`: compact summary

## Notes

- This runner auto-generates `environment/<next_step>.json` from movement output so it can progress without frontend driving each step.
- Compression utility (`reverie/compress_sim_storage.py`) is not used by this workflow.
