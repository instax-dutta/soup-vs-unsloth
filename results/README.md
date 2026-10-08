# Results artifacts

Raw artifacts behind [REPORT.md](../REPORT.md), produced 2026-10-01/02 on the 2x Tesla T4
box described there. Peak VRAM values below are `max(used_mib)` from each GPU sample CSV
and match the numbers in the report.

## logs/

Full Soup/Unsloth console output per arm, with ANSI control sequences stripped
(`*.clean.log`). Data-prep scripts print row counts and sha256 hashes at the top of the
Soup logs; dataset files themselves lived on the benchmark VM and are reproducible via
[`scripts/`](../scripts).

| File | Arm | Outcome |
|---|---|---|
| `soup_hf_pure.clean.log` | Soup, `backend: transformers`, unsloth import blocked | GPU OOM at step 1 |
| `soup_hf_plain.clean.log` | Soup, `backend: transformers`, unsloth importable | completed (silently unsloth-backed) |
| `soup_unsloth.clean.log` | Soup, `backend: unsloth` | completed |
| `unsloth_native.clean.log` | Unsloth native script, no Soup | completed |
| `eval.log` | neutral evaluator | adapter health + held-out loss |

## gpu/

nvidia-smi samples at 2 s intervals, written by [`scripts/sample_gpu.sh`](../scripts/sample_gpu.sh):
`ts,used_mib,total_mib,util_pct`.

| File | Peak |
|---|---|
| `soup_hf_pure_gpu.csv` | 14,817 MiB (then OOM) |
| `soup_hf_plain_gpu.csv` | 14,867 MiB |
| `soup_unsloth_gpu.csv` | 7,109 MiB |
| `unsloth_native_gpu.csv` | 7,111 MiB |

## trainers/

- `unsloth_native.json` - `run_unsloth.py` result: step timings, losses, parameter counts,
  peak allocator stats.
- `hf_plain_trainer_state.json`, `unsloth_trainer_state.json` - TRL trainer state dumps.
- `hf_plain_adapter_config.json`, `unsloth_adapter_config.json` - saved LoRA configs.

## environment/

- `hardware.csv` - GPU / driver table.
- `versions.txt` - library versions recorded at run time.
- `pip_freeze.txt` - full environment freeze.

## figures/

- `vram.png` - GPU memory over wall time for all four arms, rendered by
  [`scripts/make_figure.py`](../scripts/make_figure.py) from `gpu/`.
