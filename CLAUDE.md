# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**slime** is an LLM post-training framework for RL scaling that bridges Megatron (training) with SGLang (inference). It powers GLM-4.5 through GLM-5 and supports Qwen3, Qwen2.5, DeepSeek V3, and Llama 3 model families. Python 3.10+.

## Common Commands

```bash
# Install
pip install -e .

# Run tests (pytest, markers: unit/integration/system/acceptance)
pytest tests/
pytest tests/test_quick_start_qwen25_05B.py          # single test file
pytest -m "unit"                                       # by marker

# Code quality (pre-commit: ruff, black, isort, autoflake)
pre-commit run --all-files --show-diff-on-failure --color=always

# Training entry points
python train.py <args>        # synchronous training
python train_async.py <args>  # asynchronous training

# Model weight conversion (Megatron backend requires torch_dist format)
source scripts/models/qwen3-4B.sh  # load model config
PYTHONPATH=/path/to/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} --hf-checkpoint /path/to/hf --save /path/to/torch_dist
```

## Code Style

- **black** with line length 119
- **isort** with black profile
- **ruff** checks: E, F, B, UP (ignoring E402, E501)
- **autoflake** removes unused imports
- Pre-commit hooks enforce all of the above

## Architecture

The system follows a **decoupled training-rollout loop** orchestrated by Ray:

```
train.py / train_async.py
    │
    ├── Rollout (SGLang + router) ── generates completions, computes rewards
    │       └── slime/rollout/         (sglang_rollout, data_source, rm_hub, generate_hub)
    │
    ├── Training (Megatron or FSDP) ── RL training on rollout data
    │       └── slime/backends/        (megatron_utils/, fsdp_utils/)
    │
    └── Data Buffer ── bridges rollout → training
            └── slime/rollout/data_source.py
```

**Ray orchestration** (`slime/ray/`): `placement_group.py` allocates GPUs, `rollout.py` manages data generation, `train_actor.py` wraps training, `actor_group.py` groups Ray actors.

### Three-layer argument system (`slime/utils/arguments.py`)

1. **Megatron args** — passed through directly (e.g. `--tensor-model-parallel-size 2`)
2. **SGLang args** — prefixed with `--sglang-` (e.g. `--sglang-mem-fraction-static`)
3. **slime-specific args** — framework-level configuration

### Key extensibility points

- **Custom rollout**: implement `generate_rollout(args, rollout_id, data_source, evaluation=False)`
- **Custom reward models**: implement `custom_rm(args, sample) -> float` in `slime/rollout/rm_hub/`
- **Custom data conversion**: implement `convert_samples_to_train_data(args, samples) -> dict`
- **Model providers**: add model support via `slime_plugins/mbridge/`

## Package Layout

- `slime/` — main package
  - `backends/megatron_utils/` — Megatron training backend
  - `backends/fsdp_utils/` — FSDP training backend
  - `backends/sglang_utils/` — SGLang inference integration
  - `ray/` — Ray-based distributed orchestration
  - `rollout/` — data generation pipeline, reward models, filters
  - `utils/` — arguments, data, logging, misc utilities
- `slime_plugins/` — optional model bridges (mbridge, megatron_bridge)
- `tests/` — E2E and model-specific tests (multi-GPU, requires hardware)
- `scripts/` — training launch scripts and model conversion tools
- `scripts/models/` — model config scripts (source these before conversion/training)
- `examples/` — use-case examples (async training, knowledge distillation, multi-agent, etc.)

## Debugging

**Separate debugging flags** allow testing training/inference independently:
- `--debug-rollout-only` — skip Megatron, only initialize SGLang (test inference with fewer GPUs)
- `--debug-train-only` — skip SGLang, only initialize Megatron (test training)
- `--save-debug-rollout-data /path/data_{rollout_id}.pt` — save rollout results for replay
- `--load-debug-rollout-data /path/data_{rollout_id}.pt` — load fixed rollout data (implies `--debug-train-only`)
- `--check-weight-update-equal` — verify Megatron→SGLang weight sync correctness

**Validation checks** during first training step:
1. Rollout should be coherent (if not, check weight loading/conversion)
2. `log_probs` should equal `ref_log_probs` (KL=0 before any training)
3. With `num_steps_per_rollout == 1`, KL should be 0 and `grad_norm` small
