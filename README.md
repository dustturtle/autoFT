# autofinetune

**Autonomous LoRA fine-tuning, driven by an AI agent.**

You describe a goal. Claude sets up the experiment, runs a baseline, then loops autonomously — tweaking hyperparameters, training, measuring, keeping what improves, discarding what doesn't. You come back to a results table and the best adapter.

~12 experiments/hour. ~100 overnight. You sleep, it researches.

Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for LoRA fine-tuning with [Unsloth](https://github.com/unslothai/unsloth).

---

## How It Works

```
  You: "Train a 4B model for sentiment analysis"

  Claude:
    1. Creates a git branch for this goal
    2. Writes goal.md (what to optimize, what data, what metric)
    3. Configures config.py (model, LoRA settings, data paths)
    4. Runs baseline experiment
    5. Enters the loop:

        ┌──────────────────────────────────────┐
        │                                      │
        │   Edit config.py → commit → train    │
        │         │                            │
        │         ▼                            │
        │   Metric improved?                   │
        │     yes → keep commit, log result    │
        │     no  → git reset, discard         │
        │         │                            │
        │         └────── repeat forever ──────┘
        │                                      │
        └──────────────────────────────────────┘

    6. You check results.tsv whenever you're ready
```

Each training run is time-budgeted (default 5 minutes). The agent commits config changes before each run. Good experiments advance the branch. Bad ones get reset. The branch tip is always the best configuration found so far.

---

## Quick Start

```bash
git clone https://github.com/ajbmachon/autofinetune
cd autofinetune
uv sync
```

Then, in a Claude Code session from this directory:

```
You: "New goal: sentiment analysis using Qwen3-4B"
```

Claude reads `program.md`, creates a branch, sets everything up, and starts experimenting.

---

## Preparing Your Data

The harness auto-detects three data formats from file content. Put your data in `.jsonl` files (one JSON object per line).

### Chat format (recommended)

Best for most tasks. Matches how chat models are trained:

```jsonl
{"messages": [{"role": "system", "content": "Classify the sentiment. Output JSON only."}, {"role": "user", "content": "Classify:\n\nThis product exceeded my expectations, absolutely love it!"}, {"role": "assistant", "content": "{\"sentiment\": \"positive\", \"confidence\": \"high\"}"}]}
{"messages": [{"role": "system", "content": "Classify the sentiment. Output JSON only."}, {"role": "user", "content": "Classify:\n\nShipping took forever and the box arrived damaged."}, {"role": "assistant", "content": "{\"sentiment\": \"negative\", \"confidence\": \"high\"}"}]}
```

### Alpaca format

For instruction-following tasks:

```jsonl
{"instruction": "Summarize this article", "input": "The quarterly earnings report shows...", "output": "Revenue increased 12% YoY..."}
```

### Completion format

For raw text continuation:

```jsonl
{"text": "The mitochondria is the powerhouse of the cell. It produces ATP through..."}
```

Also supports `.json` (array of objects) and `.parquet` files.

### Data tips

- **Quality over quantity.** 500 clean, consistent examples often beat 10,000 noisy ones.
- **Hold out 10-15%** for validation. The agent uses eval loss (and optional custom metrics) to decide what works.
- **Check your labels.** Inconsistent labeling hurts small models disproportionately.
- **Profile your sequence lengths.** The agent can experiment with `max_seq_length`, but starting close to your data's p95 length saves wasted exploration.

---

## Project Structure

5 files. 1 editable. That's the whole thing.

```
autofinetune/
│
│   The agent reads these to understand the system:
├── program.md            Agent instructions — the experiment loop, rules, crash recovery
├── goal.md               Goal definition — objective, data, metrics, constraints (per-branch)
│
│   The agent edits this:
├── config.py             Experiment configuration — every knob the agent can turn
│
│   Fixed infrastructure (never modified during experiments):
├── train.py              Training harness — load model, LoRA, train, eval, save, print summary
├── prepare.py            Utilities — data loading, time budget, adapter merging
│
│   Generated during experiments:
├── results.tsv           Experiment log (untracked by git)
├── run.log               Latest training output (gitignored)
├── adapters/             Saved LoRA adapters, one per experiment (gitignored)
│
│   Reference:
├── docs/
│   ├── guide.md          Full architecture and usage guide
│   ├── project-overview.html  Visual architecture/usage overview
│   └── research/         Research notes on models, fine-tuning approaches
├── pyproject.toml        Dependencies
└── LICENSE               MIT
```

### Why so few files?

The agent needs to understand the entire codebase to make good decisions. 5 files fit in context. 50 files don't. Autoresearch's power is simplicity — we keep it.

---

## Key Files to Read

If you want to understand the system, read these in order:

1. **[`program.md`](program.md)** — The agent's operating manual. Defines the experiment loop, what can and can't be edited, and how results are tracked. Start here to understand the overall flow.

2. **[`goal.md`](goal.md)** — An example goal definition (sentiment analysis). Each goal branch gets its own. This is where you define what metric to optimize, where the data lives, and what constraints apply.

3. **[`config.py`](config.py)** — The agent's canvas. Every experiment is a change to this file. It's Python (not YAML), so the agent can define custom eval functions and data formatters inline.

4. **[`docs/guide.md`](docs/guide.md)** — Full walkthrough of the architecture, data formats, replay buffers, custom evaluation, branching strategy, and hardware requirements.
5. **[`docs/project-overview.html`](docs/project-overview.html)** — A local visual overview of the core architecture, design ideas, and usage flow.

### Key files to modify

- **`goal.md`** — Write this with Claude when starting a new goal. Defines the objective, data, and evaluation criteria.
- **`config.py`** — Set initial values when starting a goal. The agent takes over from there.

### Files you should never touch

- **`train.py`** and **`prepare.py`** — Fixed infrastructure. The agent doesn't touch them either.

---

## Branching Strategy

Each goal gets its own branch. Multiple goals can run concurrently (in separate sessions):

```
main                                ← Infrastructure
├── autofinetune/sentiment-4b       ← Sentiment analysis with Qwen3-4B
├── autofinetune/sentiment-0.6b     ← Same task, smaller model
├── autofinetune/summarizer         ← Document summarization
└── autofinetune/code-review        ← Code review comment generation
```

---

## What the Agent Can Experiment With

Everything in `config.py` is a knob. The agent decides what to try based on past results:

| Category | Knobs |
|----------|-------|
| **Model** | `model_name`, `max_seq_length` |
| **LoRA** | `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules`, `lora_bias` |
| **Training** | `learning_rate`, `lr_scheduler_type`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `warmup_ratio`, `weight_decay`, `optim`, `num_train_epochs`, `seed` |
| **Data** | `train_dataset`, `eval_dataset`, `dataset_format`, `formatting_func` |
| **Evaluation** | `eval_custom_func` (define any metric — accuracy, F1, ROUGE, anything) |
| **Forgetting** | `replay_dataset`, `replay_ratio` (mix general data to prevent catastrophic forgetting) |

Since `config.py` is Python, the agent can also define functions:

```python
def eval_custom(model, tokenizer, eval_dataset_path):
    """Agent writes whatever evaluation logic the goal needs."""
    # Run inference, parse outputs, compute metrics...
    return {"eval_accuracy": 0.91, "eval_f1": 0.88}

CONFIG = {
    ...
    "eval_custom_func": eval_custom,
}
```

---

## After the Experiments

### Check results

```bash
cat results.tsv
```

```
commit   eval_loss  metric  memory_gb  status   description
a1b2c3d  1.2345    0.8750  12.3       keep     baseline
b2c3d4e  1.1890    0.9100  12.5       keep     increase LR to 5e-4
c3d4e5f  1.3200    0.8200  12.3       discard  LoRA rank 8
d4e5f6g  0.0000    0.0000  0.0        crash    batch size 32 (OOM)
```

### Merge the best adapter into a deployable model

```bash
uv run prepare.py --merge adapters/<best_commit_hash> --output ~/models/my-model
```

This produces a full model (base + LoRA merged) ready for inference with any framework.

---

## Hardware

Built for NVIDIA GB10 Spark (128GB unified, Blackwell), but works on any CUDA GPU:

| Model | Training Memory | Fits on 24GB GPU? |
|-------|----------------|-------------------|
| 0.8B  | ~10GB          | Yes               |
| 2B    | ~13GB          | Yes               |
| 4B    | ~17GB          | Yes               |
| 9B    | ~30GB          | No                |
| 27B   | ~63GB          | No                |

The time budget defaults to 5 minutes per experiment. Override with `TIME_BUDGET=600 uv run train.py` for larger models.

---

## Requirements

- Python 3.10+
- CUDA GPU
- [uv](https://github.com/astral-sh/uv) package manager
- [Claude Code](https://claude.ai/claude-code) (or any AI coding agent that can read program.md and follow the loop)

---

## Attribution

This project is inspired by and adapted from [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch) (MIT license). The autonomous experiment loop, `program.md` agent instruction pattern, `results.tsv` logging, and keep/discard branching strategy all originate from autoresearch. We adapt these patterns from pretraining-from-scratch to LoRA fine-tuning of existing models with [Unsloth](https://github.com/unslothai/unsloth).

## License

[MIT](LICENSE)
