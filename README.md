# MagicBench

[Project Page](https://hsiung.cc/MagicBench/) | [Dataset](https://huggingface.co/datasets/hsiung/MagicBench)

MagicBench is a deception-sensitive cognitive benchmark for large language models built around magic-trick understanding. Instead of testing recall alone, it evaluates whether a model can reason about hidden causes, audience beliefs, violated expectations, and counterfactual changes in scenarios where the visible events are intentionally misleading.

The benchmark is inspired by cognitive abilities often discussed in AGI evaluation frameworks, including perception, attention, memory, reasoning, metacognition, social cognition, executive function, and transfer.

## Why magic tricks?

Magic tricks are useful evaluation cases because they separate:

- what is observed
- what the audience believes
- what is actually happening

That makes them a compact way to probe whether a model can track deception, infer plausible hidden mechanisms, and stay calibrated about uncertainty.

## What the benchmark contains

MagicBench currently defines:

- 50 magic scenarios
- 6 task types per scenario
- 300 total benchmark items per run

Each scenario includes:

- an audience-facing description of the effect
- the relevant violated expectations
- an abstract gold explanation of the method
- a belief trace contrasting audience belief vs. reality
- counterfactual variants
- difficulty annotations
- primary cognitive faculties

## Task types

Each scenario generates one item for each of the following task types:

1. `effect_recognition`
  Identify the type of effect the audience experienced.
2. `violation_identification`
  Determine which expectations or rules appear to be violated.
3. `best_explanation`
  Choose the most plausible hidden method.
4. `belief_trace`
  Infer what the audience believes at a specific moment.
5. `metacognitive_calibration`
  Distribute confidence across competing explanations.
6. `counterfactual_reasoning`
  Judge whether the method would still work under a changed condition.

## Cognitive profile

Scores are aggregated into five benchmark dimensions:

- `recognition`
- `causal_inference`
- `deception_modeling`
- `metacognitive_calibration`
- `transfer_robustness`

The script also maps task performance onto a broader faculty profile and reports performance by difficulty axis and trick family.

The reported `overall` score is a weighted average across task types:

- `effect_recognition`: 10%
- `violation_identification`: 10%
- `best_explanation`: 20%
- `belief_trace`: 25%
- `metacognitive_calibration`: 10%
- `counterfactual_reasoning`: 25%

## Repository layout

```text
.
├── magicbench.py                 # benchmark loader, task generation, scoring, CLI
└── scripts/
    ├── api.sh                    # local API keys (gitignored)
    └── run.sh                    # local helper script

```

## Requirements

- Python 3.9+ recommended
- `datasets` is required to load `hsiung/MagicBench` from Hugging Face
- An API key is needed for live model evaluation with supported providers

## Quick start

Clone the repo and run from the project root:

```bash
pip install datasets
python magicbench.py --help
```

### Dataset download

`magicbench.py` loads scenarios from the Hugging Face dataset `hsiung/MagicBench`.
The first run will download the dataset into the local Hugging Face cache automatically.

### Local secret file

For local runs, store API credentials in `scripts/api.sh`.

Example:

```bash
#!/bin/bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

### OpenAI example

```bash
source scripts/api.sh
python magicbench.py \
  --model gpt-4o \
  --provider openai \
  --api-key "$OPENAI_API_KEY"
```

### Dry run

Print generated prompts without calling any API:

```bash
python magicbench.py --dry-run
```

### Human baseline mode

Run the benchmark interactively as a person:

```bash
python magicbench.py --human
```

## CLI options

```text
usage: magicbench.py [-h] [--model MODEL] [--provider {anthropic,openai}]
                     [--api-key API_KEY] [--n-repeats N_REPEATS]
                     [--delay DELAY] [--dry-run] [--human]
                     [--seed SEED] [--output-dir OUTPUT_DIR]
```

Notable flags:

- `--model`: model name to send to the provider API
- `--provider`: currently `anthropic` or `openai`
- `--n-repeats`: repeat each task multiple times
- `--delay`: sleep between API calls
- `--seed`: controls randomized option ordering and sampled task variants
- `--output-dir`: directory for results artifacts

## Output files

Each run writes timestamped artifacts to `results/` by default:

- `*_results.json`: per-item responses and scores
- `*_profile.json`: aggregated benchmark profile, faculty mapping, and difficulty analysis
- `*_report.txt`: human-readable report

## Citation
If you find our work helpful or inspiring to your research, please cite our project as follows:
```
@misc{hsiung2026magicbench,
  title={{MagicBench: A Deception-Sensitive Cognitive Benchmark for LLMs}},
  author={Hsiung, Lei},
  year={2026},
  howpublished={\url{https://hsiung.cc/magicbench/}},
}
```
