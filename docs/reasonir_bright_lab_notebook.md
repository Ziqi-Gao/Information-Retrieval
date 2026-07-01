# ReasonIR-BRIGHT Lab Notebook

## Track Boundary

This is a separate experimental track from the autonomous MTEB goal workflow.
It must not read or update:

- `outputs/goal/state.json`
- `outputs/baselines/standard_frozen/`
- existing MTEB scoreboards
- existing autonomous-goal outputs

The output namespace is `outputs/reasonir_bright/`.

## Initial Goal

Re-run the original seven initial retrieval methods with:

- training data: `reasonir/reasonir-data`, config `hq`, split `train`
- evaluation data: `xlangai/BRIGHT`, config `examples` plus short-document config `documents`

The batch-local ReasonIR-trained `standard` run is the baseline. Loop methods are
compared against that baseline only.

## Dataset Schema Inspection

Inspected with:

```bash
source scripts/slurm_env.sh
"$PYTHON_BIN" -c 'from datasets import load_dataset; ...'
```

ReasonIR HQ:

- split: `train`
- rows: `100521`
- features: `query`, `pos`, `neg`
- `query`: sequence of strings, typically `[instruction, query]`
- `pos`: sequence of sequence of strings
- `neg`: sequence of sequence of strings

Observed HQ row shape:

```text
query = [instruction, query_text]
pos = [["", "camel_44852"]]
neg = [["", negative_text]]
```

The first 200 inspected positives were id-only in the first positive cell. The
new `ReasonIRRetrievalDataset` therefore resolves id-only positives through
`xlangai/BRIGHT`, config `documents`, using `documents.id -> documents.content`.
If any positive or negative cannot be resolved to text, the loader raises with a
schema/sample preview rather than silently training on ids.

BRIGHT:

- examples config: `examples`
- document config: `documents`
- dev domains for batch 001: `biology`, `economics`, `psychology`, `stackoverflow`
- example features: `query`, `reasoning`, `id`, `excluded_ids`, `gold_ids_long`, `gold_ids`, `gold_answer`
- document features: `id`, `content`

The evaluator uses only:

- `examples.query`
- `examples.gold_ids`
- `examples.excluded_ids`
- `documents.id`
- `documents.content`

It does not use `examples.reasoning` or `examples.gold_answer` as model input.

## Initial Batch

Manifest:

```text
experiments/batches/reasonir_bright_batch_001_dev.yaml
```

Versions:

1. `standard`
2. `loop_final_mean_pool`
3. `loop_matryoshka_mean_pool`
4. `loop_final_recurrent_mean_pool`
5. `loop_matryoshka_recurrent_mean_pool`
6. `loop_final_recurrent_no_memory`
7. `loop_matryoshka_recurrent_no_memory`

Training config:

```text
configs/reasonir_bright_dev.yaml
```

Key settings:

- `train_sample_size: 50000`
- `num_negatives: 1`
- `epochs: 1`
- `tmax: 10`
- `max_query_length: 256`
- `max_doc_length: 512`
- `tau: 0.05`

Loop candidates evaluate all loops for diagnostics, but predeclare loop 10 as
the candidate loop. No per-domain loop selection is allowed.

## Commands

Smoke-test ReasonIR parsing:

```bash
source scripts/slurm_env.sh
"$PYTHON_BIN" scripts/reasonir_data_smoke.py --samples 5 --num-negatives 1
```

Smoke-test BRIGHT evaluator on a tiny subset:

```bash
source scripts/slurm_env.sh
"$PYTHON_BIN" -m src.eval_bright \
  --checkpoint_dir <checkpoint> \
  --version standard \
  --domains biology \
  --use-long-documents false \
  --loop_idx 1 \
  --max_queries 3 \
  --max_docs 100 \
  --output_dir /tmp/reasonir_bright_eval_smoke
```

Dry-run batch submission:

```bash
source scripts/slurm_env.sh
"$PYTHON_BIN" scripts/reasonir_bright_submit_batch.py \
  experiments/batches/reasonir_bright_batch_001_dev.yaml \
  --dry-run \
  --submit-postprocess
```

Submit batch after smoke tests:

```bash
source scripts/slurm_env.sh
"$PYTHON_BIN" scripts/reasonir_bright_submit_batch.py \
  experiments/batches/reasonir_bright_batch_001_dev.yaml \
  --submit \
  --submit-postprocess
```

Check Slurm progress after submission:

```bash
squeue -u "$USER" -n reasonir_train,bright_eval,bright_postprocess
```

## Batch 001 Results

Batch:

```text
reasonir_bright_batch_001_dev
```

Evaluated short-document BRIGHT dev domains:

- `biology`
- `economics`
- `psychology`
- `stackoverflow`

The batch completed successfully. All seven runs produced complete
`ndcg_at_10` rows for the expected domains and loop indices.

The batch-local `standard` baseline macro over these four domains was:

```text
0.0147147443
```

Formal candidate interpretation uses only the predeclared loop for each method.
For loop methods in batch 001, that predeclared loop is `loop10`; other loop
rows are diagnostic only.

Best formal candidate:

```text
loop_matryoshka_mean_pool__loop10
macro_ndcg_at_10: 0.0181144453
mean_delta: +0.0033997011
min_delta: -0.0004924202
domains_non_regressing: 3 / 4
regressed_domain: biology
```

Protocol readout:

```text
valid completed batch, formal success = no
```

Reason: the best formal candidate improved macro NDCG@10 but regressed on
`biology`. This must not be reported as an all-domain BRIGHT success.

Diagnostic note:

```text
loop_matryoshka_mean_pool__loop1
mean_delta: +0.0054113316
min_delta: +0.0039253650
domains_non_regressing: 4 / 4
```

This is a diagnostic signal only because `loop1` was not the predeclared
candidate loop for `loop_matryoshka_mean_pool`.

## Batch 002 Remaining-Domain Evaluation

Manifest:

```text
experiments/batches/reasonir_bright_batch_002_remaining_domains.yaml
```

Purpose:

```text
Evaluate the eight BRIGHT short-document domains not covered by batch 001,
reusing the batch 001 checkpoints and not retraining.
```

Remaining domains:

- `aops`
- `earth_science`
- `leetcode`
- `pony`
- `robotics`
- `sustainable_living`
- `theoremqa_questions`
- `theoremqa_theorems`

The manifest uses:

```text
defaults.checkpoint_batch_id: reasonir_bright_batch_001_dev
```

This makes the submitter use checkpoints from batch 001 while writing batch 002
eval outputs into its own namespace. The same formal candidate rule applies:
loop methods predeclare `candidate_loop_indices: [10]`, and non-loop diagnostic
selection is not allowed for formal claims.

Submission status at handoff:

```text
submitted, pending
```

Postprocess is scheduled to run after all batch 002 evaluation jobs finish.
Do not interpret batch 002 until postprocess has written its collected results
and scoreboard.
