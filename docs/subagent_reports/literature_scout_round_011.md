# Literature Scout Round 011

## Scope

Read-only local evidence only. No web evidence, file edits, job submission, training, or eval reruns.

## Finding

Forcing FiQA qrels loading to Hugging Face config `default` is an infrastructure/data-loading repair, not a retrieval scoring semantics change, if scoped narrowly to qrels loading.

Local evidence:

- Installed versions: `mteb==1.39.7`, `datasets==2.19.2`.
- `FiQA2018` uses generic MTEB retrieval loading.
- MTEB generic retrieval loading explicitly loads corpus with config `corpus` and queries with config `queries`, but qrels with no config: `load_dataset(self.hf_repo_qrels)[split]`.
- Local FiQA cache contains `corpus`, `queries`, and `default` configs.
- The FiQA `default` cache schema is qrels: `query-id`, `corpus-id`, `score`, with `train`, `dev`, and `test` splits.
- The batch failure matches this exactly: a config-less qrels load fails when datasets sees multiple cached `mteb/fiqa` configs.

## Semantics Assessment

Changing only FiQA qrels loading from config-less `load_dataset("mteb/fiqa")` to `load_dataset("mteb/fiqa", "default")` preserves retrieval scoring semantics because it selects the intended qrels table already present in the dataset cache. It does not alter corpus text, query text, model embeddings, ranking scores, metric parsing, NDCG calculation, loop index, lexical hash rule, or candidate track.

## Alternatives And Risks

- Narrow FiQA-only qrels config fix: best for `batch_017_dev_repair`, low semantic risk.
- General qrels disambiguation: more robust, but broader and needs schema guards.
- Custom local FiQA task subclass: avoids monkey-patching MTEB internals, but adds more maintenance.
- Pin or upgrade MTEB/datasets: cleaner long-term, but risks changing multiple tasks.
- Cache cleanup: not recommended; corpus and queries configs are expected and can recreate ambiguity.

Risks: avoid metric/scoring changes, avoid broad unguarded defaults for every dataset, and keep the repair repo-local rather than patching site-packages.
