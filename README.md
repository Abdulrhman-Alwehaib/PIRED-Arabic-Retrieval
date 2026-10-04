# PIRED

**Beats multilingual-e5-base on the Arabic tasks neither model was trained on (48.5 vs 46.7 nDCG@10), at half the size.**

PIRED is an Arabic retrieval model: 134M parameters, built from scratch. It turns a query and a passage into vectors,
and the closer the two vectors are, the better the passage answers the query. It matches
`intfloat/multilingual-e5-base` (278M parameters) on Arabic retrieval at half the size.

## How PIRED was built

Each stage feeds the next one.

| # | Stage | Trained on | Output |
|---|---|---|---|
| 1 | Tokenizer | 640k Arabic documents | a 64k-token BPE tokenizer with Arabic normalization built in |
| 2 | MLM pretraining | 10.85B tokens of Arabic web text | an encoder that understands Arabic |
| 3 | Weak contrastive | 13M automatic query and passage pairs | a first search model |
| 4 | Supervised | 813k questions, 7 hard negatives each | a much better search model |
| 5 | Evaluation | | scores on every Arabic retrieval task in MTEB |

The released PIRED model mixes the stage 3 and stage 4 weights half and half (`stage4-blend-0.5`).

## Results

PIRED next to multilingual-e5-base (278M parameters) on all 11 Arabic retrieval tasks in MTEB, split into the tasks
neither model was trained on and the tasks they saw in training. Both models ran through the same evaluation code
(mteb 2.12.10). Each score is the average over the tasks in that group, multiplied by 100. Higher is better, and the
better score in each pair is in bold.

### Unseen tasks (7)

| | PIRED (134M) | multilingual-e5-base (278M) |
|---|---:|---:|
| nDCG@10 | **48.5** | 46.7 |
| Recall@100 | **74.9** | 73.6 |

XPQA (product questions), WebFAQ (FAQ pages from the web), PublicHealthQA (health questions), Sadeem (community
questions), MKQA (open-domain questions), MultiLongDoc (long documents) and Mintaka (entity questions). Neither model
was trained on any of them.

### Seen tasks (4)

| | PIRED (134M) | multilingual-e5-base (278M) |
|---|---:|---:|
| nDCG@10 | 67.4 | **70.1** |
| Recall@100 | 93.6 | **94.2** |

MIRACL, its hard-negative version and Mr.TyDi (Wikipedia search): both models were trained on their train splits.
MLQA (questions on Wikipedia): e5-base saw it through xP3, which is part of its training data. PIRED did not. MTEB
also lists a second version of the MIRACL hard-negative task, which gives identical scores for Arabic, so it is
counted once.

### All 11 tasks

| | PIRED (134M) | multilingual-e5-base (278M) |
|---|---:|---:|
| nDCG@10 | **55.4** | 55.2 |
| Recall@100 | **81.7** | 81.1 |

PIRED is also up to 40% faster than e5-base and needs half the memory.

## Running the code

```
pip install -e ".[dev]"
python -m pired tokenizer all
python -m pired pretraining corpus      then: train, probe
python -m pired contrastive data        then: selftest, evaluate-before, train, evaluate-after, report
python -m pired supervised queries      then: mine, train
python -m pired blend build
python -m pired evaluation run          or: speed
pytest
```

Add `--smoke` to any command for a small run that writes to `_smoke_tests/` and uploads nothing.

The code lives in `src/pired/`, one package per stage: `tokenizer`, `pretraining`, `contrastive`, `supervised`,
`blend` and `evaluation`, plus `encoder` (the model itself) and `common` (paths, Hub, checkpoints, logging). It loads
every saved model with `strict=True` and gives the same outputs as the original notebooks, bit for bit.

## Folders

```
project.json     folder paths used by the code and the notebooks
.env             secrets, never commit or share it
src/             the code
tests/           checks for every stage
notebooks/       the original notebooks, kept as they ran
stages/          one folder per stage: final_model/, checkpoints/, logs/, reports/, data/
downloads/       raw datasets from the Hub (89 GB, safe to delete)
_smoke_tests/    output of --smoke runs (safe to delete)
```

Each stage folder has its own README with the details of that stage.

## Data and models

Models, checkpoints and datasets are not in git. They live in private Hugging Face repos under
`Abdulrhman-Alwehaib/PIRED-*`: `tokenizer`, `pretraining-data`, `mlm-checkpoints`, `weak-pairs`,
`contrastive-checkpoints`, `contrastive-weak`, `supervised-queries`, `supervised-pairs`, `supervised-checkpoints`,
`supervised` and `evaluation`. The code downloads what it needs when it is missing locally.

## Rules

- Dev and test splits are never used for training. They are only used to remove overlapping queries and to report
  scores.
