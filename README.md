# Genetic Algorithm vs. Random Search for Configuring Production RAG Retrieval

**Course:** CECS 385, Fall 2026
**Author:** William Armentrout

A controlled computational experiment testing whether a genetic algorithm (GA) finds retrieval-augmented generation (RAG) configurations closer to the best achievable than random search, at an equal evaluation budget.

---

## 1. Research Question and Hypothesis

**RQ:** Does a GA find RAG configurations closer to the best achievable than random search at an equal evaluation budget?

**Hypothesis:** At an equal evaluation budget, a GA's best-found configuration will have a smaller gap to the best achievable score than random search's.

| Variable | Definition |
|---|---|
| Independent | Search method (GA or random search) |
| Dependent | Gap = (best score in the full grid) − (best score found by the method) |
| Held constant | Evaluation budget, search space, question set, fitness function |

## 2. Background (one paragraph)

No single RAG configuration is best: index type, chunking strategy, and retrieval settings trade speed, recall, and cost differently, and a configuration tuned on one task generalizes poorly to another. Every trial needs new embeddings and evaluation, so tuning is expensive. Random search is the standard baseline adaptive methods should beat (Bergstra and Bengio, 2012). GAs are established for neural architecture search, and existing RAG tuners (AutoRAG, AutoRAG-HP) use greedy and bandit approaches. None of the sources found in the literature survey applied a GA to RAG configuration or compared it with random search. Full references are in the proposal (`docs/`).

## 3. Experimental Design

### Three conditions

| Condition | Role |
|---|---|
| Exhaustive grid search | Ground truth. Scores every configuration once and fills the lookup table. |
| Random search | Baseline |
| Genetic algorithm (LEAP) | Method under test |

### Key design choices

- **Dataset:** WixQA enterprise RAG benchmark (questions with labeled relevant passages). Follow its licensing terms for use, attribution, and redistribution.
- **Search space:** about 500-1,000 configurations. Genes: chunk size, chunk overlap, retrieval type, reranker toggle, top-k. Index type is fixed (the proposal lists it as a gene; the presentation fixes it, so this README follows the presentation).
- **Invalid configurations:** handled identically in both methods. For example, overlap >= chunk size is resampled.
- **Fitness:** nDCG@k, retrieval only. Relevance is defined once, via overlap with the labeled passages. Generated answers are not evaluated.
- **Lookup table:** every configuration is scored once up front. GA and random-search runs are then table lookups, which makes hundreds of seeded runs cheap and lets variance be reported.
- **Hold-out split:** questions are split into tuning and test sets. Configurations are chosen on tuning questions and scored on test questions.
- **Budget:** number of unique configurations evaluated, with revisits counted once, applied identically to both methods. Budgets are fixed in advance at 5%, 10%, and 20% of the space.
- **Seeds:** multiple seeded runs per method per budget. The count is set after timing a run of each method.
- **GA settings:** fixed before looking at any comparison results, chosen from prior LEAP lab experience, and not fine-tuned.

### Analysis

- Convergence curves: best-so-far gap vs. unique configurations evaluated
- Mean and median gap at each budget, with confidence intervals
- Mann-Whitney U test comparing GA and random search at each budget
- Test-set score of each method's chosen configuration, to check for overfitting to the tuning questions

## 4. Suggested Project Structure

```
ga-rag-tuning/
├── README.md
├── requirements.txt              # pinned dependencies (leap-ec, numpy, pandas, scipy, ...)
├── .gitignore                    # data/raw, large indexes, embedding caches
│
├── config/
│   ├── search_space.yaml         # values per gene; defines the full grid
│   ├── ga_params.yaml            # population, generations, mutation, crossover, selection (frozen before comparison)
│   └── experiment.yaml           # budgets [0.05, 0.10, 0.20], seeds, nDCG k, split ratio, split seed
│
├── data/
│   ├── raw/                      # WixQA download (not committed)
│   ├── processed/                # cleaned corpus, tuning/test question splits
│   └── DATA_LICENSE.md           # WixQA terms, attribution, redistribution notes
│
├── src/ragtune/
│   ├── data.py                   # load WixQA, build and save the tuning/test split
│   ├── space.py                  # Config dataclass, validity rules, encode/decode, resampling of invalid configs
│   ├── pipeline/
│   │   ├── chunking.py           # chunk size and overlap
│   │   ├── indexing.py           # embeddings plus the fixed index type
│   │   ├── retrieval.py          # retrieval types and top-k
│   │   └── rerank.py             # reranker toggle
│   ├── metrics.py                # nDCG@k against labeled passages
│   ├── grid.py                   # exhaustive scoring, resumable, writes the lookup table
│   ├── lookup.py                 # table access plus unique-evaluation budget counter
│   ├── search/
│   │   ├── random_search.py      # baseline
│   │   └── ga.py                 # LEAP-based GA over the same space and lookup
│   ├── runner.py                 # runs one (method, budget, seed) and logs best-so-far per evaluation
│   └── analysis/
│       ├── gaps.py               # gap to best, mean/median, confidence intervals
│       ├── stats.py              # Mann-Whitney U
│       └── plots.py              # convergence curves, per-budget summaries
│
├── scripts/                      # run in order
│   ├── 01_prepare_data.py
│   ├── 02_build_lookup_table.py  # the expensive step, run once
│   ├── 03_run_experiments.py
│   └── 04_analyze.py
│
├── results/
│   ├── lookup_table/             # configs x (tuning nDCG, test nDCG), with a metadata file
│   ├── runs/                     # per-run logs: method, budget, seed, trajectory
│   ├── tables/
│   └── figures/
│
├── notebooks/                    # exploration and sanity checks only
├── tests/                        # validity rules, budget counting, seed reproducibility, metric correctness
│
└── docs/
    ├── proposal.pdf
    ├── presentation.pdf
    ├── ethics_notes.md           # grid compute and energy log, licensing, limitations
    └── report/                   # draft report and peer review material
```

## 5. Workflow

1. **Prepare data.** Download WixQA, build the corpus, and create the tuning/test split with a fixed split seed.
2. **Define the space.** Finalize gene values in `config/search_space.yaml` so the full grid lands at about 500-1,000 valid configurations.
3. **Build the lookup table.** Score every configuration once, with both tuning and test nDCG recorded. This is the single exhaustive run, so make it resumable and log its runtime and energy use.
4. **Validate before searching.** Spot-check lookup values against a direct pipeline run, and unit-test the budget counter and invalid-config handling.
5. **Freeze GA settings.** Commit `ga_params.yaml` before running any comparisons.
6. **Run experiments.** Every (method, budget, seed) combination, logging best-so-far after each unique evaluation.
7. **Analyze.** Gaps, convergence curves, confidence intervals, Mann-Whitney U, and held-out test scores.
8. **Write up.** Results, limitations, and ethics discussion.

## 6. Reproducibility Rules

- Every run is identified by (method, budget, seed) and is deterministic given its seed.
- The split seed, GA parameters, and budgets live in `config/`, never hard-coded.
- The lookup table ships with metadata: embedding model, library versions, k, split seed, and date built.
- Random search and the GA call the same lookup and the same budget counter, so the fairness rule lives in one place.

## 7. Ethical Considerations

| Concern | Mitigation |
|---|---|
| Energy and carbon cost of the exhaustive grid, the costliest method | Run it once, reuse the lookup table for all later runs, and log its compute use |
| WixQA licensing | Follow its terms on use, attribution, and redistribution |
| Wrong answers in production (high nDCG is not a correct answer) | State that human review is still recommended |
| Overfitting to the tuning questions | Select on tuning questions and score chosen configurations on the held-out test set |

## 8. Limitations

- **Scope:** one dataset (WixQA), retrieval only. Answer quality is not measured.
- **Size:** a small, exhaustively searchable space, so results may not scale to larger ones.
- **Cost model:** the lookup table hides real per-trial latency and cost.
- **GA settings:** taken from prior lab experience, not tuned.

## 9. Expected Contribution

- Evidence on how much search cost a GA saves versus random search, and what it costs in quality
- A reusable protocol: a pre-scored lookup table plus seeded runs
- A useful result even if the GA loses, since a null result tells practitioners random search is enough here

## 10. Open Decisions

- Exact gene values and the resulting grid size
- Embedding model, reranker model, vector store, and the list of retrieval types
- The value of k for nDCG@k, which should stay fixed and independent of the top-k gene
- Which score defines "best achievable" for the gap (tuning, test, or both reported)
- Number of seeds per method and budget, set after timing a run
- Whether index type stays fixed (presentation) or becomes a gene (proposal)

## 11. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/01_prepare_data.py
```
