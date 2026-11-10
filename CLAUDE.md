# Project context

CECS 385 final project: GA vs. random search for configuring production RAG
retrieval. See README.md for the full design and file structure.

## Decisions made
- Dataset: WixQA (Hugging Face `Wix/WixQA`). Configs: `wix_kb_corpus`,
  `wixqa_expertwritten`, `wixqa_simulated`. Raw data in `data/raw/` (gitignored).
- Relevance is labeled at the ARTICLE level (`article_ids` match corpus `id`).
  Every question has 1-3 labels, all present in the corpus.
- Keep all `article_type`s (feature requests are 1/3 of the corpus but only
  10 of 340 labeled articles; they act as realistic distractors).
- Questions: pool expert + simulated (400), split 50/50 into tuning/test,
  stratified by source. Seed and ratio come from `config/experiment.yaml`.
  Splits saved to `data/processed/splits.json` (built by
  `scripts/01_prepare_data.py`).
- `splits.json` holds question IDs only (`expert_000`..`expert_199`,
  `simulated_000`..`simulated_199`, the row index in each raw config), never
  question text, so it can be committed. Load records with
  `ragtune.data.load_questions(raw_dir, splits_path, "tuning" | "test")`.
- Fitness: article-level nDCG@5, retrieval only (`config/experiment.yaml`
  `fitness`). Pipeline per question: retrieve top_k chunks (hybrid = RRF,
  constant 60, of BM25 top_k and dense top_k, keep top_k fused) -> optional
  rerank of those chunks -> map chunks to articles by first occurrence,
  drop duplicates -> binary nDCG@5, ideal DCG = min(len(article_ids), 5)
  hits. k counts articles; the top_k gene counts chunks. Scoring at chunk
  level is rejected: it favors small chunks and makes ideal DCG vary by
  chunking.
- Search space (`config/search_space.yaml`): chunk_size {128,192,256,384,512}
  tokens, chunk_overlap {0,32,64,128} tokens, retrieval_type
  {bm25,dense,hybrid}, reranker {off,on}, top_k {5,10,20,30,50}. 600
  combinations, 570 valid (rule: overlap < size). Budgets 28 / 57 / 114.
- Expected plateaus: bm25/dense with no reranker score the same for every
  top_k large enough to fill 5 articles. Fair to both methods; note it in
  the write-up.
- Models (fixed): embeddings `BAAI/bge-small-en-v1.5` (512-token cap, which
  caps chunk_size; queries use the bge search prefix), reranker
  `cross-encoder/ms-marco-MiniLM-L-6-v2`. Chunk sizes are measured with the
  embedding model's tokenizer. Hardware: RTX 2080 Ti (11 GB).
- Chunking (`src/ragtune/pipeline/chunking.py`): chunk_size = total tokens
  the embedder sees INCLUDING [CLS]/[SEP] (content = size - 2, so 512 fits
  bge with no truncation); overlap in content tokens. Boundaries snap to word
  starts (end back, next start forward), so actual overlap <= configured.
  Text sliced from the original by char offsets. `contents` chunked as-is
  (starts with the title); title not prepended to later chunks.
  Full corpus: 9.8k chunks (512/0) to 46k (128/64); 73.7M tokens to embed
  across all 19 chunkings; ~2 s to chunk per setting.
- Vector store: plain NumPy exact search (normalized embeddings, inner
  product). No ANN, so the lookup table has no approximation noise.
- BM25: `bm25s`, lowercase + English stopwords, no stemming, lucene
  k1=1.5 b=0.75. Scored over all chunks, then the shared top-k.
- Indexing (`src/ragtune/pipeline/indexing.py`): embeddings in fp32 (fp16
  was 2.5x faster but fp32 takes only ~30 s per chunking on the GPU).
  GPU embedding is not bit-identical across runs (~1e-7), so passage
  embeddings are cached once per chunking in `data/cache/` (gitignored),
  keyed by a chunk fingerprint + model settings; scoring from the cache is
  exact. `topk_stable` (ties -> lower chunk index) is used by both indexes,
  so smaller top_k lists are exact prefixes of the top-50 list.
- Retrieval (`src/ragtune/pipeline/retrieval.py`): `rank_questions` searches
  both indexes once to depth 50 per (chunking, question set); `first_stage`
  derives any (retrieval_type, top_k) from that. Hybrid fuses bm25 top_k +
  dense top_k (not the full depth) with RRF; fused ties -> lower chunk index.
- Sanity numbers (tuning set, nDCG@5, no rerank, top_k 5/10/20/30/50):
  512/0: bm25 .344 .349 .349 .349 .349 | dense .504 .517 .517 .517 .517 |
  hybrid .474 .499 .507 .515 .514.
  128/0: bm25 .306 .322 ... | dense .432 .443 ... | hybrid .412 .430 .443
  .453 .446. bm25/dense plateau from top_k=10 as predicted; hybrid keeps
  varying and beats dense at 128/0 top_k=30, so there are real interactions.
- Rerank (`src/ragtune/pipeline/rerank.py`): per (chunking, question set),
  score the union of bm25 + dense depth-50 rankings once (~85 pairs per
  question; covers every first-stage list incl. hybrid). Raw logits
  (identity activation, no sigmoid ties). Rerank = stable sort of the
  first-stage list by score (ties keep first-stage order); it only
  reorders, never adds chunks. Scores cached in `data/cache/rerank/`,
  keyed by chunk/question/candidate fingerprints + settings.
  Speed: 588 pairs/s at 512/0, 1,938 at 128/0 (~10 min for the full grid).
- Known limitation: at chunk_size 512, 46% of (question, chunk) pairs
  exceed MiniLM's 512 limit and lose ~10-20 tokens off the chunk tail.
  No truncation at 384 or below.
- Sanity numbers with rerank ON (tuning, top_k 5/10/20/30/50):
  512/0: bm25 .376 .443 .473 .470 .474 | dense .482 .506 .498 .493 .481 |
  hybrid .478 .510 .491 .481 .476.
  128/0: bm25 .302 .379 .402 .402 .409 | dense .444 .459 .449 .430 .422 |
  hybrid .430 .460 .445 .423 .416.
  The reranker lifts bm25 a lot but HURTS dense at most top_k: MiniLM
  (trained on short MS MARCO passages) is weaker than bge here. Best seen
  so far is dense, no rerank, 512/0 (.517).
- Grid (`src/ragtune/grid.py`, `scripts/02_build_lookup_table.py`): loops
  over the 19 chunkings; per chunking writes
  `results/lookup_table/parts/<cs_ov>.json` (gitignored) with per-question
  scores, runtime and GPU Wh. Parts are reused only if their `run_key`
  (models + versions, bm25/rrf/k/depth, question + corpus fingerprints,
  genes) matches. Code changes are NOT in the run_key: after changing
  pipeline code, rebuild with `--force`. `--only 512/0 128/0` builds a subset
  without merging. Query embeddings are cached too (`data/cache/queries/`).
  Outputs: `lookup_table.csv` (key, genes, tuning_ndcg, test_ndcg, in
  `space.valid_configs()` order), `per_question.npz` (n_configs x
  n_questions, for paired stats), `metadata.json` (versions, settings,
  split, git commit, runtime, GPU energy, best config).
- Lookup table BUILT (2026-11-10): 570 configs, 22.0 min grid compute,
  80.9 Wh GPU. Rebuilding parts from cache reproduces per-question scores
  exactly. (cs512_ov0 and cs128_ov0 parts reused embeddings/tuning rerank
  scores cached during smoke tests, so their logged time/energy is ~1 min /
  ~3 Wh low.)
- Landscape: tuning nDCG .281-.517 (465 distinct values). Optimum is a
  4-way tie: cs512_ov0_dense_rr0_k{10,20,30,50} = .5168 (test .4938). 14
  configs within .01 of best, 40 within .02 (7% of space), so absolute gaps
  will be small; report hit-rate of the optimum and rank-based gaps too.
  Tuning/test Spearman .857; the best-tuning config ranks 20th on test
  (best test: cs384_ov0_dense_rr1_k20 = .518). Gene means: bigger chunks
  better; dense ~ hybrid >> bm25; reranker helps on average (.454 vs .424)
  but the optimum has it off.
- Energy (`src/ragtune/energy.py`): nvidia-smi power sampled every 1 s; GPU
  board power only, a lower bound on total energy.
- Environment: Python 3.14, torch 2.14.1+cu126 (install from the PyTorch
  cu126 index before `pip install -r requirements.txt`).
- Grid-cost plan: embed once per valid chunking (19). Retrieve top 50 once
  per (chunking, retrieval type); smaller top_k for bm25/dense are prefixes.
  Cross-encoder scores are pointwise, so score each (question, chunk) pair
  once and reuse across top_k.
- Method: score the full grid once into a lookup table (tuning + test nDCG).
  GA (LEAP) and random search are then lookups at equal budgets
  (5%, 10%, 20% of unique configs).
- GA settings are frozen before looking at any comparison results.

## Conventions
- Python in `.venv` (run as `.venv/Scripts/python`); scripts in `scripts/`
  (run in numbered order, from the repo root), logic in `src/ragtune/`.
- Never commit raw data. Parameters live in `config/`, not hard-coded.
- Every run is deterministic given (method, budget, seed). Pass a
  `numpy.random.Generator` explicitly; no global RNG.
- Tests: `.venv/Scripts/python -m pytest` (pyproject sets `pythonpath = src`).
- Search space (`src/ragtune/space.py`): a genome is a tuple of value indices,
  one per gene, in YAML gene order. `Config.key` is the lookup-table key.
  Invalid configs go through `SearchSpace.repair` in BOTH methods: genes in
  violated rules are resampled jointly until valid, so random search
  (uniform genome + repair) is uniform over the 570 valid configs.

## Open decisions
- Seed count (set after timing a run of each method)
- GA parameters (`config/ga_params.yaml`, frozen before any comparison)
