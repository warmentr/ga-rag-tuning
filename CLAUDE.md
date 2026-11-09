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
- BM25: `bm25s`, lowercase + English stopwords.
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
