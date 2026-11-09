# Data License and Attribution

## Source
- **Dataset:** WixQA (Wix.com AI Research)
- **URL:** https://huggingface.co/datasets/Wix/WixQA
- **Configs used:** `wix_kb_corpus`, `wixqa_expertwritten`, and `wixqa_simulated`
- **Corpus snapshot:** Wix Help Center, English, dated 2024-12-02
- **Accessed:** November 9th, 2026

## License
Released under the MIT License. Cite “Wix.com AI Research” when using the data.

## Citation
D. Cohen et al., "WixQA: A multi-dataset benchmark for enterprise
retrieval-augmented generation," arXiv:2505.08643, 2025.

## Permitted use
The dataset card lists its intended uses as benchmarking RAG pipelines, training
retrieval models, and studying hallucination versus missing-context errors. This
project uses it to benchmark retrieval configurations, which fits that scope.

## Out-of-scope uses (not done in this project)
- Generating personal, legal, or medical advice
- Deanonymizing user tickets

## Redistribution
Raw data is not committed to this repository (see `.gitignore`). To reproduce
the results, download the configs listed above directly from Hugging Face.
Derived artifacts (the lookup table, run logs, figures) contain configuration
IDs and scores only, not dataset text.

## Modifications
TODO:
[Note any processing: subsampling, filtering by `article_type`, tuning/test
split with seed X. Derived files live in `data/processed/`.]