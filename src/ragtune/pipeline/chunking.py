"""Token-based chunking with the embedding model's tokenizer.

Conventions (fixed across all configs):
- `chunk_size` is the total sequence length the embedding model sees,
  INCLUDING its special tokens ([CLS], [SEP] for bge). Content tokens per
  chunk = chunk_size - n_special, so a 512 chunk fits bge's 512 limit with
  no truncation.
- `chunk_overlap` is in content tokens.
- Boundaries snap to word starts (per the tokenizer's pre-tokenizer: words
  and punctuation marks), so no chunk starts or ends inside a word. The end
  snaps back and the next start snaps forward, so a chunk is never longer
  than the content budget and the actual overlap is never more than
  `chunk_overlap`. A single word longer than the budget is hard-cut.
- Chunk text is sliced from the original article text by character offsets,
  so casing and whitespace are preserved.
- The `contents` field is chunked as-is (it begins with the article title);
  the title is not prepended to later chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from tokenizers import Encoding, Tokenizer


@dataclass(frozen=True)
class Chunk:
    article_id: str
    index: int          # position of the chunk within its article
    text: str
    start_char: int
    end_char: int
    n_tokens: int       # content tokens (excluding special tokens)


@lru_cache(maxsize=None)
def load_tokenizer(model_name: str) -> Tokenizer:
    tok = Tokenizer.from_pretrained(model_name)
    tok.no_truncation()
    tok.no_padding()
    return tok


def n_special_tokens(tokenizer: Tokenizer) -> int:
    return tokenizer.post_processor.num_special_tokens_to_add(False)


def content_budget(chunk_size: int, tokenizer: Tokenizer) -> int:
    return chunk_size - n_special_tokens(tokenizer)


def _word_starts(enc: Encoding) -> list[bool]:
    wids = enc.word_ids
    return [i == 0 or wids[i] != wids[i - 1] for i in range(len(wids))]


def token_windows(is_word_start: list[bool], budget: int, overlap: int) -> list[tuple[int, int]]:
    """Token-index windows [s, e) over a sequence of length len(is_word_start)."""
    if budget < 1:
        raise ValueError(f"content budget must be >= 1, got {budget}")
    if not 0 <= overlap < budget:
        raise ValueError(f"need 0 <= overlap < budget, got overlap={overlap}, budget={budget}")
    n = len(is_word_start)
    windows = []
    s = 0
    while s < n:
        e = min(s + budget, n)
        if e < n:
            # snap end back to a word start (> s); hard cut if the word is too long
            snapped = e
            while snapped > s and not is_word_start[snapped]:
                snapped -= 1
            if snapped > s:
                e = snapped
        windows.append((s, e))
        if e >= n:
            break
        # next start: e - overlap, snapped forward to a word start, strictly past s
        nxt = max(e - overlap, s + 1)
        while nxt < e and not is_word_start[nxt]:
            nxt += 1
        s = nxt
    return windows


def chunk_text(
    article_id: str, text: str, tokenizer: Tokenizer, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    enc = tokenizer.encode(text, add_special_tokens=False)
    return _chunks_from_encoding(article_id, text, enc, content_budget(chunk_size, tokenizer), chunk_overlap)


def _chunks_from_encoding(article_id, text, enc, budget, overlap) -> list[Chunk]:
    chunks = []
    offsets = enc.offsets
    for i, (s, e) in enumerate(token_windows(_word_starts(enc), budget, overlap)):
        a, b = offsets[s][0], offsets[e - 1][1]
        chunks.append(Chunk(article_id, i, text[a:b], a, b, e - s))
    return chunks


def chunk_corpus(
    article_ids: list[str],
    texts: list[str],
    tokenizer: Tokenizer,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Chunk every article; output order is article order, then chunk order."""
    if len(article_ids) != len(texts):
        raise ValueError("article_ids and texts must align")
    budget = content_budget(chunk_size, tokenizer)
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    chunks = []
    for aid, text, enc in zip(article_ids, texts, encodings):
        chunks.extend(_chunks_from_encoding(aid, text, enc, budget, chunk_overlap))
    return chunks
