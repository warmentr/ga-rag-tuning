import pytest

from ragtune.pipeline.chunking import (
    chunk_corpus,
    chunk_text,
    content_budget,
    load_tokenizer,
    token_windows,
)

MODEL = "BAAI/bge-small-en-v1.5"


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(MODEL)


# ---- token_windows: pure logic -------------------------------------------

def test_windows_all_word_starts_no_overlap():
    assert token_windows([True] * 10, 4, 0) == [(0, 4), (4, 8), (8, 10)]


def test_windows_with_overlap():
    assert token_windows([True] * 10, 4, 2) == [(0, 4), (2, 6), (4, 8), (6, 10)]


def test_windows_short_input_single_chunk():
    assert token_windows([True] * 3, 4, 2) == [(0, 3)]
    assert token_windows([], 4, 0) == []


def test_windows_end_snaps_back_to_word_start():
    # words: [0,1,2] [3,4] [5] ... token 4 is a continuation
    ws = [True, False, False, True, False, True, True, True]
    # budget 5 would end at 5 (word start) -> keep; budget 4 would end at 4 -> snap to 3
    assert token_windows(ws, 4, 0)[0] == (0, 3)
    assert token_windows(ws, 5, 0)[0] == (0, 5)


def test_windows_next_start_snaps_forward():
    ws = [True, True, True, True, False, True, True, True, True]
    # end would be 4, a continuation, so it snaps back to 3
    w = token_windows(ws, 4, 2)
    assert w[0] == (0, 3)
    # next start = 3 - 2 = 1, already a word start
    assert w[1][0] == 1
    # every window starts on a word start (token 4 is never a start)
    assert all(ws[s] for s, _ in w)
    assert 4 not in [s for s, _ in w]


def test_windows_hard_cut_long_word():
    ws = [True] + [False] * 9
    assert token_windows(ws, 4, 0) == [(0, 4), (4, 8), (8, 10)]


def test_windows_invariants_random():
    import random
    rng = random.Random(0)
    for _ in range(500):
        n = rng.randint(1, 300)
        ws = [True] + [rng.random() < 0.7 for _ in range(n - 1)]
        budget = rng.randint(2, 60)
        overlap = rng.randint(0, budget - 1)
        w = token_windows(ws, budget, overlap)
        assert w[0][0] == 0 and w[-1][1] == n                       # full coverage, ends
        assert all(e - s <= budget and e > s for s, e in w)         # within budget
        assert all(b[0] > a[0] for a, b in zip(w, w[1:]))           # progress
        assert all(b[0] <= a[1] for a, b in zip(w, w[1:]))          # no gaps
        assert all(a[1] - b[0] <= overlap for a, b in zip(w, w[1:]))  # overlap cap


@pytest.mark.parametrize("args", [(0, 0), (4, 4), (4, -1)])
def test_windows_bad_params(args):
    with pytest.raises(ValueError):
        token_windows([True] * 5, *args)


# ---- with the real tokenizer ---------------------------------------------

def test_budget_excludes_special_tokens(tok):
    assert content_budget(512, tok) == 510
    assert content_budget(128, tok) == 126


TEXT = (
    "Wix Payments: Getting Started\n"
    "You can start accepting payments while your Wix Payments account is under review. "
    "Embedding long words like internationalization should never be split mid-word. "
) * 20


@pytest.mark.parametrize("size,overlap", [(128, 0), (128, 64), (256, 32), (512, 128)])
def test_chunk_text_fits_model_and_preserves_text(tok, size, overlap):
    chunks = chunk_text("a1", TEXT, tok, size, overlap)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text == TEXT[c.start_char:c.end_char]
        # re-encoding with special tokens must fit within chunk_size
        assert len(tok.encode(c.text).ids) <= size
        assert c.n_tokens == len(tok.encode(c.text, add_special_tokens=False).ids)
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(TEXT.rstrip())
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_no_chunk_starts_mid_word(tok):
    for c in chunk_text("a1", TEXT, tok, 128, 32):
        if c.start_char > 0:
            prev, cur = TEXT[c.start_char - 1], TEXT[c.start_char]
            assert not (prev.isalnum() and cur.isalnum())


def test_chunk_corpus_matches_chunk_text(tok):
    ids, texts = ["a", "b"], [TEXT, "Short article."]
    batch = chunk_corpus(ids, texts, tok, 256, 64)
    single = chunk_text("a", TEXT, tok, 256, 64) + chunk_text("b", "Short article.", tok, 256, 64)
    assert batch == single
    assert batch[-1].article_id == "b" and batch[-1].text == "Short article."
