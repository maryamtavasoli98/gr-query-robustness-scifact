"""Seeded, intent-preserving query perturbations (typos, reordering, truncation, word dropout)."""

from __future__ import annotations

import random

KEYBOARD = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok", "o": "ipl",
    "p": "ol", "a": "qsz", "s": "awdx", "d": "serfc", "f": "drtgv", "g": "ftyhb", "h": "gyujn", "j": "huikm",
    "k": "jiol", "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}  # fmt: skip
MIN_LEN = 4  # only words with at least 4 letters are typo candidates (short words stay readable)


def typo_word(word: str, rng: random.Random) -> str:
    """Apply exactly one character edit: adjacent swap, deletion, insertion or keyboard substitution."""
    if len(word) < MIN_LEN:
        return word
    i = rng.randrange(1, len(word) - 1)  # keep the first letter, as human typos mostly do
    op = rng.choice(["swap", "delete", "insert", "sub"])
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    if op == "delete":
        return word[:i] + word[i + 1 :]
    c = word[i].lower()
    nb = KEYBOARD.get(c, "e")
    if op == "insert":
        return word[:i] + rng.choice(nb) + word[i:]
    return word[:i] + rng.choice(nb) + word[i + 1 :]


def typos(query: str, rate: float, seed: int) -> str:
    """Each eligible word gets one typo with probability `rate` (at least one typo if any word is eligible)."""
    rng = random.Random(seed)
    words = query.split()
    elig = [i for i, w in enumerate(words) if len(w) >= MIN_LEN and w.isalpha()]
    if not elig:
        return query
    hit = {i for i in elig if rng.random() < rate} or {rng.choice(elig)}
    return " ".join(typo_word(w, rng) if i in hit else w for i, w in enumerate(words))


def reorder(query: str, seed: int) -> str:
    rng = random.Random(seed)
    words = query.split()
    if len(words) < 2:
        return query
    for _ in range(10):
        shuffled = words[:]
        rng.shuffle(shuffled)
        if shuffled != words:
            return " ".join(shuffled)
    return query


def truncate(query: str, keep: float = 0.5, seed: int = 0) -> str:
    """Keep the first ceil(keep * n) words (deterministic; `seed` is accepted for a uniform interface)."""
    words = query.split()
    k = max(1, -(-int(len(words) * keep * 1000) // 1000))
    return " ".join(words[:k])


def word_dropout(query: str, rate: float, seed: int) -> str:
    rng = random.Random(seed)
    words = query.split()
    if len(words) < 2:
        return query
    kept = [w for w in words if rng.random() >= rate]
    return " ".join(kept or [rng.choice(words)])


CONDITIONS = {
    "typo10": lambda q, s: typos(q, 0.10, s),
    "typo25": lambda q, s: typos(q, 0.25, s),
    "typo50": lambda q, s: typos(q, 0.50, s),
    "reorder": reorder,
    "truncate50": lambda q, s: truncate(q, 0.5, s),
    "dropout30": lambda q, s: word_dropout(q, 0.30, s),
}
DETERMINISTIC = {"truncate50"}


def perturb(query: str, condition: str, seed: int) -> str:
    return CONDITIONS[condition](query, seed)
