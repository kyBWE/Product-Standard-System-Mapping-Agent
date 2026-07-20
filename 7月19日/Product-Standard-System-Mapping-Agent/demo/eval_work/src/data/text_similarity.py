"""中文友好的文本相似度（替代 pg_trgm：其对纯中文 show_trgm 常为空、similarity 恒为 0）。"""
from __future__ import annotations

from difflib import SequenceMatcher


def _char_bigram_jaccard(a: str, b: str) -> float:
    def grams(s: str) -> set[str]:
        s = (s or "").strip()
        if not s:
            return set()
        if len(s) < 2:
            return {s}
        return {s[i : i + 2] for i in range(len(s) - 1)}

    A, B = grams(a), grams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def chinese_text_similarity(a: str, b: str) -> float:
    """返回 [0,1]。越大表示字面越像（近重复），越小表示写法差异越大。"""
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    sm = SequenceMatcher(None, a, b).ratio()
    jac = _char_bigram_jaccard(a, b)
    cont = 0.0
    if a in b or b in a:
        short, long_ = (a, b) if len(a) <= len(b) else (b, a)
        cont = len(short) / max(len(long_), 1)
    return max(sm, jac, cont)


def cosine_similarity(v1: list[float] | None, v2: list[float] | None) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = 0.0
    n1 = 0.0
    n2 = 0.0
    for x, y in zip(v1, v2):
        fx, fy = float(x), float(y)
        dot += fx * fy
        n1 += fx * fx
        n2 += fy * fy
    if n1 <= 0 or n2 <= 0:
        return 0.0
    return dot / ((n1 ** 0.5) * (n2 ** 0.5))
