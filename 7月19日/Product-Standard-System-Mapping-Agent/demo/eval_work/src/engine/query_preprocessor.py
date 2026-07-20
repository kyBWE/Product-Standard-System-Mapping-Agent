from __future__ import annotations
import re
import unicodedata


_PAREN_PATTERN = re.compile(r"[（(][^）)]*[）)]")
_DASH_VARIANTS = re.compile(r"[－—–﹣−]")
_MULTI_SEP_PATTERN = re.compile(r"[、，，|]+")
_CN_PUNCT_PATTERN = re.compile(r"[。；：？！\u201c\u201d\u2018\u2019（）【】《》]")
_MULTI_SPACE = re.compile(r"\s+")

_NOISE_SUFFIXES = (
    "类产品", "及其产品", "及其", "其他产品",
)


def preprocess_query(raw: str) -> str:
    if not raw:
        return ""
    text = raw.strip()
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = _PAREN_PATTERN.sub("", text)

    text = _DASH_VARIANTS.sub("-", text)

    text = _CN_PUNCT_PATTERN.sub("", text)

    text = _MULTI_SEP_PATTERN.sub(",", text)

    for suffix in _NOISE_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]

    text = _MULTI_SPACE.sub(" ", text).strip()

    parts = [p.strip() for p in text.split(",") if p.strip()]
    text = parts[0] if parts else ""

    text = text.strip("- \t\n")

    return text