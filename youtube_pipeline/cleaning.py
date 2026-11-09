from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

# Broad emoji ranges (covers common unicode emoji blocks).
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

URL_RE = re.compile(r"(http[s]?://\S+|www\.\S+)", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_TEXT_RE = re.compile(r"[^a-z0-9_áéíóúñü\s!?]+", flags=re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"([a-záéíóúñü])\1{2,}", flags=re.IGNORECASE)
REPEATED_TOKEN_RE = re.compile(r"\b(\w+)(\s+\1){2,}\b", flags=re.IGNORECASE)

# Minimal slang map; extend in notebooks by passing custom_slang_map.
DEFAULT_SLANG_MAP = {
    "q": "que",
    "xq": "porque",
    "pq": "porque",
    "k": "que",
    "bn": "bien",
    "tmb": "tambien",
    "xd": "emoji_laugh",
    "jajaja": "risa",
}


@dataclass(frozen=True)
class CleaningConfig:
    """Input, output, and existing cleaning parameters for one preparation stage."""

    input_path: str | Path
    output_path: str | Path
    raw_text_col: str = "text"
    timestamp_col: str = "published_at"
    keep_spam: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "CleaningConfig":
        config_payload = payload.get("cleaning", payload)
        if not isinstance(config_payload, dict):
            raise ValueError("Cleaning config must be an object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(config_payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown cleaning config fields: {unknown}")
        return cls(**config_payload)


def _emoji_tokenize(text: str) -> tuple[str, int]:
    count = 0

    def replacer(match: re.Match[str]) -> str:
        nonlocal count
        token = match.group(0)
        count += len(token)
        encoded = "_".join(f"{ord(ch):x}" for ch in token)
        return f" emoji_{encoded} "

    return EMOJI_RE.sub(replacer, text), count


def _replace_repeated_punctuation(text: str) -> tuple[str, int, int]:
    exclamation_count = text.count("!")
    question_count = text.count("?")

    def _tokenize_punct(symbol: str, token: str, src: str) -> str:
        return re.sub(
            rf"\{symbol}+",
            lambda m: " " + " ".join([token] * min(3, len(m.group(0)))) + " ",
            src,
        )

    out = _tokenize_punct("!", "exclaim_token", text)
    out = _tokenize_punct("?", "question_token", out)
    return out, exclamation_count, question_count


def normalize_comment(
    text: str,
    custom_slang_map: dict[str, str] | None = None,
) -> dict[str, object]:
    """Noise reduction while preserving emotional cues for polarization."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        text = ""

    raw = str(text)
    raw = html.unescape(raw)
    raw = unicodedata.normalize("NFKC", raw)

    uppercase_chars = sum(ch.isupper() for ch in raw)
    alpha_chars = sum(ch.isalpha() for ch in raw)
    caps_ratio = (uppercase_chars / alpha_chars) if alpha_chars else 0.0

    link_count = len(URL_RE.findall(raw))
    normalized = URL_RE.sub(" url_token ", raw)
    normalized = MENTION_RE.sub(" user_token ", normalized)
    normalized = HASHTAG_RE.sub(r" \1 ", normalized)

    normalized, emoji_count = _emoji_tokenize(normalized)
    normalized, exclamation_count, question_count = _replace_repeated_punctuation(
        normalized
    )

    normalized = normalized.lower()
    normalized = REPEATED_CHAR_RE.sub(r"\1\1", normalized)
    normalized = NON_TEXT_RE.sub(" ", normalized)
    normalized = REPEATED_TOKEN_RE.sub(r"\1", normalized)

    slang_map = dict(DEFAULT_SLANG_MAP)
    if custom_slang_map:
        slang_map.update(custom_slang_map)
    for slang, replacement in slang_map.items():
        normalized = re.sub(
            rf"\b{re.escape(slang.lower())}\b", replacement.lower(), normalized
        )

    normalized = SPACE_RE.sub(" ", normalized).strip()
    token_count = len(normalized.split()) if normalized else 0

    return {
        "text_clean": normalized,
        "emoji_count": emoji_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "caps_ratio": round(caps_ratio, 4),
        "link_count": link_count,
        "token_count": token_count,
    }


def flag_probable_spam(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    author_col: str = "author_id",
    link_col: str = "link_count",
    min_tokens: int = 2,
    duplicate_limit: int = 4,
) -> pd.Series:
    """Lightweight heuristic to reduce high-noise comments."""
    base = pd.Series(False, index=df.index)

    too_short = df[text_col].fillna("").str.split().map(len) < min_tokens
    too_many_links = df[link_col].fillna(0) >= 2
    repeated_text = df[text_col].fillna("").str.len() > 0

    if author_col in df.columns:
        group_cols = [author_col, text_col]
        repeated_count = df.loc[repeated_text].groupby(group_cols, dropna=False).size()
        repeated_pairs = repeated_count[repeated_count >= duplicate_limit].index
        duplicate_by_author = df.set_index(group_cols).index.isin(repeated_pairs)
    else:
        duplicate_by_author = pd.Series(False, index=df.index)

    return base | too_short | too_many_links | pd.Series(duplicate_by_author, index=df.index)


def _remove_orphan_replies(
    df: pd.DataFrame,
    is_reply_col: str = "is_reply",
    parent_col: str = "reply_to_comment_id",
    comment_id_col: str = "comment_id",
) -> pd.DataFrame:
    if (
        is_reply_col not in df.columns
        or parent_col not in df.columns
        or comment_id_col not in df.columns
    ):
        return df
    valid_comment_ids = set(df[comment_id_col].dropna().astype(str).tolist())
    orphan_mask = df[is_reply_col].fillna(False) & ~df[parent_col].astype(str).isin(
        valid_comment_ids
    )
    return df.loc[~orphan_mask].copy()


def _drop_temporal_text_duplicates(
    df: pd.DataFrame,
    *,
    time_col: str = "event_time_utc",
    author_col: str = "author_id",
    text_col: str = "text_clean",
    freq: str = "2min",
) -> pd.DataFrame:
    if time_col not in df.columns or author_col not in df.columns or text_col not in df.columns:
        return df

    out = df.copy()
    out["_text_norm"] = out[text_col].fillna("").astype(str).str.lower().str.strip()

    candidate_mask = (
        out[time_col].notna()
        & out[author_col].notna()
        & out["_text_norm"].str.len().gt(0)
    )
    if not candidate_mask.any():
        return out.drop(columns=["_text_norm"], errors="ignore")

    candidates = out.loc[candidate_mask].sort_values(time_col, kind="stable")
    duplicate_rank = candidates.groupby(
        [pd.Grouper(key=time_col, freq=freq), author_col, "_text_norm"],
        dropna=False,
        sort=False,
    ).cumcount()

    keep_mask = pd.Series(True, index=out.index)
    keep_mask.loc[candidates.index] = duplicate_rank.eq(0).to_numpy()

    out = out.loc[keep_mask].copy()
    return out.drop(columns=["_text_norm"], errors="ignore")


def _unix_seconds_or_na(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    return int(pd.Timestamp(value).timestamp())


def _ensure_event_time_unix_fields(df: pd.DataFrame) -> pd.DataFrame:
    if "event_time_utc" not in df.columns:
        return df

    out = df.copy()
    event_ts = pd.to_datetime(out["event_time_utc"], utc=True, errors="coerce")
    if "event_time_unix_s" not in out.columns:
        values = event_ts.map(_unix_seconds_or_na)
        dtype = "Int64" if values.isna().any() else "int64"
        out["event_time_unix_s"] = values.astype(dtype)
    if "event_time_unix_ms" not in out.columns and "event_time_unix_s" in out.columns:
        # Legacy alias retained for compatibility; values are Unix seconds.
        out["event_time_unix_ms"] = out["event_time_unix_s"]
    return out


def clean_comments_dataframe(
    df: pd.DataFrame,
    raw_text_col: str = "text",
    timestamp_col: str = "published_at",
    keep_spam: bool = False,
    custom_slang_map: dict[str, str] | None = None,
    required_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Clean comments for TF-IDF/polarization without erasing emotional signal.

    Output columns added:
      - text_clean
      - emoji_count, exclamation_count, question_count, caps_ratio, link_count
      - is_probable_spam
    """
    if raw_text_col not in df.columns:
        raise KeyError(f"Column '{raw_text_col}' not found in dataframe.")

    out = df.copy()
    norm_records = out[raw_text_col].map(
        lambda text: normalize_comment(text, custom_slang_map=custom_slang_map)
    )
    norm_df = pd.DataFrame(list(norm_records), index=out.index)
    out = pd.concat([out, norm_df], axis=1)

    if timestamp_col in out.columns:
        out["event_time_utc"] = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")

    out["is_probable_spam"] = flag_probable_spam(out)
    if not keep_spam:
        out = out.loc[~out["is_probable_spam"]].copy()

    out = out.loc[out["text_clean"].fillna("").str.len() > 0].copy()
    out = _remove_orphan_replies(out)
    out = _drop_temporal_text_duplicates(out)
    out = _ensure_event_time_unix_fields(out)

    if required_cols:
        missing = [col for col in required_cols if col not in out.columns]
        if missing:
            raise KeyError(f"Missing required columns after cleaning: {missing}")

    return out.reset_index(drop=True)


def _read_cleaning_input(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.is_dir() or source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported file format: {source}")


def _write_cleaning_output(df: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        df.to_csv(output, index=False)
    else:
        df.to_parquet(output, index=False)
    return output


def clean_comments_from_config(config: CleaningConfig) -> Path:
    """Execute the existing cleaning behavior from its component configuration."""

    if not isinstance(config, CleaningConfig):
        raise TypeError("config must be CleaningConfig.")
    comments_df = _read_cleaning_input(config.input_path)
    timestamp_col = config.timestamp_col
    if (
        timestamp_col not in comments_df.columns
        and "event_time_utc" in comments_df.columns
    ):
        timestamp_col = "event_time_utc"
    clean_df = clean_comments_dataframe(
        comments_df,
        raw_text_col=config.raw_text_col,
        timestamp_col=timestamp_col,
        keep_spam=config.keep_spam,
    )
    return _write_cleaning_output(clean_df, config.output_path)


def top_emoji_tokens(df: pd.DataFrame, text_col: str = "text_clean", n: int = 20) -> pd.Series:
    tokens = []
    for text in df[text_col].fillna(""):
        tokens.extend([token for token in text.split() if token.startswith("emoji_")])
    return pd.Series(Counter(tokens)).sort_values(ascending=False).head(n)
