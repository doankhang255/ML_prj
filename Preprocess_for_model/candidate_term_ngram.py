from __future__ import annotations

from pathlib import Path
import math
import sys

import pandas as pd

try:
    from build_ngram_terms import (
        NGRAM_SEPARATOR,
        OUTPUT_PATH as NGRAM_TERMS_PATH,
        PROJECT_ROOT,
    )
except ImportError:
    from Preprocess_for_model.build_ngram_terms import (
        NGRAM_SEPARATOR,
        OUTPUT_PATH as NGRAM_TERMS_PATH,
        PROJECT_ROOT,
    )


INPUT_PATH = NGRAM_TERMS_PATH
OUTPUT_PATH = PROJECT_ROOT / "Dataset" / "candidate_ngram_terms.parquet"
OUTPUT_CSV_PATH = PROJECT_ROOT / "Dataset" / "candidate_ngram_terms.csv"
STOPWORDS_PATH = PROJECT_ROOT / "Dataset" / "vietnamese-stopwords-dash.txt"
SENTIMENT_WORD_PATH = PROJECT_ROOT / "Dataset" / "sentiment_word.txt"

MAX_DF_RATIO = 0.22
MIN_DF_BY_NGRAM = {1: 5000, 2: 200,}
REQUIRED_COLUMNS = {"term", "ngram_n", "tf", "df"}


def load_stopwords(path: Path = STOPWORDS_PATH) -> set[str]:
    with open(path, encoding="utf-8") as file:
        return {line.strip().casefold() for line in file if line.strip()}


def load_sentiment_tokens(path: Path = SENTIMENT_WORD_PATH) -> set[str]:
    text = path.read_text(encoding="utf-8")
    tokens: set[str] = set()
    for raw_item in text.replace("\n", ",").split(","):
        item = raw_item.strip().casefold()
        if not item:
            continue

        tokens.add(item)
        tokens.add("_".join(item.split()))

    return tokens


def has_stopword_boundary(term: str, stopwords: set[str]) -> bool:
    if not stopwords:
        return False

    tokens = str(term).split(NGRAM_SEPARATOR)
    if not tokens:
        return False

    return tokens[0].casefold() in stopwords or tokens[-1].casefold() in stopwords


def split_by_mask(df: pd.DataFrame, mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    failed_df = df.loc[mask].copy().reset_index(drop=True)
    passed_df = df.loc[~mask].copy().reset_index(drop=True)
    return passed_df, failed_df


def infer_total_documents(ngram_terms_df: pd.DataFrame) -> int:
    if ngram_terms_df.empty:
        return 0
    if "df_ratio" not in ngram_terms_df.columns:
        raise ValueError("total_documents is required when df_ratio column is missing")

    valid_df_ratio = ngram_terms_df["df_ratio"].gt(0)
    if not valid_df_ratio.any():
        return 0

    estimates = (
        ngram_terms_df.loc[valid_df_ratio, "df"]
        / ngram_terms_df.loc[valid_df_ratio, "df_ratio"]
    )
    return int(round(estimates.median()))


def validate_ngram_terms(ngram_terms_df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(ngram_terms_df.columns)
    if missing_columns:
        raise ValueError(f"Input n-gram terms must contain columns: {sorted(missing_columns)}")


def add_candidate_statistics(
    ngram_terms_df: pd.DataFrame,
    total_documents: int,
) -> pd.DataFrame:
    out = ngram_terms_df.copy()
    if "df_ratio" not in out.columns:
        out["df_ratio"] = out["df"] / total_documents if total_documents else 0.0

    out["avg_tf_per_doc"] = out["tf"] / out["df"]
    out["candidate_score"] = out["tf"] * out["df"].apply(
        lambda df: math.log((total_documents + 1) / (df + 1))
    )
    return out


def build_min_df_mask(
    term_stats: pd.DataFrame,
    min_df_by_ngram: dict[int, int],
    sentiment_tokens: set[str] | None = None,
) -> pd.Series:
    sentiment_tokens = sentiment_tokens or set()
    min_df_threshold = term_stats["ngram_n"].map(min_df_by_ngram).fillna(0)
    below_min_df_mask = term_stats["df"].lt(min_df_threshold)
    if not sentiment_tokens:
        return below_min_df_mask

    sentiment_token_mask = build_sentiment_priority_mask(
        term_stats,
        sentiment_tokens=sentiment_tokens,
    )
    return below_min_df_mask & ~sentiment_token_mask


def build_sentiment_min_df_keep_mask(
    term_stats: pd.DataFrame,
    min_df_by_ngram: dict[int, int],
    sentiment_tokens: set[str] | None = None,
) -> pd.Series:
    sentiment_tokens = sentiment_tokens or set()
    if not sentiment_tokens:
        return pd.Series(False, index=term_stats.index)

    min_df_threshold = term_stats["ngram_n"].map(min_df_by_ngram).fillna(0)
    below_min_df_mask = term_stats["df"].lt(min_df_threshold)
    sentiment_token_mask = build_sentiment_priority_mask(
        term_stats,
        sentiment_tokens=sentiment_tokens,
    )
    return below_min_df_mask & sentiment_token_mask


def find_sentiment_tokens(term: str, sentiment_tokens: set[str]) -> set[str]:
    term_text = str(term).casefold()
    return {token for token in sentiment_tokens if token in term_text}


def contains_sentiment_token(term: str, sentiment_tokens: set[str]) -> bool:
    return bool(find_sentiment_tokens(term, sentiment_tokens))


def build_sentiment_priority_mask(
    term_stats: pd.DataFrame,
    sentiment_tokens: set[str] | None = None,
    preferred_ngram_n: int = 2,
    fallback_ngram_n: int = 1,
) -> pd.Series:
    sentiment_tokens = sentiment_tokens or set()
    if not sentiment_tokens:
        return pd.Series(False, index=term_stats.index)

    matched_tokens = term_stats["term"].apply(
        lambda term: find_sentiment_tokens(term, sentiment_tokens)
    )
    preferred_tokens: set[str] = set()
    for tokens in matched_tokens.loc[term_stats["ngram_n"].eq(preferred_ngram_n)]:
        preferred_tokens.update(tokens)

    def is_priority_match(index: int) -> bool:
        tokens = matched_tokens.loc[index]
        if not tokens:
            return False

        ngram_n = term_stats.at[index, "ngram_n"]
        if ngram_n == preferred_ngram_n:
            return True
        if ngram_n == fallback_ngram_n:
            return bool(tokens.difference(preferred_tokens))

        return True

    return pd.Series(
        (is_priority_match(index) for index in term_stats.index),
        index=term_stats.index,
    )


def build_shadowed_sentiment_ngram1_mask(
    term_stats: pd.DataFrame,
    sentiment_tokens: set[str] | None = None,
    preferred_ngram_n: int = 2,
    fallback_ngram_n: int = 1,
) -> pd.Series:
    sentiment_tokens = sentiment_tokens or set()
    if not sentiment_tokens:
        return pd.Series(False, index=term_stats.index)

    matched_tokens = term_stats["term"].apply(
        lambda term: find_sentiment_tokens(term, sentiment_tokens)
    )
    preferred_tokens: set[str] = set()
    for tokens in matched_tokens.loc[term_stats["ngram_n"].eq(preferred_ngram_n)]:
        preferred_tokens.update(tokens)

    if not preferred_tokens:
        return pd.Series(False, index=term_stats.index)

    return (
        term_stats["ngram_n"].eq(fallback_ngram_n)
        & matched_tokens.apply(lambda tokens: bool(tokens.intersection(preferred_tokens)))
    )


def choose_candidate_ngram_terms(
    ngram_terms_df: pd.DataFrame,
    total_documents: int | None = None,
    min_df_by_ngram: dict[int, int] | None = None,
    sentiment_tokens: set[str] | None = None,
    max_df_ratio: float = MAX_DF_RATIO,
    remove_stopwords: bool = True,
    stopwords: set[str] | None = None,
    return_groups: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    validate_ngram_terms(ngram_terms_df)
    total_documents = (
        infer_total_documents(ngram_terms_df)
        if total_documents is None
        else total_documents
    )
    term_stats = add_candidate_statistics(
        ngram_terms_df=ngram_terms_df,
        total_documents=total_documents,
    )
    term_stats["rejection_reason"] = None

    if remove_stopwords:
        stopwords = stopwords or set()
        stopword_boundary_mask = term_stats["term"].apply(
            lambda term: has_stopword_boundary(term, stopwords)
        )
    else:
        stopword_boundary_mask = pd.Series(False, index=term_stats.index)

    non_stopword_df, stopword_boundary_df = split_by_mask(
        term_stats,
        stopword_boundary_mask,
    )
    stopword_boundary_df["rejection_reason"] = "stopword_boundary"

    min_df_by_ngram = min_df_by_ngram or MIN_DF_BY_NGRAM
    below_min_df_mask = build_min_df_mask(
        non_stopword_df,
        min_df_by_ngram=min_df_by_ngram,
        sentiment_tokens=sentiment_tokens,
    )
    sentiment_min_df_keep_mask = build_sentiment_min_df_keep_mask(
        non_stopword_df,
        min_df_by_ngram=min_df_by_ngram,
        sentiment_tokens=sentiment_tokens,
    )
    min_df_pass_df, below_min_df_df = split_by_mask(
        non_stopword_df,
        below_min_df_mask,
    )
    below_min_df_df["rejection_reason"] = "below_min_df_by_ngram"
    sentiment_min_df_keep_df = non_stopword_df.loc[
        sentiment_min_df_keep_mask
    ].copy().reset_index(drop=True)

    above_max_df_ratio_mask = min_df_pass_df["df_ratio"].gt(max_df_ratio)
    candidate_terms_df, above_max_df_ratio_df = split_by_mask(
        min_df_pass_df,
        above_max_df_ratio_mask,
    )
    above_max_df_ratio_df["rejection_reason"] = "above_max_df_ratio"

    sentiment_shadowed_ngram1_mask = build_shadowed_sentiment_ngram1_mask(
        candidate_terms_df,
        sentiment_tokens=sentiment_tokens,
    )
    candidate_terms_df, sentiment_shadowed_ngram1_df = split_by_mask(
        candidate_terms_df,
        sentiment_shadowed_ngram1_mask,
    )
    sentiment_shadowed_ngram1_df["rejection_reason"] = (
        "sentiment_shadowed_by_ngram_2"
    )

    candidate_terms_df = candidate_terms_df.drop(columns=["rejection_reason"])
    candidate_terms_df = candidate_terms_df.sort_values(
        by=["candidate_score", "tf", "df", "ngram_n", "term"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )
    candidate_terms_df = candidate_terms_df.reset_index(drop=True)

    if not return_groups:
        return candidate_terms_df

    return {
        "all_ngram_terms_df": term_stats.reset_index(drop=True),
        "stopword_boundary_df": stopword_boundary_df,
        "sentiment_min_df_keep_df": sentiment_min_df_keep_df,
        "sentiment_shadowed_ngram1_df": sentiment_shadowed_ngram1_df,
        "below_min_df_df": below_min_df_df,
        "above_max_df_ratio_df": above_max_df_ratio_df,
        "candidate_terms_df": candidate_terms_df,
    }


def build_candidate_ngram_terms(
    path: Path = INPUT_PATH,
    min_df_by_ngram: dict[int, int] | None = None,
    max_df_ratio: float = MAX_DF_RATIO,
    remove_stopwords: bool = True,
    stopwords_path: Path = STOPWORDS_PATH,
    sentiment_word_path: Path = SENTIMENT_WORD_PATH,
    total_documents: int | None = None,
) -> dict[str, pd.DataFrame]:
    ngram_terms_df = pd.read_parquet(path)
    stopwords = load_stopwords(stopwords_path) if remove_stopwords else set()
    sentiment_tokens = load_sentiment_tokens(sentiment_word_path)
    return choose_candidate_ngram_terms(
        ngram_terms_df=ngram_terms_df,
        total_documents=total_documents,
        min_df_by_ngram=min_df_by_ngram,
        sentiment_tokens=sentiment_tokens,
        max_df_ratio=max_df_ratio,
        remove_stopwords=remove_stopwords,
        stopwords=stopwords,
        return_groups=True,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    result = build_candidate_ngram_terms()
    candidate_terms_df = result["candidate_terms_df"]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidate_terms_df.to_parquet(OUTPUT_PATH, index=False)
    candidate_terms_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Input n-gram terms:", INPUT_PATH)
    print("Stopwords path:", STOPWORDS_PATH)
    print("Sentiment word path:", SENTIMENT_WORD_PATH)
    print("Minimum df by n-gram:", MIN_DF_BY_NGRAM)
    print("Maximum df_ratio:", MAX_DF_RATIO)
    print("Output parquet:", OUTPUT_PATH)
    print("Output csv:", OUTPUT_CSV_PATH)
    print(
        "N-grams removed by stopword boundary:",
        len(result["stopword_boundary_df"]),
    )
    print(
        "N-grams kept below min_df because they contain sentiment tokens:",
        len(result["sentiment_min_df_keep_df"]),
    )
    print(
        "Sentiment unigram terms skipped because n-gram 2 matched first:",
        len(result["sentiment_shadowed_ngram1_df"]),
    )
    print(
        "N-grams removed by min_df by n-gram:",
        len(result["below_min_df_df"]),
    )
    print(
        "N-grams removed by df_ratio > max_df_ratio:",
        len(result["above_max_df_ratio_df"]),
    )
    print("Candidate n-gram terms:", len(candidate_terms_df))
    print(candidate_terms_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
