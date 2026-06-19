from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from tqdm.auto import tqdm

try:
    from News.data_for_model.build_ngram_terms import (
        normalize_sentence_token_lists,
    )
except ImportError:
    from build_ngram_terms import normalize_sentence_token_lists


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DICTIONARY_PATH = PROJECT_ROOT / "Dataset" / "candidate_ngram_terms_dictionary.parquet"
INPUT_PATH = PROJECT_ROOT / "Dataset" / "equity_news_tokenized_vncorenlp.parquet"
OUTPUT_PARQUET_PATH = PROJECT_ROOT / "Dataset" / "equity_news_content_sentiment_ratios.parquet"
OUTPUT_SAMPLE_CSV_PATH = (
    PROJECT_ROOT / "Dataset" / "equity_news_content_sentiment_ratios_sample.csv"
)

TOKENIZED_SENTENCES_COLUMN = "Tokenize_content_sentences"
TOKENIZED_FALLBACK_COLUMN = "Tokenize_content"
TOTAL_TOKEN_COLUMN = "total_tokenizer"
SAMPLE_CSV_ROWS = 1000
POSITIVE_THRESHOLD = 0.30
NEGATIVE_THRESHOLD = -0.30
MIN_POLARITY_HITS = 2

LABEL_ALIASES = {
    "pos": "positive",
    "posi": "positive",
    "positive": "positive",
    "neg": "negative",
    "negative": "negative",
    "neu": "neutral",
    "neutral": "neutral",
}
VALID_LABELS = {"positive", "negative", "neutral"}


def normalize_term(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def normalize_label(value: object) -> str:
    label = normalize_term(value)
    label = LABEL_ALIASES.get(label, label)
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid sentiment label: {value!r}")
    return label


def load_sentiment_dictionary(
    path: Path = DICTIONARY_PATH,
) -> dict[int, dict[str, str]]:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path).astype(str)
    else:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    required_columns = {"term", "ngram_n", "sentiment_label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dictionary is missing columns: {sorted(missing_columns)}")

    df = df.loc[:, ["term", "ngram_n", "sentiment_label"]].dropna().copy()
    df["term"] = df["term"].apply(normalize_term)
    df["ngram_n"] = pd.to_numeric(df["ngram_n"], errors="coerce").astype("Int64")
    df["sentiment_label"] = df["sentiment_label"].apply(normalize_label)
    df = df.loc[df["term"].ne("") & df["ngram_n"].isin([1, 2])].copy()

    conflicting_terms = (
        df.groupby("term")["sentiment_label"].nunique().loc[lambda values: values.gt(1)]
    )
    if not conflicting_terms.empty:
        raise ValueError(
            "Dictionary has terms with conflicting sentiment labels: "
            f"{conflicting_terms.index[:10].tolist()}"
        )

    df = df.drop_duplicates(subset=["term"], keep="first")
    return {
        int(ngram_n): dict(zip(group["term"], group["sentiment_label"], strict=False))
        for ngram_n, group in df.groupby("ngram_n")
    }


def count_labeled_content_tokens(
    raw_sentences: object,
    dictionary_by_ngram: dict[int, dict[str, str]],
) -> dict[str, object]:
    token_counts = {"positive": 0, "negative": 0, "neutral": 0}
    term_counts = {"positive": 0, "negative": 0, "neutral": 0}

    unigram_dictionary = dictionary_by_ngram.get(1, {})
    bigram_dictionary = dictionary_by_ngram.get(2, {})

    for sentence_tokens in normalize_sentence_token_lists(raw_sentences):
        tokens = [
            normalized_token
            for token in sentence_tokens
            if (normalized_token := normalize_term(token))
        ]

        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens):
                bigram = f"{tokens[index]} {tokens[index + 1]}"
                bigram_label = bigram_dictionary.get(bigram)
                if bigram_label is not None:
                    token_counts[bigram_label] += 2
                    term_counts[bigram_label] += 1
                    index += 2
                    continue

            unigram_label = unigram_dictionary.get(tokens[index])
            if unigram_label is not None:
                token_counts[unigram_label] += 1
                term_counts[unigram_label] += 1

            index += 1

    return {
        "pos_count": token_counts["positive"],
        "neg_count": token_counts["negative"],
        "neutral_count": token_counts["neutral"],
        "pos_term_count": term_counts["positive"],
        "neg_term_count": term_counts["negative"],
        "neutral_term_count": term_counts["neutral"],
    }


def calculate_sentiment_score(pos_count: int, neg_count: int) -> float:
    polarity_count = pos_count + neg_count
    if polarity_count == 0:
        return 0.0

    return (pos_count - neg_count) / polarity_count


def assign_sentiment_label(
    sentiment_score: float,
    polarity_count: int,
    positive_threshold: float = POSITIVE_THRESHOLD,
    negative_threshold: float = NEGATIVE_THRESHOLD,
    min_polarity_hits: int = MIN_POLARITY_HITS,
) -> str:
    if polarity_count < min_polarity_hits:
        return "neutral"
    if sentiment_score >= positive_threshold:
        return "positive"
    if sentiment_score <= negative_threshold:
        return "negative"
    return "neutral"


def add_sentiment_ratios(
    news_df: pd.DataFrame,
    dictionary_by_ngram: dict[int, dict[str, str]],
) -> pd.DataFrame:
    tokenized_column = (
        TOKENIZED_SENTENCES_COLUMN
        if TOKENIZED_SENTENCES_COLUMN in news_df.columns
        else TOKENIZED_FALLBACK_COLUMN
    )
    if tokenized_column not in news_df.columns:
        raise ValueError(
            "Input data must contain either "
            f"{TOKENIZED_SENTENCES_COLUMN!r} or {TOKENIZED_FALLBACK_COLUMN!r}"
        )
    if TOTAL_TOKEN_COLUMN not in news_df.columns:
        raise ValueError(f"Input data is missing column: {TOTAL_TOKEN_COLUMN!r}")

    count_rows = [
        count_labeled_content_tokens(raw_sentences, dictionary_by_ngram)
        for raw_sentences in tqdm(
            news_df[tokenized_column],
            total=len(news_df),
            desc="Build content sentiment ratios",
        )
    ]
    count_df = pd.DataFrame(count_rows)
    out = pd.concat([news_df.reset_index(drop=True), count_df], axis=1)

    denominator = pd.to_numeric(out[TOTAL_TOKEN_COLUMN], errors="coerce").fillna(0)
    denominator = denominator.where(denominator.gt(0), other=pd.NA)

    out["pos_ratio"] = (out["pos_count"] / denominator).fillna(0.0)
    out["neg_ratio"] = (out["neg_count"] / denominator).fillna(0.0)
    out["neutral_ratio"] = (out["neutral_count"] / denominator).fillna(0.0)
    out["sentiment_coverage_ratio"] = (
        (out["pos_count"] + out["neg_count"] + out["neutral_count"]) / denominator
    ).fillna(0.0)
    out["polarity_count"] = out["pos_count"] + out["neg_count"]
    out["sentiment_score"] = [
        calculate_sentiment_score(pos_count, neg_count)
        for pos_count, neg_count in zip(out["pos_count"], out["neg_count"], strict=False)
    ]
    out["sentiment_label"] = [
        assign_sentiment_label(score, polarity_count)
        for score, polarity_count in zip(
            out["sentiment_score"],
            out["polarity_count"],
            strict=False,
        )
    ]

    return out


def build_content_sentiment_ratios(
    input_path: Path = INPUT_PATH,
    dictionary_path: Path = DICTIONARY_PATH,
) -> pd.DataFrame:
    dictionary_by_ngram = load_sentiment_dictionary(dictionary_path)
    news_df = pd.read_parquet(input_path)
    return add_sentiment_ratios(news_df, dictionary_by_ngram)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out = build_content_sentiment_ratios()

    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    out.head(SAMPLE_CSV_ROWS).to_csv(
        OUTPUT_SAMPLE_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("Input:", INPUT_PATH)
    print("Dictionary:", DICTIONARY_PATH)
    print("Output parquet:", OUTPUT_PARQUET_PATH)
    print("Output sample csv:", OUTPUT_SAMPLE_CSV_PATH)
    print("Rows:", len(out))
    print("Content rows:", len(out))
    print("Positive matched tokens:", int(out["pos_count"].sum()))
    print("Negative matched tokens:", int(out["neg_count"].sum()))
    print("Neutral matched tokens:", int(out["neutral_count"].sum()))
    print("Positive matched terms:", int(out["pos_term_count"].sum()))
    print("Negative matched terms:", int(out["neg_term_count"].sum()))
    print("Neutral matched terms:", int(out["neutral_term_count"].sum()))
    print("Positive content:", int(out["sentiment_label"].eq("positive").sum()))
    print("Negative content:", int(out["sentiment_label"].eq("negative").sum()))
    print("Neutral content:", int(out["sentiment_label"].eq("neutral").sum()))
    print("Sentiment label counts:")
    print(out["sentiment_label"].value_counts(dropna=False).to_string())
    print(
        out[
            [
                "pos_count",
                "neg_count",
                "neutral_count",
                "polarity_count",
                "pos_ratio",
                "neg_ratio",
                "neutral_ratio",
                "sentiment_coverage_ratio",
                "sentiment_score",
            ]
        ]
        .describe()
        .to_string()
    )


if __name__ == "__main__":
    main()
