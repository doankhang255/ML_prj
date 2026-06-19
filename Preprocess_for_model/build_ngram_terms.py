from __future__ import annotations

from collections import Counter
from pathlib import Path
import ast
import sys
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "Dataset" / "equity_news_tokenized_vncorenlp.parquet"
OUTPUT_PATH = PROJECT_ROOT / "Dataset" / "ngram_terms.parquet"
OUTPUT_CSV_PATH = PROJECT_ROOT / "Dataset" / "ngram_terms.csv"

TOKENIZED_COLUMN = "Tokenize_content_sentences"
FALLBACK_TOKENIZED_COLUMN = "Tokenize_content"
NGRAM_SEPARATOR = " "
DEFAULT_MIN_N = 1
DEFAULT_MAX_N = 2


def normalize_token_list(raw: object) -> list[str]:
    if isinstance(raw, list):
        tokens = raw
    elif isinstance(raw, tuple):
        tokens = list(raw)
    elif raw is None:
        return []
    elif isinstance(raw, str):
        text = raw.strip()
        if text == "":
            return []

        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = text.split()

        if isinstance(parsed, list):
            tokens = parsed
        elif isinstance(parsed, tuple):
            tokens = list(parsed)
        else:
            tokens = str(parsed).split()
    else:
        try:
            if pd.isna(raw):
                return []
        except (TypeError, ValueError):
            pass

        try:
            tokens = list(raw)
        except TypeError:
            return []

    return [str(token).strip() for token in tokens if str(token).strip()]


def normalize_sentence_token_lists(raw: object) -> list[list[str]]:
    if raw is None:
        return []

    if isinstance(raw, str):
        text = raw.strip()
        if text == "":
            return []

        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [normalize_token_list(text)]

        raw = parsed

    try:
        if pd.isna(raw):
            return []
    except (TypeError, ValueError):
        pass

    try:
        values = list(raw)
    except TypeError:
        return []

    if not values:
        return []

    first_value = next((value for value in values if value is not None), None)
    if first_value is None:
        return []

    if isinstance(first_value, str):
        return [normalize_token_list(values)]

    try:
        list(first_value)
    except TypeError:
        return [normalize_token_list(values)]

    return [
        tokens
        for tokens in (normalize_token_list(sentence_tokens) for sentence_tokens in values)
        if tokens
    ]


def build_ngrams(
    tokens: list[str],
    min_n: int = DEFAULT_MIN_N,
    max_n: int = DEFAULT_MAX_N,
    separator: str = NGRAM_SEPARATOR,
) -> list[str]:
    if min_n < 1:
        raise ValueError("min_n must be >= 1")
    if max_n < min_n:
        raise ValueError("max_n must be >= min_n")

    ngrams: list[str] = []
    token_count = len(tokens)
    for n in range(min_n, max_n + 1):
        if token_count < n:
            continue

        for start_index in range(token_count - n + 1):
            ngrams.append(separator.join(tokens[start_index : start_index + n]))

    return ngrams


def build_ngram_tf_df_dataframe(
    tokenized_documents: Iterable[object],
    total_documents: int,
    min_n: int = DEFAULT_MIN_N,
    max_n: int = DEFAULT_MAX_N,
    return_summary: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    tf_counter: Counter[str] = Counter()
    df_counter: Counter[str] = Counter()
    documents_with_ngrams = 0

    for raw_document_tokens in tokenized_documents:
        document_ngrams: list[str] = []
        for tokens in normalize_sentence_token_lists(raw_document_tokens):
            ngrams = build_ngrams(tokens, min_n=min_n, max_n=max_n)
            if ngrams:
                document_ngrams.extend(ngrams)

        if not document_ngrams:
            continue

        documents_with_ngrams += 1
        tf_counter.update(document_ngrams)
        df_counter.update(set(document_ngrams))

    summary = {
        "total_documents": total_documents,
        "documents_with_ngrams": documents_with_ngrams,
        "unique_ngrams": len(df_counter),
        "total_ngram_occurrences": sum(tf_counter.values()),
        "min_n": min_n,
        "max_n": max_n,
    }

    records = []
    for term, document_frequency in df_counter.items():
        records.append(
            {
                "term": term,
                "ngram_n": term.count(NGRAM_SEPARATOR) + 1,
                "tf": tf_counter[term],
                "df": document_frequency,
                "df_ratio": document_frequency / total_documents
                if total_documents
                else 0.0,
            }
        )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        out = pd.DataFrame(columns=["term", "ngram_n", "tf", "df", "df_ratio"])
    else:
        out = out.sort_values(
            by=["tf", "df", "ngram_n", "term"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        out = out.reset_index(drop=True)

    if return_summary:
        return out, summary

    return out


def load_tokenized_documents(
    path: Path = INPUT_PATH,
    tokenized_column: str = TOKENIZED_COLUMN,
) -> tuple[pd.DataFrame, str]:
    try:
        return pd.read_parquet(path, columns=[tokenized_column]), tokenized_column
    except Exception:
        if tokenized_column != TOKENIZED_COLUMN:
            raise

        tokenized_column = FALLBACK_TOKENIZED_COLUMN
        return pd.read_parquet(path, columns=[tokenized_column]), tokenized_column


def build_ngram_terms_with_summary(
    path: Path = INPUT_PATH,
    tokenized_column: str = TOKENIZED_COLUMN,
    min_n: int = DEFAULT_MIN_N,
    max_n: int = DEFAULT_MAX_N,
) -> tuple[pd.DataFrame, dict[str, object]]:
    df, selected_tokenized_column = load_tokenized_documents(
        path=path,
        tokenized_column=tokenized_column,
    )
    ngram_terms_df, summary = build_ngram_tf_df_dataframe(
        tokenized_documents=df[selected_tokenized_column],
        total_documents=len(df),
        min_n=min_n,
        max_n=max_n,
        return_summary=True,
    )
    summary["tokenized_column"] = selected_tokenized_column
    return ngram_terms_df, summary


def build_ngram_terms(
    path: Path = INPUT_PATH,
    tokenized_column: str = TOKENIZED_COLUMN,
    min_n: int = DEFAULT_MIN_N,
    max_n: int = DEFAULT_MAX_N,
) -> pd.DataFrame:
    ngram_terms_df, _ = build_ngram_terms_with_summary(
        path=path,
        tokenized_column=tokenized_column,
        min_n=min_n,
        max_n=max_n,
    )
    return ngram_terms_df


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ngram_terms_df, summary = build_ngram_terms_with_summary()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ngram_terms_df.to_parquet(OUTPUT_PATH, index=False)
    ngram_terms_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Input path:", INPUT_PATH)
    print("Tokenized column:", summary["tokenized_column"])
    print("N-gram range:", f"{summary['min_n']} to {summary['max_n']}")
    print("Output parquet:", OUTPUT_PATH)
    print("Total documents:", summary["total_documents"])
    print("Documents with n-grams:", summary["documents_with_ngrams"])
    print("Total n-gram occurrences:", summary["total_ngram_occurrences"])
    print("Unique n-grams:", summary["unique_ngrams"])
    print("N-gram terms:", len(ngram_terms_df))
    print(ngram_terms_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
