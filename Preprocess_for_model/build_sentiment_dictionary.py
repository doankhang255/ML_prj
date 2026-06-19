from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_PATH = PROJECT_ROOT / "Dataset" / "candidate_ngram_terms.parquet"
DEFAULT_LOCAL_MODEL_PATH = (
    SCRIPT_DIR
    / "hub"
    / "models--wonrax--phobert-base-vietnamese-sentiment"
    / "snapshots"
    / "9076a5896971b5d551588fe8a51c722c89731d36"
)
DEFAULT_MODEL_NAME = "wonrax/phobert-base-vietnamese-sentiment"
MODEL_PATH = DEFAULT_LOCAL_MODEL_PATH if DEFAULT_LOCAL_MODEL_PATH.exists() else DEFAULT_MODEL_NAME

OUTPUT_DICTIONARY_PARQUET_PATH = (
    PROJECT_ROOT / "Dataset" / "candidate_ngram_terms_dictionary.parquet"
)
OUTPUT_DICTIONARY_CSV_PATH = PROJECT_ROOT / "Dataset" / "candidate_ngram_terms_dictionary.csv"

BATCH_SIZE = 128
MAX_LENGTH = 32

LABEL_MAP = {
    "NEG": "negative",
    "POS": "positive",
    "NEU": "neutral",
}
VALID_LABELS = {"positive", "negative", "neutral"}


def normalize_term(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def load_candidate_terms(path: Path = INPUT_PATH) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
    else:
        df = pd.read_parquet(path)

    required_columns = {"term", "ngram_n"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Candidate terms are missing columns: {sorted(missing_columns)}")

    out = df.copy()
    out["term"] = out["term"].astype("string").fillna("").str.strip()
    out["term_norm"] = out["term"].apply(normalize_term)
    out["ngram_n"] = pd.to_numeric(out["ngram_n"], errors="coerce").astype("Int64")
    out = out.loc[out["term_norm"].ne("") & out["ngram_n"].isin([1, 2])].copy()
    return out.reset_index(drop=True)


def predict_term_sentiment(
    candidate_terms_df: pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    local_files_only: bool | None = None,
) -> pd.DataFrame:
    model_ref = str(model_path)
    model_path_obj = Path(model_ref)
    is_existing_local_path = model_path_obj.exists()
    looks_like_missing_local_path = "\\" in model_ref or ":" in model_ref
    if looks_like_missing_local_path and not is_existing_local_path:
        raise FileNotFoundError(
            "Local sentiment model path does not exist: "
            f"{model_ref}\n"
            "Pass --model wonrax/phobert-base-vietnamese-sentiment to download/use "
            "the HuggingFace model, or pass the correct local snapshot folder."
        )

    if local_files_only is None:
        local_files_only = is_existing_local_path

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        use_fast=False,
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id_to_label = {
        int(label_id): LABEL_MAP.get(str(label).upper(), str(label).casefold())
        for label_id, label in model.config.id2label.items()
    }
    texts = candidate_terms_df["term_norm"].tolist()

    labels: list[str] = []
    confidence_scores: list[float] = []
    prob_rows: list[dict[str, float]] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Label candidate terms"):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_ids = probs.argmax(dim=-1).cpu().tolist()
            batch_probs = probs.cpu().tolist()

            for pred_id, prob_values in zip(pred_ids, batch_probs, strict=False):
                label = id_to_label[pred_id]
                labels.append(label)
                confidence_scores.append(float(max(prob_values)))

                prob_row = {f"{label_name}_prob": 0.0 for label_name in VALID_LABELS}
                for label_id, probability in enumerate(prob_values):
                    label_name = id_to_label[label_id]
                    prob_row[f"{label_name}_prob"] = float(probability)
                prob_rows.append(prob_row)

    out = candidate_terms_df.copy()
    out["sentiment_label"] = labels
    out["sentiment_confidence"] = confidence_scores
    out = pd.concat([out, pd.DataFrame(prob_rows)], axis=1)
    return out


def build_sentiment_dictionary(
    input_path: Path = INPUT_PATH,
    model_path: str | Path = MODEL_PATH,
    local_files_only: bool | None = None,
) -> pd.DataFrame:
    candidate_terms_df = load_candidate_terms(input_path)
    labeled_terms_df = predict_term_sentiment(
        candidate_terms_df,
        model_path=model_path,
        local_files_only=local_files_only,
    )
    dictionary_df = labeled_terms_df.copy()

    dictionary_df = dictionary_df.drop(columns=["term_norm"])
    return dictionary_df.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sentiment dictionary for candidate n-gram terms."
    )
    parser.add_argument("--input", default=INPUT_PATH, type=Path)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--output-parquet", default=OUTPUT_DICTIONARY_PARQUET_PATH, type=Path)
    parser.add_argument("--output-csv", default=OUTPUT_DICTIONARY_CSV_PATH, type=Path)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load an already downloaded/local HuggingFace model.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    local_files_only = True if args.local_files_only else None
    dictionary_df = build_sentiment_dictionary(
        input_path=args.input,
        model_path=args.model,
        local_files_only=local_files_only,
    )

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    dictionary_df.to_parquet(args.output_parquet, index=False)
    dictionary_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print("Input:", args.input)
    print("Model:", args.model)
    print("Dictionary parquet:", args.output_parquet)
    print("Dictionary csv:", args.output_csv)
    print("Dictionary terms:", len(dictionary_df))
    label_counts = dictionary_df["sentiment_label"].value_counts(dropna=False)
    print("Positive terms:", int(label_counts.get("positive", 0)))
    print("Negative terms:", int(label_counts.get("negative", 0)))
    print("Neutral terms:", int(label_counts.get("neutral", 0)))
    print("Dictionary label counts:")
    print(label_counts.to_string())
    print(dictionary_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
