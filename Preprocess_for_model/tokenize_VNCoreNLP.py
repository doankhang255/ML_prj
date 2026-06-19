from pathlib import Path
import os
import re
import pandas as pd
import py_vncorenlp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "Dataset" / "equity_news_clean_content.parquet"
OUTPUT_PATH = PROJECT_ROOT / "Dataset" / "equity_news_tokenized_vncorenlp.parquet"

TEXT_COLUMN = "content"
TOKENIZED_COLUMN = "Tokenize_content"
TOKENIZED_SENTENCES_COLUMN = "Tokenize_content_sentences"
TOTAL_TOKENIZER_COLUMN = "total_tokenizer"

VNCORENLP_MODEL_DIR = Path(__file__).resolve().parent / "vncorenlp"

TOKEN_PATTERN = re.compile(r"[^\W\d_]", flags=re.UNICODE)
SPECIAL_CHARACTER_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+", flags=re.UNICODE)
NUMBER_PATTERN = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?:%|đ|vnđ)?(?!\w)", flags=re.UNICODE | re.IGNORECASE)
SENTENCE_SPLIT_PATTERN = re.compile(r"\.+", flags=re.UNICODE)


def clean_token(token: str) -> str | None:
    token = SPECIAL_CHARACTER_PATTERN.sub(" ", token.strip().lower())
    token = WHITESPACE_PATTERN.sub("_", token).strip("_")

    if len(token) <= 1:
        return None

    if not TOKEN_PATTERN.search(token):
        return None

    if token[0].isdigit():
        return None

    return token


def configure_java_home() -> None:
    if os.environ.get("JAVA_HOME"):
        return

    java_home_candidates = [
        Path(r"C:\Program Files\Java\latest"),
        *sorted(Path(r"C:\Program Files\Java").glob("jdk-*"), reverse=True),
    ]
    for java_home in java_home_candidates:
        java_bin = java_home / "bin" / "java.exe"
        if java_bin.exists():
            os.environ["JAVA_HOME"] = str(java_home)
            os.environ["PATH"] = f"{java_home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
            return


def load_vncorenlp_segmenter() -> py_vncorenlp.VnCoreNLP:
    configure_java_home()
    return py_vncorenlp.VnCoreNLP(
        annotators=["wseg"],
        save_dir=str(VNCORENLP_MODEL_DIR),
    )


RDRSEGMENTER = load_vncorenlp_segmenter()


def remove_numbers_before_sentence_split(text: str) -> str:
    text = NUMBER_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def split_text_by_period(text: object) -> list[str]:
    if pd.isna(text):
        return []

    text = remove_numbers_before_sentence_split(str(text).strip())
    if not text:
        return []

    return [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_PATTERN.split(text)
        if sentence.strip()
    ]


def tokenize_sentence(sentence: str) -> list[str]:
    segmented_sentences = RDRSEGMENTER.word_segment(sentence)
    segmented_text = " ".join(segmented_sentences)
    raw_tokens = segmented_text.split()
    cleaned_tokens = [clean_token(token) for token in raw_tokens]
    return [token for token in cleaned_tokens if token is not None]


def tokenize_vietnamese_text_by_sentence(text: object) -> list[list[str]]:
    tokenized_sentences: list[list[str]] = []
    for sentence in split_text_by_period(text):
        try:
            tokens = tokenize_sentence(sentence)
        except Exception as e:
            print(f"Tokenize error: {e}")
            tokens = []

        if tokens:
            tokenized_sentences.append(tokens)

    return tokenized_sentences


def tokenize_vietnamese_text(text: object) -> list[str]:
    tokenized_sentences = tokenize_vietnamese_text_by_sentence(text)
    return [token for sentence in tokenized_sentences for token in sentence]


def load_and_tokenize_equity_news(path: Path = INPUT_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not found in input file.")

    out = df.copy()
    out[TOKENIZED_SENTENCES_COLUMN] = out[TEXT_COLUMN].apply(tokenize_vietnamese_text_by_sentence)
    out[TOKENIZED_COLUMN] = out[TOKENIZED_SENTENCES_COLUMN].apply(
        lambda sentences: [token for sentence in sentences for token in sentence]
    )
    out[TOTAL_TOKENIZER_COLUMN] = out[TOKENIZED_COLUMN].apply(len)

    return out


def main() -> None:
    df = load_and_tokenize_equity_news(INPUT_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved tokenized dataframe to: {OUTPUT_PATH}")
    print("Rows:", len(df))
    print("Total sentences:", df[TOKENIZED_SENTENCES_COLUMN].apply(len).sum())
    print("Total tokens:", df[TOTAL_TOKENIZER_COLUMN].sum())


if __name__ == "__main__":
    main()
