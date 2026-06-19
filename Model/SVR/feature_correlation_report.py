from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEKLY_PIPELINE_PATH = Path(__file__).resolve().parent / "Weekly" / "svr_weekly_sentiment_pipeline.py"
DAILY_PIPELINE_PATH = Path(__file__).resolve().parent / "Daily" / "svr_daily_sentiment_pipeline.py"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "feature_correlation_outputs"

WEEKLY_SHOCK_FEATURE_COLUMNS = [
    "sentiment_z_shock_1w",
    "sentiment_shock_1w",
    "sentiment_shock_vs_ma4",
    "extreme_negative_sentiment",
    "extreme_positive_sentiment",
    "negative_attention",
    "positive_attention",
    "negative_ratio_change_1w",
    "positive_ratio_change_1w",
    "news_attention_shock",
    "volume_shock_12w",
    "range_shock_12w",
    "return_shock_z_12w",
    "large_down_week",
    "large_up_week",
    "negative_sentiment_market_stress",
    "negative_sentiment_volume_shock",
    "bad_news_after_down_week",
    "good_news_after_up_week",
]

DAILY_SHOCK_FEATURE_COLUMNS = [
    "sentiment_z_shock_1d",
    "sentiment_shock_vs_ma5",
    "extreme_negative_sentiment",
    "negative_attention",
    "negative_ratio_change_1d",
    "news_attention_shock",
    "volume_shock_20d",
    "range_shock_20d",
    "return_shock_z_20d",
    "large_down_day",
]


def import_module_from_path(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_pipeline(frequency: str) -> ModuleType:
    if frequency == "weekly":
        return import_module_from_path(WEEKLY_PIPELINE_PATH, "svr_weekly_sentiment_pipeline")
    if frequency == "daily":
        return import_module_from_path(DAILY_PIPELINE_PATH, "svr_daily_sentiment_pipeline")
    raise ValueError("frequency must be weekly or daily")


def comma_list(text: str | None) -> list[str]:
    if text is None or str(text).strip() == "":
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def load_feature_frame(
    pipeline: ModuleType,
    frequency: str,
    data_path: Path | None,
    feature_set: str,
    extra_features: Sequence[str],
    drop_features: Sequence[str],
) -> tuple[pd.DataFrame, list[str], Path]:
    resolved_data_path = data_path or Path(pipeline.DEFAULT_DATA_PATH)
    if frequency == "weekly":
        df = pipeline.read_weekly_data(resolved_data_path)
    else:
        df = pipeline.read_daily_data(resolved_data_path)

    feature_columns = pipeline.build_feature_columns(
        df,
        feature_set=feature_set,
        extra_features=extra_features,
        drop_features=drop_features,
    )
    feature_df = df[feature_columns].replace([np.inf, -np.inf], np.nan).dropna()
    if feature_df.empty:
        raise ValueError("No rows left after dropping NaN/inf values from selected features.")
    return feature_df, feature_columns, Path(resolved_data_path)


def correlation_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            abs_value = abs(float(value))
            if abs_value >= threshold:
                rows.append(
                    {
                        "feature_1": left,
                        "feature_2": right,
                        "correlation": float(value),
                        "abs_correlation": abs_value,
                    }
                )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)


def save_heatmap(corr: pd.DataFrame, output_path: Path, title: str) -> None:
    n = len(corr.columns)
    fig_width = max(10, min(28, n * 0.65))
    fig_height = max(8, min(26, n * 0.58))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(corr.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title(title)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlation")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_feature_list(title: str, features: Sequence[str]) -> None:
    print(f"\n{title} ({len(features)} features)")
    for idx, feature in enumerate(features, start=1):
        print(f"  {idx:02d}. {feature}")


def run_report(args: argparse.Namespace) -> dict[str, Path | int | str]:
    pipeline = load_pipeline(args.frequency)
    extra_features = comma_list(args.extra_features)
    if args.include_shock_features:
        if args.frequency == "weekly":
            extra_features.extend(WEEKLY_SHOCK_FEATURE_COLUMNS)
        else:
            extra_features.extend(DAILY_SHOCK_FEATURE_COLUMNS)

    feature_df, feature_columns, data_path = load_feature_frame(
        pipeline=pipeline,
        frequency=args.frequency,
        data_path=args.data,
        feature_set=args.feature_set,
        extra_features=extra_features,
        drop_features=comma_list(args.drop_features),
    )

    corr = feature_df.corr(method=args.method)
    high_pairs = correlation_pairs(corr, args.threshold)

    output_dir = Path(args.output_dir) / args.frequency / args.feature_set
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / f"{args.frequency}_{args.feature_set}_{args.method}_correlation_matrix.csv"
    pairs_path = output_dir / f"{args.frequency}_{args.feature_set}_high_correlation_pairs.csv"
    heatmap_path = output_dir / f"{args.frequency}_{args.feature_set}_{args.method}_correlation_heatmap.png"

    corr.to_csv(matrix_path, encoding="utf-8-sig")
    high_pairs.to_csv(pairs_path, index=False, encoding="utf-8-sig")
    save_heatmap(
        corr,
        heatmap_path,
        title=f"{args.frequency.title()} {args.feature_set} feature correlation ({args.method})",
    )

    print(f"\nData: {data_path}")
    print(f"Frequency: {args.frequency}")
    print(f"Feature set: {args.feature_set}")
    print(f"Correlation method: {args.method}")
    print(f"Rows used: {len(feature_df)}")
    print_feature_list("Selected features", feature_columns)
    print(f"\nCorrelation matrix: {matrix_path}")
    print(f"High-correlation pairs: {pairs_path}")
    print(f"Heatmap: {heatmap_path}")

    if high_pairs.empty:
        print(f"\nNo feature pairs with abs(correlation) >= {args.threshold}.")
    else:
        print(f"\nTop correlated pairs abs(correlation) >= {args.threshold}")
        print(high_pairs.head(args.top_n).to_string(index=False))

    return {
        "matrix_path": matrix_path,
        "pairs_path": pairs_path,
        "heatmap_path": heatmap_path,
        "rows_used": len(feature_df),
        "feature_count": len(feature_columns),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create feature correlation reports for weekly/daily sentiment SVR features."
    )
    parser.add_argument("--frequency", choices=["weekly", "daily"], required=True)
    parser.add_argument("--feature-set", default="combined", choices=["sentiment", "market", "combined"])
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--method", default="pearson", choices=["pearson", "spearman"])
    parser.add_argument("--threshold", default=0.85, type=float)
    parser.add_argument("--top-n", default=30, type=int)
    parser.add_argument("--extra-features", default="")
    parser.add_argument("--drop-features", default="")
    parser.add_argument(
        "--include-shock-features",
        action="store_true",
        help="Append derived shock features to the selected feature set before computing correlation.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_report(args)


if __name__ == "__main__":
    main()
