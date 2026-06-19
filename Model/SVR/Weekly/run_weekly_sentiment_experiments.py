from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parent / "svr_weekly_sentiment_pipeline.py"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "weekly_sentiment_outputs" / "experiments"


def comma_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def make_run_name(target: str, feature_set: str, scaler: str, objective: str) -> str:
    return f"{target}__{feature_set}__{scaler}__{objective}"


def run_one(
    target: str,
    feature_set: str,
    scaler: str,
    objective: str,
    output_root: Path,
    look_back_grid: str,
    c_grid: str,
    gamma_grid: str,
    epsilon_grid: str,
) -> dict[str, object]:
    run_name = make_run_name(target, feature_set, scaler, objective)
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--target",
        target,
        "--feature-set",
        feature_set,
        "--scaler",
        scaler,
        "--tune",
        "--objective",
        objective,
        "--look-back-grid",
        look_back_grid,
        "--c-grid",
        c_grid,
        "--gamma-grid",
        gamma_grid,
        "--epsilon-grid",
        epsilon_grid,
        "--output-dir",
        str(output_dir),
        "--plot-path",
        "weekly_svr_return_prediction.png",
    ]

    print("\n" + "=" * 100)
    print(f"Running: {run_name}")
    print(" ".join(command))
    subprocess.run(command, check=True)

    summary_path = output_dir / "weekly_svr_vnindex_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    row: dict[str, object] = {
        "run_name": run_name,
        "target": target,
        "feature_set": feature_set,
        "scaler": scaler,
        "objective": objective,
        "look_back": summary["look_back"],
        "best_C": summary["best_params"]["C"],
        "best_gamma": summary["best_params"]["gamma"],
        "best_epsilon": summary["best_params"]["epsilon"],
        "feature_count": len(summary["features"]),
        "output_dir": str(output_dir),
        "plot_path": summary.get("plot_path"),
    }

    metric_prefixes = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }
    for split, prefix in metric_prefixes.items():
        metrics = summary[split]
        row[f"{prefix}_mae"] = metrics["mae"]
        row[f"{prefix}_rmse"] = metrics["rmse"]
        row[f"{prefix}_diracc"] = metrics["direction_accuracy"]
        row[f"{prefix}_corr"] = metrics["correlation"]
        row[f"{prefix}_true_std"] = metrics["true_std"]
        row[f"{prefix}_pred_std"] = metrics["pred_std"]

    for split_name, summary_key in [
        ("val", "validation_baseline_summary"),
        ("test", "test_baseline_summary"),
    ]:
        baseline = summary[summary_key]
        row[f"{split_name}_best_mae_baseline"] = baseline["best_mae_baseline"]
        row[f"{split_name}_mae_improvement_pct"] = baseline["model_mae_improvement_pct"]
        row[f"{split_name}_best_rmse_baseline"] = baseline["best_rmse_baseline"]
        row[f"{split_name}_rmse_improvement_pct"] = baseline["model_rmse_improvement_pct"]

    row["train_val_mae_gap"] = row["val_mae"] - row["train_mae"]
    row["train_test_mae_gap"] = row["test_mae"] - row["train_mae"]
    return row


def print_rankings(results: pd.DataFrame) -> None:
    columns = [
        "run_name",
        "look_back",
        "best_C",
        "best_gamma",
        "best_epsilon",
        "val_mae",
        "test_mae",
        "val_corr",
        "test_corr",
        "val_mae_improvement_pct",
        "test_mae_improvement_pct",
        "train_val_mae_gap",
    ]

    print("\nTop by validation MAE improvement")
    print(
        results.sort_values(
            ["val_mae_improvement_pct", "test_mae_improvement_pct"],
            ascending=[False, False],
        )[columns].head(10).to_string(index=False)
    )

    print("\nTop by validation correlation")
    print(
        results.sort_values(
            ["val_corr", "test_corr"],
            ascending=[False, False],
        )[columns].head(10).to_string(index=False)
    )

    print("\nTop by test MAE improvement")
    print(
        results.sort_values(
            ["test_mae_improvement_pct", "val_mae_improvement_pct"],
            ascending=[False, False],
        )[columns].head(10).to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple weekly sentiment SVR tuning experiments and summarize results."
    )
    parser.add_argument("--targets", default="future_ret_1w,future_ret_4w")
    parser.add_argument("--feature-sets", default="market,sentiment,combined")
    parser.add_argument("--scalers", default="standard,robust")
    parser.add_argument("--objectives", default="mae,corr,diracc")
    parser.add_argument("--look-back-grid", default="1,2,4")
    parser.add_argument("--c-grid", default="0.1,0.5,1,2,5")
    parser.add_argument("--gamma-grid", default="scale,auto,0.001,0.005")
    parser.add_argument("--epsilon-grid", default="0.003,0.005,0.01,0.02")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for target in comma_list(args.targets):
        for feature_set in comma_list(args.feature_sets):
            for scaler in comma_list(args.scalers):
                for objective in comma_list(args.objectives):
                    rows.append(
                        run_one(
                            target=target,
                            feature_set=feature_set,
                            scaler=scaler,
                            objective=objective,
                            output_root=args.output_root,
                            look_back_grid=args.look_back_grid,
                            c_grid=args.c_grid,
                            gamma_grid=args.gamma_grid,
                            epsilon_grid=args.epsilon_grid,
                        )
                    )

    results = pd.DataFrame(rows)
    summary_path = args.output_root / "weekly_experiment_summary.csv"
    results.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(f"Saved experiment summary: {summary_path}")
    print_rankings(results)


if __name__ == "__main__":
    main()
