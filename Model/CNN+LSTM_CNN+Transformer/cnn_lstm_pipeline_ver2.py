from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from vnindex_feature_utils import (  # noqa: E402
    DEFAULT_PROCESSED_DATA_PATH,
    FEATURE_COLUMNS,
    RETURN_1D_TARGET_COLUMN,
    SPLIT_RATIOS,
    create_sequence_windows,
    inverse_target,
    load_feature_data,
    scale_feature_splits,
    split_dataframe,
)


class CNNLSTMRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        filters: int,
        kernel_size: int,
        lstm_units: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=filters,
            hidden_size=lstm_units,
            batch_first=True,
        )
        self.output = nn.Linear(lstm_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x arrives as (batch, days, features). Conv1d expects (batch, features, days).
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        # LSTM expects (batch, days, channels).
        x = x.permute(0, 2, 1)
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        return self.output(x)


def split_internal_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    internal_val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    val_size = max(1, int(len(x_train) * internal_val_ratio))
    return x_train[:-val_size], y_train[:-val_size], x_train[-val_size:], y_train[-val_size:]


def make_loader(
    x_values: np.ndarray,
    y_values: np.ndarray,
    batch_size: int,
) -> DataLoader:
    x_tensor = torch.tensor(x_values, dtype=torch.float32)
    y_tensor = torch.tensor(y_values, dtype=torch.float32).reshape(-1, 1)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_direction = np.asarray(y_true).ravel() > 0
    pred_direction = np.asarray(y_pred).ravel() > 0
    return float(np.mean(true_direction == pred_direction))


def evaluate_return_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "direction_accuracy": direction_accuracy(y_true, y_pred),
        "mean_true_return": float(np.mean(y_true)),
        "mean_pred_return": float(np.mean(y_pred)),
    }


def train_model(
    model: nn.Module,
    fit_x_train: np.ndarray,
    fit_y_train: np.ndarray,
    internal_x_val: np.ndarray,
    internal_y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, list[float]]:
    train_loader = make_loader(fit_x_train, fit_y_train, batch_size)
    val_x = torch.tensor(internal_x_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(internal_y_val, dtype=torch.float32, device=device).reshape(-1, 1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(batch_losses))
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x), val_y).detach().cpu())

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch}/{epochs} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return history


def predict_scaled(model: nn.Module, x_values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    x_tensor = torch.tensor(x_values, dtype=torch.float32, device=device)
    preds = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, len(x_tensor), batch_size):
            pred = model(x_tensor[start : start + batch_size])
            preds.append(pred.detach().cpu().numpy())
    return np.vstack(preds).ravel()


def predict_and_evaluate(
    model: nn.Module,
    x_values: np.ndarray,
    y_values: np.ndarray,
    y_scaler,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pred_scaled = predict_scaled(model, x_values, device)
    y_true = inverse_target(y_values, y_scaler)
    y_pred = inverse_target(pred_scaled, y_scaler)
    return y_true, y_pred, evaluate_return_predictions(y_true, y_pred)


def cumulative_return(daily_returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1 + np.asarray(daily_returns).ravel()) - 1


def cumulative_return_from_base(base_return: float, daily_returns: np.ndarray) -> np.ndarray:
    return (1 + base_return) * np.cumprod(1 + np.asarray(daily_returns).ravel()) - 1


def window_target_dates(split_df: pd.DataFrame, look_back: int) -> pd.Series:
    date_column = "Target_Date_1D" if "Target_Date_1D" in split_df.columns else "Date"
    return pd.to_datetime(split_df[date_column].iloc[look_back - 1 :], errors="coerce")


def window_current_returns(split_df: pd.DataFrame, look_back: int) -> np.ndarray:
    return split_df["Return"].iloc[look_back - 1 :].to_numpy(dtype=float)


def plot_cumulative_predictions(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    look_back: int,
    split_code: str,
    output_path: str | Path,
) -> None:
    date_column = "Target_Date_1D" if "Target_Date_1D" in train.columns else "Date"
    all_dates = pd.to_datetime(
        pd.concat(
            [train[date_column], val[date_column], test[date_column]],
            ignore_index=True,
        ),
        errors="coerce",
    )
    all_true_returns = pd.concat(
        [
            train[RETURN_1D_TARGET_COLUMN],
            val[RETURN_1D_TARGET_COLUMN],
            test[RETURN_1D_TARGET_COLUMN],
        ],
        ignore_index=True,
    ).to_numpy(dtype=float)
    all_true_cum = cumulative_return(all_true_returns)

    val_dates = window_target_dates(val, look_back)
    test_dates = window_target_dates(test, look_back)
    val_start_pos = len(train) + look_back - 1
    test_start_pos = len(train) + len(val) + look_back - 1
    val_base = all_true_cum[val_start_pos - 1] if val_start_pos > 0 else 0.0
    test_base = all_true_cum[test_start_pos - 1] if test_start_pos > 0 else 0.0
    val_pred_cum = cumulative_return_from_base(val_base, val_pred)
    test_pred_cum = cumulative_return_from_base(test_base, test_pred)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.plot(all_dates, all_true_cum * 100, label="VNINDEX actual cumulative return", color="blue")
    ax.plot(val_dates, val_pred_cum * 100, label="Validation predicted cumulative", color="purple")
    ax.plot(test_dates, test_pred_cum * 100, label="Test predicted cumulative", color="green")
    ax.set_title(f"PyTorch CNN+LSTM - Predicted vs Actual VNINDEX Cumulative Return ({split_code})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_pipeline(
    csv_path: str | Path,
    split_code: str,
    look_back: int = 30,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    internal_val_ratio: float = 0.15,
    plot_dir: str | Path | None = None,
    seed: int = 42,
    filters: int = 64,
    kernel_size: int = 3,
    lstm_units: int = 100,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    device_name: str = "auto",
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    df = load_feature_data(csv_path, target_column=RETURN_1D_TARGET_COLUMN)
    train, val, test, split_sizes = split_dataframe(df, split_code)
    x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, x_test_scaled, y_test_scaled, _, y_scaler = scale_feature_splits(
        train,
        val,
        test,
        target_column=RETURN_1D_TARGET_COLUMN,
    )

    x_train_full, y_train_full = create_sequence_windows(x_train_scaled, y_train_scaled, look_back)
    x_val, y_val = create_sequence_windows(x_val_scaled, y_val_scaled, look_back)
    x_test, y_test = create_sequence_windows(x_test_scaled, y_test_scaled, look_back)
    fit_x_train, fit_y_train, internal_x_val, internal_y_val = split_internal_validation(
        x_train_full,
        y_train_full,
        internal_val_ratio,
    )
    if len(fit_x_train) == 0:
        raise ValueError(
            "Internal validation split consumed all training windows. "
            "Reduce --internal-val-ratio or --look-back."
        )

    model = CNNLSTMRegressor(
        num_features=x_train_full.shape[2],
        filters=filters,
        kernel_size=kernel_size,
        lstm_units=lstm_units,
        dropout=dropout,
    ).to(device)
    history = train_model(
        model=model,
        fit_x_train=fit_x_train,
        fit_y_train=fit_y_train,
        internal_x_val=internal_x_val,
        internal_y_val=internal_y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        learning_rate=learning_rate,
        device=device,
    )

    val_true, val_pred, val_metrics = predict_and_evaluate(model, x_val, y_val, y_scaler, device)
    test_true, test_pred, test_metrics = predict_and_evaluate(model, x_test, y_test, y_scaler, device)

    current_return_val = window_current_returns(val, look_back)
    current_return_test = window_current_returns(test, look_back)
    train_return_mean = float(train[RETURN_1D_TARGET_COLUMN].mean())
    mean_val = np.full_like(val_true, train_return_mean)
    mean_test = np.full_like(test_true, train_return_mean)
    zero_val = np.zeros_like(val_true)
    zero_test = np.zeros_like(test_true)

    current_baseline_val = evaluate_return_predictions(val_true, current_return_val)
    current_baseline_test = evaluate_return_predictions(test_true, current_return_test)
    mean_baseline_val = evaluate_return_predictions(val_true, mean_val)
    mean_baseline_test = evaluate_return_predictions(test_true, mean_test)
    zero_baseline_val = evaluate_return_predictions(val_true, zero_val)
    zero_baseline_test = evaluate_return_predictions(test_true, zero_test)

    plot_path = None
    if plot_dir is not None:
        plot_path = Path(plot_dir) / f"cnn_lstm_pytorch_daily_return_cumulative_{split_code}.png"
        plot_cumulative_predictions(
            train,
            val,
            test,
            val_pred,
            test_pred,
            look_back,
            split_code,
            plot_path,
        )

    return {
        "model": model,
        "history": history,
        "split": split_code,
        "device": str(device),
        "epochs_ran": len(history["loss"]),
        "target": RETURN_1D_TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "split_sizes": split_sizes,
        "sequence_shapes": {
            "x_train": x_train_full.shape,
            "x_val": x_val.shape,
            "x_test": x_test.shape,
        },
        "internal_train_windows": len(fit_x_train),
        "internal_val_windows": len(internal_x_val),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "current_baseline_val": current_baseline_val,
        "current_baseline_test": current_baseline_test,
        "mean_baseline_val": mean_baseline_val,
        "mean_baseline_test": mean_baseline_test,
        "zero_baseline_val": zero_baseline_val,
        "zero_baseline_test": zero_baseline_test,
        "plot_path": plot_path,
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PyTorch CNN+LSTM next-day return prediction and plot cumulative return."
    )
    parser.add_argument("--data", default=str(DEFAULT_PROCESSED_DATA_PATH), help="Path to processed CSV data.")
    parser.add_argument("--split", choices=["652510", "702010", "751510", "all"], default="702010")
    parser.add_argument("--look-back", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--internal-val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--lstm-units", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="auto", help="Use auto, cpu, cuda, or cuda:0.")
    parser.add_argument(
        "--plot-dir",
        default=str(Path(__file__).with_name("cnn_lstm_outputs")),
        help="Directory for prediction plots. Use empty string to skip plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_codes = list(SPLIT_RATIOS) if args.split == "all" else [args.split]
    plot_dir = args.plot_dir or None

    print(f"PyTorch version: {torch.__version__}")
    print(f"Data: {args.data}")
    print(f"Target: {RETURN_1D_TARGET_COLUMN}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Look-back sequence length: {args.look_back} trading days")

    for split_code in split_codes:
        print(f"\n=== PyTorch CNN+LSTM ver2 {split_code} ===")
        result = run_pipeline(
            csv_path=args.data,
            split_code=split_code,
            look_back=args.look_back,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            internal_val_ratio=args.internal_val_ratio,
            plot_dir=plot_dir,
            seed=args.seed,
            filters=args.filters,
            kernel_size=args.kernel_size,
            lstm_units=args.lstm_units,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            device_name=args.device,
        )
        print(f"Device: {result['device']}")
        print(f"Epochs ran: {result['epochs_ran']}")
        print(f"Split sizes: {result['split_sizes']}")
        print(f"Sequence shapes: {result['sequence_shapes']}")
        print(
            "Internal early-stopping windows: "
            f"train={result['internal_train_windows']}, "
            f"val={result['internal_val_windows']}"
        )
        print_metrics("Validation metrics:", result["val_metrics"])
        print_metrics("Test metrics:", result["test_metrics"])
        print_metrics("Current-return validation:", result["current_baseline_val"])
        print_metrics("Current-return test:", result["current_baseline_test"])
        print_metrics("Train-mean validation:", result["mean_baseline_val"])
        print_metrics("Train-mean test:", result["mean_baseline_test"])
        print_metrics("Zero validation:", result["zero_baseline_val"])
        print_metrics("Zero test:", result["zero_baseline_test"])
        if result["plot_path"]:
            print(f"Plot saved to: {result['plot_path']}")


if __name__ == "__main__":
    main()
