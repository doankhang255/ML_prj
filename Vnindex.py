from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "Dataset" / "data_VNINDEX_processed.csv"
OUTPUT_PATH = PROJECT_ROOT / "Dataset" / "vnindex_cumulative_return.png"


df = pd.read_csv(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Return is stored as a decimal ratio: 0.01 means 1%.
df["Cumulative_Return"] = (1 + df["Return"]).cumprod() - 1

plt.figure(figsize=(14, 6))
plt.plot(
    df["Date"],
    df["Cumulative_Return"] * 100,
    color="blue",
    label="VNINDEX cumulative return",
)
plt.axhline(0, color="gray", linewidth=1, linestyle="--")
plt.title("VNINDEX Cumulative Return Based on Daily Return")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
plt.close()

print(f"Saved chart to: {OUTPUT_PATH}")
