import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset/data_VNINDEX_processed.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Return đang là tỷ lệ thập phân: 0.01 = 1%
df["Growth_Index"] = (1 + df["Return"]).cumprod()

plt.figure(figsize=(14, 6))
plt.plot(df["Date"], df["Growth_Index"], color="blue", label="VNINDEX growth from return")
plt.axhline(1, color="gray", linewidth=1, linestyle="--")

plt.title("VNINDEX Growth Based on Daily Return")
plt.xlabel("Date")
plt.ylabel("Growth Index")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()