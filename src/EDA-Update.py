import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_excel("/Users/liyansong/Desktop/5450_Final_Project/data/final_table_merged.xlsx")

# plot 1： Delisted vs not delisted

firm_status = df.groupby("permno")["delist"].max()
status_counts = firm_status.value_counts().sort_index()

plt.figure()
bars = plt.bar(["Not Delisted (0)", "Delisted (1)"], status_counts.values)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height,f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=10
    )
plt.ylabel("Number of Firms")
plt.title("Firm-level Distribution: Delisted vs Not Delisted")
plt.tight_layout()
plt.show()


# plot 2.1: The Distribution of Tobin's Q--Without zoom
tq = df["tobinq"].dropna()
plt.hist(tq, bins=40)
plt.title("Distribution of Tobin's Q without zoom")
plt.xlabel("Tobin's Q")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

## plot2.2: The Distribution of Tobin's Q - With zoom
tq_zoom = tq[(tq >= 0) & (tq <= 10)]
plt.hist(tq_zoom, bins=40)
plt.title("Distribution of Tobin's Q (0–10 range)")
plt.xlabel("Tobin's Q")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# plot 3.1: ROA Distribution by Delisting Status -- With Outlier
sns.boxplot(
    data=df,
    x=df['delist'].map({0:'Not Delisted', 1:'Delisted'}),
    y='roa',
    palette='Set2',
    showfliers=True
)
plt.title("ROA Distribution (Outliers Shown)", fontsize=16)
plt.xlabel("")
plt.ylabel("ROA")
plt.show()

# plot 3.2: ROA Distribution by Delisting Status -- Without Outlier
sns.boxplot(
    data=df,
    x=df['delist'].map({0:'Not Delisted', 1:'Delisted'}),
    y='roa',
    palette='Set2',
    showfliers=False
)
plt.title("ROA Distribution (Outliers Hidden)", fontsize=16)
plt.xlabel("")
plt.ylabel("ROA")
plt.show()

# plot Four：Delisting Rate Over Time
if "public_year" in df.columns:
    year_delist = df.groupby("public_year")["delist"].mean()

    plt.plot(year_delist.index, year_delist.values, marker="o")
    plt.title("Delisting Rate Over Time")
    plt.xlabel("Year")
    plt.ylabel("Delisting Rate")
    plt.tight_layout()
    plt.show()

# plot Five：Correlation Heatmap
# --- Correlation Heatmap with Annotations ---
plt.figure(figsize=(10, 6))

corr_cols = ["tobinq", "revtq", "bm", "roa", "roe", "de_ratio", "pe_op_basic"]
corr_matrix = df[corr_cols].corr()

plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")

# Tick labels
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
plt.yticks(range(len(corr_cols)), corr_cols)

# ---- Add Annotation ----
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        value = corr_matrix.iloc[i, j]
        plt.text(
            j, i, f"{value:.2f}",
            ha="center", va="center",
            color="black" if abs(value) < 0.5 else "white",
            fontsize=9
        )

plt.title("Correlation Matrix of Key Financial Ratios (Annotated)")
plt.tight_layout()
plt.show()


# plot six

sns.set_style("whitegrid")

# 1. Cut off the outliers，improve the readability of the graph
df = df.copy()
df["roa_clip"] = df["roa"].clip(-1, 1)

# Build 10 bins, and split the ROA equally
bins = np.linspace(-1.0, 1.0, 11)  # -1.0, -0.8, ..., 0.8, 1.0
bin_labels = [f"{bins[i]:.1f} ~ {bins[i+1]:.1f}" for i in range(len(bins) - 1)]

df["roa_bin"] = pd.cut(
    df["roa_clip"],
    bins=bins,
    labels=bin_labels,
    include_lowest=True
)

# 3. 计算每个 bin 的退市率 + 样本量
bin_stats = (
    df.groupby("roa_bin", observed=False)
      .agg(
          delist_rate=("delist", "mean"),   # Delisting rate
          n_obs=("delist", "size")          # the number of observations
      )
      .reset_index()
)

# plot
plt.figure(figsize=(10, 5))

ax = sns.barplot(
    data=bin_stats,
    x="roa_bin",
    y="delist_rate",
    color="C0"
)

# y 轴显示百分比
ax.set_ylabel("Delisting Rate", fontsize=12)
ax.set_xlabel("ROA (binned)", fontsize=12)
ax.set_title("Delisting Rate Across ROA Bins", fontsize=14)

# x 轴标签旋转，防止重叠
plt.xticks(rotation=45, ha="right")

# 在柱子上标注退市率（百分比）和样本量 n
for i, row in bin_stats.iterrows():
    x = i
    y = row["delist_rate"]
    ax.text(
        x,
        y + 0.002,
        f"{y*100:.1f}%\n(n={row['n_obs']})",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()
plt.show()

