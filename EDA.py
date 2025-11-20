import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_excel("/Users/liyansong/Desktop/final_table_merged.xlsx")

# plot one： Delisted vs not delisted

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


# plot two: The Distribution of Tobin's Q
tq = df["tobinq"].dropna()
tq_zoom = tq[(tq >= 0) & (tq <= 10)]

plt.hist(tq_zoom, bins=40)
plt.title("Distribution of Tobin's Q (0–10 range)")
plt.xlabel("Tobin's Q")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# plot Three: ROA Distribution by Delisting Status
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
corr_cols = ["tobinq", "revtq", "bm", "roa", "roe", "de_ratio", "pe_op_basic"]
corr_cols = [c for c in corr_cols if c in df.columns]

corr = df[corr_cols].corr()

plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()

plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
plt.yticks(range(len(corr_cols)), corr_cols)

plt.title("Correlation Matrix of Key Financial Ratios")
plt.tight_layout()
plt.show()

