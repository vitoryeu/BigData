import pandas as pd, matplotlib.pyplot as plt, os
os.makedirs("output/visualizations", exist_ok=True)

tsv = "output/page_counts_sorted.tsv"
png = "output/visualizations/top_pages.png"

df = pd.read_csv(tsv, sep="\t", names=["page","count"], header=None)
top10 = df.sort_values("count", ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.bar(top10["page"], top10["count"])
plt.title("TOP-10 сторінок")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(png)
print("Saved", png)
