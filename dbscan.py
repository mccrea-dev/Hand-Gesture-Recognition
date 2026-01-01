import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# CONFIG
EPS = 1.6
MINPTS = 25
RANDOM_STATE = 42

# INPUT / filtering: load features and keep top-30% activity per gesture
INPUT_CSV = "features_all.csv"
TOP_PERCENT = 0.30
OUTPUT_TOP_CSV = "features_top_activity.csv"

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} samples from {INPUT_CSV}")

LABEL_COL = "label"
df[LABEL_COL] = df["src"].astype(str).apply(lambda s: s.split("_")[0].split("-")[0])
print("Detected gesture labels:", np.unique(df[LABEL_COL].values))

# keep top TOP_PERCENT activity windows per gesture
kept = []
for g in sorted(df[LABEL_COL].unique()):
    block = df[df[LABEL_COL] == g]
    if len(block) == 0:
        continue
    cutoff = np.quantile(block["activity"].values, 1 - TOP_PERCENT)
    top_block = block[block["activity"] >= cutoff]
    kept.append(top_block)

df_top = pd.concat(kept, ignore_index=True)
print(f"After top {int(TOP_PERCENT*100)}% activity filter: {len(df_top)} samples")
df_top.to_csv(OUTPUT_TOP_CSV, index=False)
print(f"Saved filtered features -> {OUTPUT_TOP_CSV}")

# features to use for clustering (exclude metadata)
exclude = ["src", "t_start_ms", "t_end_ms", LABEL_COL]
feature_cols = [c for c in df_top.columns if c not in exclude]
X = df_top[feature_cols].values

# time axis for plotting (seconds, relative)
time_axis = (df_top["t_start_ms"].fillna(0).astype(float).values - df_top["t_start_ms"].min()) / 1000.0

# SCALE 
scaler = StandardScaler().fit(X)
Xz = scaler.transform(X)

# DBSCAN
db = DBSCAN(eps=EPS, min_samples=MINPTS, n_jobs=-1)
labels = db.fit_predict(Xz)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_frac = float(np.mean(labels == -1))
print(f"DBSCAN → clusters={n_clusters}, noise={noise_frac:.1%}")

# silhouette on non-noise when we have at least 2 clusters
unique_labs = set(labels)
if -1 in unique_labs:
    unique_labs.discard(-1)
if len(unique_labs) > 1 and np.sum(labels != -1) > 1:
    try:
        s = silhouette_score(Xz[labels != -1], labels[labels != -1])
        print(f"Silhouette (non-noise): {s:.3f}")
    except Exception:
        pass

# attach labels back to dataframe and save
df_top["dbscan_label"] = labels
df_top.to_csv("training_clean.csv", index=False)
print("Saved training_clean.csv with dbscan_label column")

# PCA(2) map
pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xz)
X2 = pca.transform(Xz)
var2 = pca.explained_variance_ratio_.sum()

plt.figure(figsize=(8,6))
uniq = sorted(set(labels))
for lab in uniq:
    m = (labels == lab)
    if lab == -1:
        plt.scatter(X2[m,0], X2[m,1], s=16, c="k", alpha=0.6, label="Noise")
    else:
        plt.scatter(X2[m,0], X2[m,1], s=16, alpha=0.85, label=f"C{lab}")
plt.title(f"IMU DBSCAN — PCA(2) var≈{var2:.2%} | eps={EPS}, min={MINPTS}")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend(ncol=2, fontsize=8, frameon=True); plt.grid(True, lw=0.3)
plt.tight_layout(); plt.show()

