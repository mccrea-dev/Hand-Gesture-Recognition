#!/usr/bin/env python3
#you will need to use aspects of this program in your final project.
# please pirate as needed

import os, sys, glob, json
import numpy as np, pandas as pd

WIN = 25     # 100 Hz ⇒ 250 ms
HOP = 12     # 50% overlap
AXES = ["ax","ay","az","gx","gy","gz"]

def window_ix(n, win=WIN, hop=HOP):
    i = 0
    while i + win <= n:
        yield i, i + win
        i += hop

def features_from_csv(path):
    df = pd.read_csv(path)
    cols = [c for c in AXES if c in df.columns]
    if len(cols) < 3:
        raise SystemExit(f"Not enough axis columns in {path} (found: {cols})")
    A = df[cols].astype(float).to_numpy()
    out_rows = []
    for s, e in window_ix(len(A)):
        seg = A[s:e]
        d = np.diff(seg, axis=0)
        mean = seg.mean(0)
        std  = seg.std(0)
        rms  = np.sqrt((seg**2).mean(0))
        drms = np.sqrt((d**2).mean(0))
        p2p  = (seg.max(0) - seg.min(0))
        activity = float(np.sqrt((d**2).sum(1)).mean())  # simple activity proxy
        row = {
            "src": os.path.basename(path),
            "t_start_ms": float(df.iloc[s, 0]),
            "t_end_ms":   float(df.iloc[e-1, 0]),
            "activity":   activity,
        }
        # add per-axis features
        for i, c in enumerate(cols):
            row[f"mean_{c}"] = mean[i]
            row[f"std_{c}"]  = std[i]
            row[f"rms_{c}"]  = rms[i]
            row[f"drms_{c}"] = drms[i]
            row[f"p2p_{c}"]  = p2p[i]
        out_rows.append(row)
    return pd.DataFrame(out_rows)

def main():
    inpaths = sys.argv[1:] or glob.glob("*.csv")
    if not inpaths:
        print("No CSVs found. Usage: python step1_features_csv.py file1.csv file2.csv ...")
        sys.exit(2)

    os.makedirs("features", exist_ok=True)
    all_rows = []
    for p in inpaths:
        print("→", p)
        feat = features_from_csv(p)
        outp = os.path.join("features", os.path.splitext(os.path.basename(p))[0] + ".csv")
        # stable column order: meta, then sorted feature names
        meta = ["src", "t_start_ms", "t_end_ms", "activity"]
        feats = sorted([c for c in feat.columns if c not in meta])
        feat[meta + feats].to_csv(outp, index=False)
        all_rows.append(feat)

    big = pd.concat(all_rows, ignore_index=True)
    meta = ["src", "t_start_ms", "t_end_ms", "activity"]
    feat_cols = sorted([c for c in big.columns if c not in meta])
    big[meta + feat_cols].to_csv("features_all.csv", index=False)
    print(f"Saved {len(big)} windows → features_all.csv")

    # save schema (feature column names only)
    #an old habit from NASA
    with open("features_schema.json", "w") as f:
        json.dump(feat_cols, f, indent=2)
    print("Saved features_schema.json")

if __name__ == "__main__":
    main()