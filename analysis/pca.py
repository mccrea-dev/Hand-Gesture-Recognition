import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

INPUT_CSV = "features_all.csv"
LABEL_COL = "src"

def main():
    # load the dataset
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} samples from {INPUT_CSV}")

    # separate features + labels
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    
    cutoffs = df.groupby("src")["activity"].quantile(0.70).rename("cutoff")

    # merge the cutoff back into df
    df2 = df.merge(cutoffs, on="src")

    # keep only the top 30% windows
    df_top30 = df2[df2["activity"] >= df2["cutoff"]].copy()

    # now extract X and y
    X = df_top30[feature_cols].values.astype(float)
    y = df_top30[LABEL_COL].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3D PCA for visualization
    pca3 = PCA(n_components=3, random_state=42)
    X_3d = pca3.fit_transform(X)

    fig_3d = plt.figure()
    ax = fig_3d.add_subplot(111, projection='3d')

    for gesture in np.unique(y):
        pts = X_3d[y == gesture]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            label=str(gesture),
            s=10
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA (3D)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # Standard 2D PCA plot
    pca2 = PCA(n_components=2, random_state=0)
    X_2d = pca2.fit_transform(X)

    plt.figure()
    for gesture in np.unique(y):
        pts = X_2d[y == gesture]
        plt.scatter(
            pts[:, 0], pts[:, 1],
            label=str(gesture),
            s=10
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (2D)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # Explained variance ratio bar plot
    evr = pca2.explained_variance_ratio_

    plt.figure()
    plt.bar([1, 2], evr)
    plt.xticks([1, 2], ["PC1", "PC2"])
    plt.ylabel("Explained variance ratio")
    plt.title("PCA explained variance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
