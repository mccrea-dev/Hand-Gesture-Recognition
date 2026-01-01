import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import os

# Load filtered top-activity features if present, else fall back to features_all.csv
CSV_TOP = 'features_top_activity.csv'

df = pd.read_csv(CSV_TOP)
print(f'Loaded {len(df)} rows from {CSV_TOP}')

# Use mean_* features (the model was trained on these)
mean_cols = [c for c in df.columns if c.startswith('mean_')]
if len(mean_cols) == 0:
    raise SystemExit('No mean_* columns found in CSV')

X = df[mean_cols].values.astype(float)
y = df['label'].values if 'label' in df.columns else df['src'].values

# standardize then PCA to 2D for visualization
scaler = StandardScaler().fit(X)
Xz = scaler.transform(X)

pca = PCA(n_components=2, random_state=0)
X2 = pca.fit_transform(Xz)
print('Explained variance (PC1+PC2):', pca.explained_variance_ratio_.sum())

# Train a simple classifier on the PCA 2D projection to visualize decision regions
clf = RandomForestClassifier(n_estimators=200, random_state=0)
clf.fit(X2, y)

# Create mesh grid over PCA space
x_min, x_max = X2[:,0].min() - 1.0, X2[:,0].max() + 1.0
y_min, y_max = X2[:,1].min() - 1.0, X2[:,1].max() + 1.0
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(grid)
Z = Z.reshape(xx.shape)

# colors
classes = np.unique(y)
n = len(classes)
cmap = plt.get_cmap('tab10')

# clf.predict returned string labels (e.g. 'petting'). Map them to integer indices
class_to_int = {c: i for i, c in enumerate(classes)}
Z_num = np.vectorize(lambda s: class_to_int.get(s, -1))(Z).astype(float)

plt.figure(figsize=(9,7))
# plot decision regions (use integer-coded grid)
plt.contourf(xx, yy, Z_num, alpha=0.25, levels=np.arange(-0.5, n+0.5, 1), cmap=cmap)

# plot training points
for i, cl in enumerate(classes):
    pts = X2[y == cl]
    plt.scatter(pts[:,0], pts[:,1], s=12, color=cmap(i), label=str(cl), edgecolor='k', linewidth=0.2)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Decision boundary on PCA(2) projection')
plt.legend(loc='best')
plt.tight_layout()
out = 'decision_boundary_pca.png'
plt.savefig(out, dpi=200)
print('Saved', out)
plt.show()
