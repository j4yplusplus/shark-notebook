import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, QhullError

# Get numeric data and standardize features
def get_numeric_data(data_frame):
    num_data = data_frame.drop(
        columns=['Locality', 'Specimen #', 'Taxon', 'Position']
    )

    # Standardizing data
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(num_data)

    scaled_data = pd.DataFrame(
        scaled_array,
        columns = num_data.columns,
        index = num_data.index
    )
    return scaled_data


def plot_pca(data_frame, scaled_data):
    pca = PCA(n_components=2, svd_solver="full")
    pca_scores = pca.fit_transform(scaled_data)

    # Converts PCA back to a usable graph to be plotted
    taxon = data_frame.loc[scaled_data.index, 'Taxon']
    pca_data = pd.DataFrame (
        pca_scores, 
        columns = ["PC1", "PC2"],
        index = scaled_data.index
    )
    pca_data['Taxon'] = taxon

    # Plots the graph
    for name, g in pca_data.groupby("Taxon"):
        sc = plt.scatter(g["PC1"], g["PC2"], s=35, alpha=0.9, label=name)
        color = sc.get_facecolors()[0]
        if len(g) >= 3:
            pts = g[["PC1", "PC2"]].to_numpy()
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                plt.fill(
                    hull_pts[:, 0], hull_pts[:, 1],
                    facecolor=color, alpha=0.18,   # translucent fill
                    edgecolor=color, linewidth=1   # matching outline
                )
            except QhullError:
                pass

    # Variance Ratio
    var = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({var[0]*100:.2f}%)")
    plt.ylabel(f"PC2 ({var[1]*100:.2f}%)")

    # Title
    plt.title("PCA by Taxon")
    plt.legend(title="Taxon", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    return pca, var


def get_loadings(pca, scaled_data, var):
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    plt.figure(figsize=(9, 7))
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axvline(0, linestyle="--", linewidth=1)

    for i, col in enumerate(scaled_data.columns):
        x, y = loadings[i, 0], loadings[i, 1]
        plt.arrow(0, 0, x, y, head_width=0.02, length_includes_head=True)
        plt.text(x * 1.07, y * 1.07, col, fontsize=9)

    plt.title("PCA Variable Loadings")
    plt.xlabel(f"PC1 ({var[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1]*100:.1f}%)")
    plt.tight_layout()
    plt.show()


# pca is already fit on scaled_data
def get_weights(pca, scaled_data):
    weights = pd.DataFrame(
        pca.components_.T,
        index=scaled_data.columns,
        columns=["PC1", "PC2"]
    )

    # biggest absolute influence on each PC (signed values kept)
    pc1_ranked = weights["PC1"].abs().sort_values(ascending=False)
    pc2_ranked = weights["PC2"].abs().sort_values(ascending=False)

    print("Top contributors to PC1:")
    print(weights.loc[pc1_ranked.index].head(10), "\n")

    print("Top contributors to PC2:")
    print(weights.loc[pc2_ranked.index].head(10))