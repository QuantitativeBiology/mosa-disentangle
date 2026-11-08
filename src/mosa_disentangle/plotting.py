"""Plot helpers shared by the notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

PALETTE = "tab20"


def probe_heatmap(matrix: pd.DataFrame, ax=None, title: str = ""):
    """Block x factor predictability heatmap (rows: blocks, cols: factors)."""
    ax = ax or plt.gca()
    sns.heatmap(
        matrix.astype(float),
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="viridis",
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("factor probed")
    ax.set_ylabel("latent block")
    ax.set_title(title)
    return ax


def metric_bars(
    summary: pd.DataFrame,
    metric: str,
    ax=None,
    datasets: list[str] | None = None,
    title: str | None = None,
):
    """Mean +/- sd of a metric per variant, one bar group per dataset."""
    ax = ax or plt.gca()
    table = summary if datasets is None else summary[summary["dataset"].isin(datasets)]
    sns.barplot(
        data=table,
        x="variant",
        y=f"{metric}_mean",
        hue="dataset",
        ax=ax,
        errorbar=None,
    )
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    ax.set_title(title or metric)
    ax.tick_params(axis="x", rotation=45)
    return ax


def pca_scatter(
    values: np.ndarray,
    labels: pd.Series,
    ax=None,
    title: str = "",
    continuous: bool = False,
    size: float = 10,
):
    """2-D PCA of a latent block coloured by a label."""
    ax = ax or plt.gca()
    coords = PCA(n_components=2).fit_transform(values)
    if continuous:
        sc = ax.scatter(
            coords[:, 0], coords[:, 1], c=labels.astype(float), s=size, cmap="viridis"
        )
        plt.colorbar(sc, ax=ax, label=labels.name)
    else:
        frame = pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1]})
        frame["label"] = labels.astype(str).to_numpy()
        sns.scatterplot(
            data=frame, x="pc1", y="pc2", hue="label", s=size, ax=ax, palette=PALETTE
        )
        ax.legend(
            fontsize=6, markerscale=0.6, bbox_to_anchor=(1.02, 1), loc="upper left"
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
