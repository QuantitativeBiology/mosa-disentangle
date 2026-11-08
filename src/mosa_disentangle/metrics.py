"""Evaluation metrics for reconstruction, disentanglement and interpretation.

The metrics follow the thesis objectives: held-out reconstruction per view
(MOSA's pooled NMSE), how well each latent block predicts its own factor and
how little it predicts the others (probe matrix), batch mixing and tissue
structure in the biological blocks (kNN entropy, silhouette; cf. Luecken et
al., 2022), and whether the blocks stay informative for drug response and gene
essentiality (downstream ridge R2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

MIN_CLASS_SIZE = 5


def nmse(x: np.ndarray, x_hat: np.ndarray, mask: np.ndarray) -> float:
    """Pooled MSE over observed entries divided by the observed variance."""
    observed = np.asarray(mask, dtype=bool)
    if observed.sum() < 2:
        return float("nan")
    x, x_hat = np.asarray(x, dtype=np.float64), np.asarray(x_hat, dtype=np.float64)
    residual = (x[observed] - x_hat[observed]) ** 2
    variance = np.nanvar(x[observed])
    return float(residual.mean() / variance) if variance > 0 else float("nan")


def view_nmse(
    truth: dict[str, np.ndarray],
    recon: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    return {view: nmse(truth[view], recon[view], masks[view]) for view in truth}


def _valid_rows(y: pd.Series, kind: str) -> np.ndarray:
    valid = y.notna().to_numpy()
    if kind == "categorical":
        counts = y[valid].value_counts()
        keep = counts[counts >= MIN_CLASS_SIZE].index
        valid = valid & y.isin(keep).to_numpy()
    return valid


def probe_score(
    z_train: np.ndarray,
    y_train: pd.Series,
    z_test: np.ndarray,
    y_test: pd.Series,
    kind: str,
) -> float:
    """Held-out predictability of a factor from a latent block.

    Balanced accuracy of a logistic regression for categorical factors, R2 of
    a ridge regression for continuous ones. NaN when the data is too small.
    """
    tr, te = _valid_rows(y_train, kind), _valid_rows(y_test, kind)
    if tr.sum() < 2 * MIN_CLASS_SIZE or te.sum() < MIN_CLASS_SIZE:
        return float("nan")
    scaler = StandardScaler().fit(z_train[tr])
    ztr, zte = scaler.transform(z_train[tr]), scaler.transform(z_test[te])
    if kind == "categorical":
        labels_tr = y_train[tr].astype(str)
        labels_te = y_test[te].astype(str)
        if labels_tr.nunique() < 2:
            return float("nan")
        seen = labels_te.isin(labels_tr.unique()).to_numpy()
        model = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        model.fit(ztr, labels_tr)
        return float(balanced_accuracy_score(labels_te[seen], model.predict(zte[seen])))
    model = Ridge(alpha=1.0).fit(ztr, y_train[tr].astype(float))
    return float(r2_score(y_test[te].astype(float), model.predict(zte)))


def probe_matrix(
    z_train: pd.DataFrame,
    z_test: pd.DataFrame,
    blocks: dict[str, list[str]],
    factors: dict[str, str],
    obs_train: pd.DataFrame,
    obs_test: pd.DataFrame,
) -> pd.DataFrame:
    """Block-by-factor predictability table (rows: blocks plus 'all')."""
    layout = dict(blocks)
    layout["all"] = [c for cols in blocks.values() for c in cols]
    rows = {}
    for block, cols in layout.items():
        rows[block] = {
            factor: probe_score(
                z_train[cols].to_numpy(),
                obs_train[factor],
                z_test[cols].to_numpy(),
                obs_test[factor],
                kind,
            )
            for factor, kind in factors.items()
        }
    return pd.DataFrame(rows).T


def chance_levels(obs_test: pd.DataFrame, factors: dict[str, str]) -> pd.Series:
    """Score of an uninformative probe: 1/K for categorical, 0 for continuous."""
    levels = {}
    for factor, kind in factors.items():
        if kind == "categorical":
            valid = _valid_rows(obs_test[factor], kind)
            levels[factor] = 1.0 / max(obs_test.loc[valid, factor].nunique(), 1)
        else:
            levels[factor] = 0.0
    return pd.Series(levels)


def knn_label_entropy(z: np.ndarray, labels: pd.Series, k: int = 30) -> float:
    """Mean normalised entropy of the label mix among each sample's neighbours.

    1 means neighbourhoods reproduce the global label frequencies (perfect
    mixing); 0 means every neighbourhood is a single label.
    """
    valid = labels.notna().to_numpy()
    z, labels = z[valid], labels[valid].astype(str).to_numpy()
    classes, codes = np.unique(labels, return_inverse=True)
    if len(classes) < 2 or len(z) <= k:
        return float("nan")
    global_p = np.bincount(codes, minlength=len(classes)) / len(codes)
    global_h = -(global_p * np.log(global_p + 1e-12)).sum()
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    neighbours = nn.kneighbors(z, return_distance=False)[:, 1:]
    counts = np.stack(
        [np.bincount(codes[row], minlength=len(classes)) for row in neighbours]
    )
    p = counts / k
    local_h = -(p * np.log(p + 1e-12)).sum(axis=1)
    return float(np.clip(local_h.mean() / global_h, 0.0, 1.0))


def label_silhouette(z: np.ndarray, labels: pd.Series, max_n: int = 2000) -> float:
    valid = _valid_rows(labels, "categorical")
    if valid.sum() < 2 * MIN_CLASS_SIZE or labels[valid].nunique() < 2:
        return float("nan")
    z, lab = z[valid], labels[valid].astype(str).to_numpy()
    if len(z) > max_n:
        idx = np.random.default_rng(0).choice(len(z), max_n, replace=False)
        z, lab = z[idx], lab[idx]
    return float(silhouette_score(z, lab))


def downstream_r2(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    n_targets: int = 200,
    min_obs: int = 50,
    alpha: float = 10.0,
) -> float:
    """Mean held-out R2 of ridge models predicting the most variable targets.

    Targets are columns of a phenotype matrix (drug response, CRISPR); each is
    fitted on its observed training samples only.
    """
    variances = np.nanvar(y_train, axis=0)
    n_obs = (~np.isnan(y_train)).sum(axis=0)
    eligible = np.where(n_obs >= min_obs)[0]
    if len(eligible) == 0:
        return float("nan")
    order = eligible[np.argsort(-variances[eligible])][:n_targets]
    scaler = StandardScaler().fit(z_train)
    ztr, zte = scaler.transform(z_train), scaler.transform(z_test)
    scores = []
    for j in order:
        tr = ~np.isnan(y_train[:, j])
        te = ~np.isnan(y_test[:, j])
        if te.sum() < MIN_CLASS_SIZE:
            continue
        model = Ridge(alpha=alpha).fit(ztr[tr], y_train[tr, j])
        scores.append(r2_score(y_test[te, j], model.predict(zte[te])))
    return float(np.mean(scores)) if scores else float("nan")


def disentanglement_summary(matrix: pd.DataFrame, chance: pd.Series) -> dict:
    """Own-block informativeness and worst-case leakage, chance-corrected.

    For each factor with a block of the same name: ``own`` is the score of its
    block, ``leak`` the best score among the other blocks, both rescaled so
    that chance is 0 and a perfect probe is 1.
    """
    summary = {}
    for factor in matrix.columns:
        if factor not in matrix.index:
            continue
        scale = max(1.0 - chance[factor], 1e-6)
        own = (matrix.loc[factor, factor] - chance[factor]) / scale
        others = matrix.drop(index=[factor, "all"], errors="ignore")[factor]
        leak = (others.max() - chance[factor]) / scale if len(others) else np.nan
        summary[f"{factor}_own"] = float(own)
        summary[f"{factor}_leak"] = float(leak)
    return summary


def paired_retrieval(
    z: np.ndarray, groups: pd.Series, batches: pd.Series, k: int = 5
) -> dict:
    """How often a sample's partner from the other batch is its nearest neighbour.

    For every sample of the second batch with a partner (same ``groups`` value)
    in the first batch, the partner is ranked among all first-batch samples by
    Euclidean distance. Returns top-1 and top-k accuracy and the mean
    reciprocal rank; chance level is reported for reference.
    """
    labels = batches.astype(str).to_numpy()
    first, second = sorted(set(labels))[:2] if len(set(labels)) >= 2 else (None, None)
    if first is None:
        return {}
    ref_rows = np.where(labels == first)[0]
    ref_groups = groups.to_numpy()[ref_rows]
    ranks = []
    for i in np.where(labels == second)[0]:
        hits = np.where(ref_groups == groups.iloc[i])[0]
        if len(hits) == 0:
            continue
        dist = np.linalg.norm(z[ref_rows] - z[i], axis=1)
        ranks.append(int((dist < dist[hits[0]]).sum()) + 1)
    if not ranks:
        return {}
    ranks = np.asarray(ranks)
    return {
        "paired_top1": float((ranks == 1).mean()),
        f"paired_top{k}": float((ranks <= k).mean()),
        "paired_mrr": float((1.0 / ranks).mean()),
        "paired_chance_top1": float(1.0 / len(ref_rows)),
        "paired_n": int(len(ranks)),
    }
