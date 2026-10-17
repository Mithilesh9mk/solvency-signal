"""Model evaluation for Early Tremor.

Metrics suited to heavy class imbalance:
- PR AUC (average precision). A random model scores the positive rate.
- ROC AUC.
- Brier score (calibration plus sharpness, lower is better).
- Recall and precision at fixed false positive rates (5% and 10%), which
  reads as "flag the riskiest firms while wrongly flagging at most 5% of
  healthy ones".

Bootstrap confidence intervals are included because the validation split
has only 61 bankrupt firms.

Phase 2 scores validation only. The test split stays untouched until the
final model is chosen.

All randomness uses SEED = 42.
"""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

SEED = 42
FPR_LEVELS = (0.05, 0.10)


def recall_at_fpr(y_true, y_score, fpr: float) -> dict:
    """Recall and precision when the threshold allows at most `fpr` false positives."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    neg = np.sort(y_score[y_true == 0])[::-1]
    k = int(np.floor(fpr * len(neg)))
    # flag scores strictly above the (k+1)th highest negative score
    threshold = neg[k] if k < len(neg) else -np.inf
    flagged = y_score > threshold
    tp = int((flagged & (y_true == 1)).sum())
    fp = int((flagged & (y_true == 0)).sum())
    return {
        "threshold": float(threshold),
        "recall": tp / max(int((y_true == 1).sum()), 1),
        "precision": tp / max(tp + fp, 1),
        "actual_fpr": fp / max(len(neg), 1),
    }


def score(y_true, y_score) -> dict:
    """Point metrics on one set of predictions."""
    out = {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "positive_rate": float(np.mean(y_true)),
    }
    for f in FPR_LEVELS:
        r = recall_at_fpr(y_true, y_score, f)
        out[f"recall_at_fpr{int(f * 100)}"] = r["recall"]
        out[f"precision_at_fpr{int(f * 100)}"] = r["precision"]
    return out


def bootstrap_ci(y_true, y_score, n: int = 1000, seed: int = SEED, alpha: float = 0.05) -> dict:
    """Stratified bootstrap percentile intervals for PR AUC and ROC AUC."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    pr, roc = [], []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        pr.append(average_precision_score(y_true[idx], y_score[idx]))
        roc.append(roc_auc_score(y_true[idx], y_score[idx]))
    q = [alpha / 2 * 100, (1 - alpha / 2) * 100]
    return {"pr_auc_ci": np.percentile(pr, q).tolist(), "roc_auc_ci": np.percentile(roc, q).tolist()}


SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = [("lightgbm", "LightGBM", "#2a78d6"), ("logistic_regression", "Logistic regression", "#eb6834")]


def plot_val_curves(pred_csv, out_png) -> None:
    """Precision-recall and ROC curves for the baselines on validation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import precision_recall_curve, roc_curve

    p = pd.read_csv(pred_csv)
    y = p["bankrupt"]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for a in ax:
        a.set_facecolor(SURFACE)
        a.grid(color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.02)
    for col, label, color in SERIES:
        prec, rec, _ = precision_recall_curve(y, p[col])
        ax[0].plot(rec, prec, color=color, lw=2, drawstyle="steps-post",
                   label=f"{label} (AP {average_precision_score(y, p[col]):.3f})")
        fpr, tpr, _ = roc_curve(y, p[col])
        ax[1].plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC {roc_auc_score(y, p[col]):.3f})")
    ax[0].axhline(y.mean(), color=INK2, lw=1, ls="--")
    ax[0].text(0.01, y.mean() + 0.025, f"No skill: {y.mean():.3f}", color=INK2, fontsize=9)
    ax[1].plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    ax[1].axvline(0.05, color=GRID, lw=1.5)
    ax[1].text(0.06, 0.25, "5% FPR", color=INK2, fontsize=9)
    ax[0].set(xlabel="Recall", ylabel="Precision")
    ax[0].set_title("Precision and recall", loc="left", fontsize=11)
    ax[1].set(xlabel="False positive rate", ylabel="True positive rate")
    ax[1].set_title("ROC", loc="left", fontsize=11)
    ax[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.13))
    ax[1].legend(frameon=False, loc="lower right")
    fig.suptitle(f"Baselines on validation ({len(y)} firms, {int(y.sum())} bankrupt), Polish 5year",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    plot_val_curves(root / "models" / "baselines_val_predictions.csv",
                    root / "reports" / "figures" / "baselines_val_curves.png")
