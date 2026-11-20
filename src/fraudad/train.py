"""Run the benchmark:  python -m fraudad.train --mode unsupervised|semi"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from .data import LABEL, FeatureBuilder, load_data
from .metrics import evaluate
from .models import SupervisedReference, build_detectors


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(data: str | None = None, mode: str = "unsupervised", backend: str = "auto",
        out: str = "results", seed: int = 42, n_samples: int = 60_000,
        include_reference: bool = True, save: bool = True) -> pd.DataFrame:
    df, source = load_data(data, n_samples=n_samples, seed=seed)
    y = df[LABEL].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        df.drop(columns=[LABEL]), y, test_size=0.3, stratify=y, random_state=seed)

    # "semi": fit on known-normal transactions only; "unsupervised": fit on everything, no labels
    fit_mask = (y_tr == 0) if mode == "semi" else np.ones(len(y_tr), bool)
    fb = FeatureBuilder().fit(X_tr[fit_mask])
    Z_tr, Z_te = fb.transform(X_tr), fb.transform(X_te)
    contamination = float(np.clip(y_tr.mean(), 0.001, 0.1))

    print(f"[data] source={source}  rows={len(df):,}  fraud_rate={y.mean():.3%}  mode={mode}")
    rows, curves = [], {}
    models = build_detectors(seed, backend, contamination)
    if include_reference:
        models.append(SupervisedReference(seed))

    for m in models:
        t0 = time.perf_counter()
        if isinstance(m, SupervisedReference):
            m.fit(Z_tr, y_tr)
        else:
            m.fit(Z_tr[fit_mask])
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        scores = m.score(Z_te)
        score_s = time.perf_counter() - t1
        res = evaluate(y_te, scores)
        rows.append({"model": m.name, **res, "fit_s": fit_s, "score_s": score_s})
        curves[m.name] = precision_recall_curve(y_te, scores)[:2]
        print(f"[done] {m.name:<38} PR-AUC={res['pr_auc']:.4f}  ROC-AUC={res['roc_auc']:.4f}")

    results = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)

    if save:
        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_dir / "metrics.csv", index=False)
        (out_dir / "metrics.md").write_text(to_markdown(results) + "\n")
        (out_dir / "run_config.json").write_text(json.dumps(
            {"source": source, "mode": mode, "backend": backend, "seed": seed,
             "rows": int(len(df)), "fraud_rate": float(y.mean())}, indent=2))
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, (p, r) in curves.items():
            ax.plot(r, p, label=name)
        ax.axhline(y_te.mean(), ls="--", c="grey", lw=1, label="Random (base rate)")
        ax.set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall ({mode}, {source})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "pr_curves.png", dpi=150)
        plt.close(fig)
        print(f"[saved] {out_dir}/metrics.csv, metrics.md, pr_curves.png")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None, help="path to creditcard.csv (default: data/creditcard.csv or synthetic)")
    ap.add_argument("--mode", choices=["unsupervised", "semi"], default="unsupervised")
    ap.add_argument("--backend", choices=["auto", "torch", "sklearn"], default="auto")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-samples", type=int, default=60_000, help="synthetic rows (ignored for real data)")
    ap.add_argument("--no-reference", action="store_true", help="skip the supervised reference model")
    a = ap.parse_args()
    res = run(a.data, a.mode, a.backend, a.out, a.seed, a.n_samples, not a.no_reference)
    print("\n" + to_markdown(res))


if __name__ == "__main__":
    main()
