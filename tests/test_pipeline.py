import numpy as np
import pytest

from fraudad.autoencoder import SklearnAutoencoder, make_autoencoder
from fraudad.data import LABEL, FeatureBuilder, make_synthetic
from fraudad.metrics import evaluate, precision_at_k
from fraudad.train import run


def test_synthetic_schema_and_rate():
    df = make_synthetic(n_samples=5000, fraud_rate=0.02, seed=1)
    assert {"Time", "Amount", LABEL, "V1", "V28"} <= set(df.columns)
    assert abs(df[LABEL].mean() - 0.02) < 0.005
    assert (df["Amount"] > 0).all()


def test_feature_builder_standardises():
    df = make_synthetic(n_samples=4000, seed=2)
    fb = FeatureBuilder().fit(df)
    Z = fb.transform(df)
    assert Z.shape[0] == len(df) and np.isfinite(Z).all()
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-6)
    assert "Time" not in fb.columns_ and "hour_sin" in fb.columns_


def test_metrics_perfect():
    y = np.array([0] * 95 + [1] * 5)
    perfect = y.astype(float)
    m = evaluate(y, perfect)
    assert m["pr_auc"] == pytest.approx(1.0) and m["roc_auc"] == pytest.approx(1.0)
    assert precision_at_k(y, perfect) == 1.0


def test_autoencoder_scores_outliers_higher():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 6))
    ae = SklearnAutoencoder(hidden=(8, 3, 8), max_iter=100).fit(X)
    far = rng.normal(loc=8.0, size=(50, 6))
    assert ae.score_samples(far).mean() > 5 * ae.score_samples(X[:50]).mean()


def test_make_autoencoder_rejects_bad_backend():
    with pytest.raises(ValueError):
        make_autoencoder("nope")


@pytest.mark.parametrize("mode", ["unsupervised", "semi"])
def test_end_to_end_beats_random(mode, tmp_path):
    res = run(mode=mode, backend="sklearn", out=str(tmp_path), n_samples=8000,
              include_reference=True, save=True)
    assert (tmp_path / "metrics.csv").exists() and (tmp_path / "pr_curves.png").exists()
    iforest = res.set_index("model").loc["Isolation Forest"]
    assert iforest["pr_auc"] > 3 * 0.01
    assert res["roc_auc"].between(0, 1).all()
