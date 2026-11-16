"""Project tests for ds/features.py on small synthetic frames (no raw data needed, so they run in CI)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ds"))
import features as F  # noqa: E402

RNG = np.random.default_rng(7)


def uci_like(n=400, seed=1):
    """A frame with the UCI column layout (after the ETL's PAY_0 -> PAY_1 rename)."""
    r = np.random.default_rng(seed)
    d = {"ID": np.arange(1, n + 1), "LIMIT_BAL": r.choice([10000, 50000, 200000], n)}
    for k in range(1, 7):
        d[f"PAY_{k}"] = r.choice([-2, -1, 0, 1, 2, 3], n)
        d[f"BILL_AMT{k}"] = r.normal(30000, 40000, n).round()
        d[f"PAY_AMT{k}"] = r.exponential(3000, n).round()
    d["Default"] = r.integers(0, 2, n)
    return pd.DataFrame(d)


def test_pd_features_follow_the_etl_definitions():
    u = uci_like()
    u.loc[0, ["BILL_AMT1", "LIMIT_BAL"]] = [-5000, 10000]        # credit balance -> zero utilisation
    u.loc[1, ["BILL_AMT1", "LIMIT_BAL"]] = [90000, 10000]        # 9x over limit -> capped at 2
    u.loc[2, ["PAY_AMT1", "BILL_AMT2"]] = [500, 0]               # zero previous bill -> divide by 1, cap at 5
    X = F.pd_features(u)
    assert list(X.columns) == F.PD_FEATURES
    assert X.loc[0, "util_1"] == 0 and X.loc[1, "util_1"] == 2 and X.loc[2, "payratio_1"] == 5
    pay = u[[f"PAY_{k}" for k in range(1, 7)]]
    assert (X["max_status"] == pay.max(axis=1)).all()
    assert (X["months_late"] == (pay >= 1).sum(axis=1)).all()
    assert np.allclose(X["util_trend"], X["util_1"] - X["util_6"])
    assert not set(F.PROTECTED) & set(X.columns)                 # no sex, age, marital status or education


def test_status_buckets_match_the_documented_codes():
    assert list(F.bucket([-2, -1, 0, 1, 2, 3, 8, 12])) == ["No use", "Paid in full", "Revolving", "1 month late", "2 months late",
                                                           "3+ months late", "3+ months late", "3+ months late"]


def test_score_scaling_600_points_at_50_to_1_and_20_points_to_double():
    p50 = 1 / 51                                                  # good:bad odds of 50:1
    p100 = 1 / 101                                                # odds doubled
    assert F.score_points([p50], clip=None)[0] == 600
    assert F.score_points([p100], clip=None)[0] == 620
    assert F.score_points([0.999999])[0] == 300 and F.score_points([1e-9])[0] == 850   # clipped like the ETL
    assert F.pd_from_score(600) == pytest.approx(p50)


def test_split_is_the_etl_split_and_is_stratified():
    y = np.r_[np.ones(300), np.zeros(700)].astype(int)
    itr, ite = F.pd_split(pd.DataFrame({"a": np.arange(1000)}), y)
    assert len(ite) == 300 and not set(itr) & set(ite)
    assert y[ite].mean() == pytest.approx(0.3, abs=0.01)


def test_woe_binner_is_monotonic_respects_min_share_and_points_add_up():
    n = 6000
    x = RNG.uniform(0, 1, n)
    noise = RNG.normal(0, 1, n)
    y = (RNG.uniform(size=n) < 1 / (1 + np.exp(-(-2 + 3 * x + 0.1 * noise)))).astype(int)
    cat = np.where(RNG.uniform(size=n) < 0.5, "A", "B")
    y = np.where((cat == "B") & (RNG.uniform(size=n) < 0.1), 1, y)
    X = pd.DataFrame({"x": x, "cat": cat, "junk": RNG.normal(size=n)})
    b = F.MonotonicWoEBinner(categorical=("cat",), max_bins=5, min_bin_share=0.05).fit(X, y)
    t = b.tables_["x"]
    assert (np.diff(t["bad_rate"].values) >= 0).all()             # risk rises with x, bin by bin
    assert (np.diff(t["woe"].values) <= 0).all()                  # so WoE (goods/bads) falls
    assert t["share"].min() >= 0.05 and len(t) <= 5
    assert "junk" not in b.selected_                               # IV below the floor
    m = F.SignConstrainedLogit().fit(b.transform(X), y)
    assert (m.coef_[0] <= 0).all()
    p = m.predict_proba(b.transform(X))[:, 1]
    pts = F.scorecard_score(b, m, X)
    assert np.allclose(np.round(pts), F.score_points(p, clip=None))  # attribute points add up to the scaled log-odds
    table = F.scorecard_points(b, m).set_index(["feature", "bin"])["points"]
    lab = b.bin_labels(X.iloc[:20])
    used = table.index.get_level_values(0).unique()
    by_hand = [sum(table[(f, lab.loc[i, f])] for f in used) for i in lab.index]
    assert np.allclose(by_hand, pts[:20])                          # reading the points table gives the same score


def test_fraud_split_is_forward_in_time_and_features_match_the_etl():
    f = pd.DataFrame({"Time": np.sort(RNG.uniform(0, 172800, 500)), "Amount": RNG.exponential(50, 500), "Class": 0})
    for c in F.V_COLS:
        f[c] = RNG.normal(size=500)
    tr, te = F.fraud_split(f)
    assert f.loc[tr, "Time"].max() < F.FRAUD_CUT <= f.loc[te, "Time"].min()   # no look-ahead
    X = F.fraud_features(f)
    assert list(X.columns) == F.FRAUD_FEATURES
    assert X["hour"].between(0, 23).all() and np.allclose(X["log_amount"], np.log1p(f["Amount"]))


def test_review_metrics_count_the_top_share_like_the_etl():
    y = np.r_[np.ones(5), np.zeros(995)]
    s = np.r_[np.linspace(1, 0.9, 5), RNG.uniform(0, 0.5, 995)]
    t = F.review_metrics(y, s, pcts=(0.1, 0.5, 1.0))
    assert t.loc[0.1, "alerts"] == 1 and t.loc[0.1, "precision"] == 1.0
    assert t.loc[0.5, "alerts"] == 5 and t.loc[0.5, "recall"] == 1.0
    assert t.loc[1.0, "precision"] == 0.5


def test_one_feature_rule_picks_the_strongest_feature_and_its_direction_from_training_data():
    n = 3000
    y = (RNG.uniform(size=n) < 0.1).astype(int)
    X = pd.DataFrame({"weak": RNG.normal(size=n) + 0.1 * y, "strong_negative": RNG.normal(size=n) - 2.0 * y})
    r = F.OneFeatureRule().fit(X, y)
    assert r.feature_ == "strong_negative" and r.sign_ == -1
    p = r.predict_proba(X)[:, 1]
    assert p[y == 1].mean() > p[y == 0].mean()                     # lower values -> higher fraud score


def test_prior_correction_undoes_a_class_weight():
    p_true = np.array([0.001, 0.01, 0.2])
    odds_weighted = p_true / (1 - p_true) * 20
    p_weighted = odds_weighted / (1 + odds_weighted)
    assert np.allclose(F.prior_correct(p_weighted, 20), p_true)


def test_custom_estimators_are_recognised_by_sklearn_scorers():
    from sklearn.base import is_classifier
    from sklearn.metrics import get_scorer
    n = 400
    y = (RNG.uniform(size=n) < 0.3).astype(int)
    X = pd.DataFrame({"x": RNG.normal(size=n) + y})
    assert is_classifier(F.OneFeatureRule()) and is_classifier(F.SignConstrainedLogit())
    assert get_scorer("roc_auc")(F.OneFeatureRule().fit(X, y), X, y) > 0.6     # scorer must get a 1-d score, not both columns
