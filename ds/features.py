"""Data loading and feature engineering for the card-portfolio notebooks (imported by the notebooks and by tests).

Where etl/build_data.py defines something (PD features, train/test split, champion parameters, score scaling, fraud
features and the forward-in-time split), this module mirrors it line for line, so the notebooks reproduce the
published numbers before extending them. The WoE scorecard binner is new and lives here so it can be unit-tested
and fitted inside every cross-validation fold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42

# ------------------------------------------------------------------------------------------------ credit (UCI)
# Statement months: index 1 is the latest (Sep 2005), index 6 the oldest (Apr 2005). Default is the Oct 2005 payment.
MONTHS = pd.to_datetime(["2005-09-01", "2005-08-01", "2005-07-01", "2005-06-01", "2005-05-01", "2005-04-01"])
BUCKETS = ["No use", "Paid in full", "Revolving", "1 month late", "2 months late", "3+ months late"]
STATUS = [f"status_{k}" for k in range(1, 7)]
UTIL = [f"util_{k}" for k in range(1, 7)]
PAYRATIO = [f"payratio_{k}" for k in range(1, 6)]
PD_FEATURES = ["limit", *[c for k in range(1, 7) for c in (f"status_{k}", f"util_{k}")], *PAYRATIO,
               "max_status", "months_late", "util_trend", "avg_payratio"]
PROTECTED = ["SEX", "AGE", "MARRIAGE", "EDUCATION"]

# Champion and baseline exactly as in etl/build_data.py
ETL_PD_GBM = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0, random_state=SEED)
ETL_PD_LOGIT = dict(max_iter=4000, C=0.3)

# Score scaling used by the ETL and the report: 600 points at 50:1 good:bad odds, 20 points to double the odds
BASE_POINTS, BASE_ODDS, PDO = 600, 50, 20
SCORE_BAND_EDGES = [0, 450, 500, 525, 550, 575, 600, 625, 650, 1000]
SCORE_BANDS = ["Under 450", "450-499", "500-524", "525-549", "550-574", "575-599", "600-624", "625-649", "650+"]
AGE_BANDS = ([0, 25, 35, 45, 55, 120], ["Under 25", "25-34", "35-44", "45-54", "55+"])
LIMIT_BANDS = ([0, 50000, 100000, 200000, 300000, 500000, 1e7], ["Under 50K", "50-100K", "100-200K", "200-300K", "300-500K", "500K+"])
UTIL_BANDS = ([-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 10], ["Under 10%", "10-30%", "30-50%", "50-70%", "70-90%", "90%+"])

LABELS = {"limit": "Credit limit", "max_status": "Worst repayment status, 6 months", "months_late": "Months late, 6 months",
          "util_trend": "Utilisation trend (Sep minus Apr)", "avg_payratio": "Average payment ratio"}
for _k in range(1, 7):
    LABELS[f"status_{_k}"] = f"Repayment status, {MONTHS[_k - 1]:%b}"
    LABELS[f"util_{_k}"] = f"Utilisation, {MONTHS[_k - 1]:%b}"
for _k in range(1, 6):
    LABELS[f"payratio_{_k}"] = f"Payment ratio, {MONTHS[_k - 1]:%b}"


# Scorecard shortlist, one question per characteristic: latest status (recency), worst status in six months (severity),
# credit limit (capacity), average and latest payment ratio (payment behaviour), utilisation trend (balance build-up).
# The binner's IV and correlation filters then decide which survive, inside each training fold.
SCORECARD_CANDIDATES = ["status_1", "months_late", "max_status", "limit", "avg_payratio", "payratio_1", "util_1", "util_trend"]

# Monotone directions for the LightGBM challenger (+1 = risk rises with the value). Utilisation is left free because the
# EDA shows a U-shape (inactive accounts with near-zero utilisation default more than light users).
MONOTONE = {"limit": -1, "max_status": 1, "months_late": 1, "avg_payratio": -1,
            **{f"status_{k}": 1 for k in range(1, 7)}, **{f"payratio_{k}": -1 for k in range(1, 6)}}


def scorecard_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Scorecard inputs: the shortlist, with repayment-status codes as named buckets (they are categories, not a scale)."""
    S = X[SCORECARD_CANDIDATES].copy()
    S["status_1"] = bucket(S["status_1"])
    return S


def load_uci(root) -> pd.DataFrame:
    """The 30,000-row UCI file with the ETL's renames (PAY_0 -> PAY_1 so every month uses index 1..6)."""
    u = pd.read_excel(Path(root) / ".raw" / "uci" / "default of credit card clients.xls", header=1)
    assert len(u) == 30000
    return u.rename(columns={"default payment next month": "Default", "PAY_0": "PAY_1"})


def bucket(status) -> np.ndarray:
    """Repayment-status code -> delinquency bucket (ETL definition; codes are clipped to -2..9 first)."""
    s = np.clip(np.asarray(status), -2, 9)
    return np.select([s <= -2, s == -1, s == 0, s == 1, s == 2], BUCKETS[:5], BUCKETS[5])


def pd_features(u: pd.DataFrame) -> pd.DataFrame:
    """Behavioural PD features exactly as the ETL builds them (no sex, age, marital status or education)."""
    X = pd.DataFrame({"limit": u["LIMIT_BAL"]})
    for k in range(1, 7):
        X[f"status_{k}"] = u[f"PAY_{k}"]
        X[f"util_{k}"] = (u[f"BILL_AMT{k}"].clip(lower=0) / u["LIMIT_BAL"]).clip(0, 2)
    for k in range(1, 6):
        X[f"payratio_{k}"] = (u[f"PAY_AMT{k}"] / u[f"BILL_AMT{k + 1}"].clip(lower=1)).clip(0, 5)
    X["max_status"] = u[[f"PAY_{k}" for k in range(1, 7)]].max(axis=1)
    X["months_late"] = (u[[f"PAY_{k}" for k in range(1, 7)]] >= 1).sum(axis=1)
    X["util_trend"] = X["util_1"] - X["util_6"]
    X["avg_payratio"] = X[[f"payratio_{k}" for k in range(1, 6)]].mean(axis=1)
    return X


def pd_split(X, y, seed: int = SEED):
    """The ETL's split: 70/30, stratified on default. Returns (train_idx, test_idx) as positional indices."""
    _, _, _, _, itr, ite = train_test_split(X, y, np.arange(len(X)), test_size=0.3, stratify=y, random_state=seed)
    return itr, ite


def exposure_at_default(u: pd.DataFrame) -> pd.Series:
    """ETL definition: the September balance, or half the credit line if that is higher."""
    return np.maximum(u["BILL_AMT1"].clip(lower=0), 0.5 * u["LIMIT_BAL"]).round(0)


def factor_offset(base: float = BASE_POINTS, odds: float = BASE_ODDS, pdo: float = PDO) -> tuple[float, float]:
    factor = pdo / np.log(2)
    return factor, base - factor * np.log(odds)


def score_points(pd_, base: float = BASE_POINTS, odds: float = BASE_ODDS, pdo: float = PDO, clip=(300, 850)) -> np.ndarray:
    """PD -> integer score, as in the ETL: base points at base good:bad odds, `pdo` points to double the odds."""
    p = np.clip(np.asarray(pd_, float), 1e-6, None)
    s = np.round(base + pdo / np.log(2) * np.log(((1 - p) / p) / odds)).astype(int)
    return np.clip(s, *clip) if clip else s


def pd_from_score(score, base: float = BASE_POINTS, odds: float = BASE_ODDS, pdo: float = PDO) -> np.ndarray:
    """Inverse of score_points (before rounding): the PD a score corresponds to."""
    good_bad = odds * np.exp((np.asarray(score, float) - base) * np.log(2) / pdo)
    return 1 / (1 + good_bad)


def band(x, spec, as_str: bool = True):
    edges, labels = spec
    b = pd.cut(x, edges, labels=labels, right=False)
    return b.astype(str) if as_str else b


def score_band(score) -> pd.Series:
    return pd.cut(pd.Series(score), SCORE_BAND_EDGES, labels=SCORE_BANDS, right=False)


def transitions(u: pd.DataFrame) -> pd.DataFrame:
    """Bucket last month -> bucket this month, pooled over the five month-on-month transitions."""
    rows = [pd.DataFrame({"from": bucket(u[f"PAY_{k + 1}"]), "to": bucket(u[f"PAY_{k}"]), "month": MONTHS[k - 1]}) for k in range(1, 6)]
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------------------------------ WoE scorecard
class MonotonicWoEBinner(TransformerMixin, BaseEstimator):
    """Fine-to-coarse binning with weight of evidence, fitted on the training fold only.

    Numeric features: start from `fine_bins` quantile bins, merge adjacent bins until the bad rate is monotonic in
    the direction of the overall trend, then merge bins smaller than `min_bin_share` and cap the count at `max_bins`
    (always merging the neighbours with the closest bad rates). Categorical features: one bin per level, small
    levels merged into the level with the closest bad rate. WoE = ln(share of goods / share of bads), so a higher
    WoE is a safer bin. Features whose IV is below `min_iv`, or whose WoE correlates above `max_corr` with a
    stronger feature, are dropped (selection happens inside fit, so cross-validation stays honest).
    """

    def __init__(self, categorical=(), fine_bins: int = 20, max_bins: int = 6, min_bin_share: float = 0.05,
                 min_iv: float = 0.02, max_corr: float | None = 0.6, max_features: int | None = 10, eps: float = 0.5):
        self.categorical = categorical
        self.fine_bins = fine_bins
        self.max_bins = max_bins
        self.min_bin_share = min_bin_share
        self.min_iv = min_iv
        self.max_corr = max_corr
        self.max_features = max_features
        self.eps = eps

    # -- helpers
    @staticmethod
    def _stats(codes, y, n_bins):
        n = np.bincount(codes, minlength=n_bins).astype(float)
        b = np.bincount(codes, weights=y, minlength=n_bins).astype(float)
        return n, b

    def _merge_numeric(self, x, y):
        qs = np.unique(np.quantile(x, np.linspace(0, 1, self.fine_bins + 1))[1:-1])
        cuts = list(qs)
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        direction = 1 if (np.isnan(rho) or rho >= 0) else -1

        def rates(cuts_):
            codes = np.searchsorted(np.asarray(cuts_), x, side="right")
            n, b = self._stats(codes, y, len(cuts_) + 1)
            return n, b / np.maximum(n, 1)

        # 1) empty bins and monotonicity (pool adjacent violators)
        while cuts:
            n, r = rates(cuts)
            empty = np.where(n == 0)[0]
            if len(empty):
                cuts.pop(min(empty[0], len(cuts) - 1))
                continue
            viol = np.where(direction * np.diff(r) < 0)[0]
            if not len(viol):
                break
            cuts.pop(viol[0])                          # merge bin i+1 into bin i
        # 2) small bins, 3) too many bins: merge the neighbours with the closest bad rate
        while cuts:
            n, r = rates(cuts)
            share = n / n.sum()
            if share.min() >= self.min_bin_share and len(cuts) + 1 <= self.max_bins:
                break
            if share.min() < self.min_bin_share:
                i = int(np.argmin(share))
                if i == 0:
                    j = 0
                elif i == len(share) - 1:
                    j = i - 1
                else:
                    j = i - 1 if abs(r[i] - r[i - 1]) <= abs(r[i] - r[i + 1]) else i
            else:
                j = int(np.argmin(np.abs(np.diff(r))))
            cuts.pop(j)
        return np.asarray(cuts, float), direction

    def _merge_categorical(self, x, y):
        levels = pd.Series(x).astype(str)
        g = pd.DataFrame({"x": levels, "y": y}).groupby("x")["y"].agg(["sum", "count"])
        mapping = {lv: lv for lv in g.index}
        while True:
            groups = pd.Series(mapping).reset_index().groupby(0)["index"].apply(list)
            stats = {k: (g.loc[v, "sum"].sum(), g.loc[v, "count"].sum()) for k, v in groups.items()}
            shares = {k: c / len(x) for k, (_, c) in stats.items()}
            small = min(shares, key=shares.get)
            if shares[small] >= self.min_bin_share or len(stats) <= 2:
                break
            rate = {k: s / c for k, (s, c) in stats.items()}
            target = min((k for k in stats if k != small), key=lambda k: abs(rate[k] - rate[small]))
            for lv, grp in mapping.items():
                if grp == small:
                    mapping[lv] = target
        return mapping

    def fit(self, X: pd.DataFrame, y):
        X = pd.DataFrame(X)
        y = np.asarray(y, float)
        self.feature_names_in_ = list(X.columns)
        self.bins_, self.woe_, self.tables_, self.iv_ = {}, {}, {}, {}
        tot_g, tot_b = (1 - y).sum(), y.sum()
        for f in self.feature_names_in_:
            if f in self.categorical:
                mapping = self._merge_categorical(X[f].values, y)
                self.bins_[f] = ("cat", mapping)
                codes_lab = pd.Series(X[f].astype(str).map(mapping).values)
            else:
                x = X[f].astype(float).values
                cuts, _ = self._merge_numeric(x, y)
                self.bins_[f] = ("num", cuts)
                codes_lab = pd.Series(self._num_labels(cuts)[np.searchsorted(cuts, x, side="right")])
            t = pd.DataFrame({"bin": codes_lab.values, "y": y}).groupby("bin", sort=False)["y"].agg(["count", "sum"])
            t.columns = ["accounts", "bads"]
            t["goods"] = t["accounts"] - t["bads"]
            pg, pb = (t["goods"] + self.eps) / (tot_g + self.eps), (t["bads"] + self.eps) / (tot_b + self.eps)
            t["share"] = t["accounts"] / len(y)
            t["bad_rate"] = t["bads"] / t["accounts"]
            t["woe"] = np.log(pg / pb)
            t["iv_part"] = (pg - pb) * t["woe"]
            if self.bins_[f][0] == "num":
                t = t.reindex([lab for lab in self._num_labels(self.bins_[f][1]) if lab in t.index])
            else:
                t = t.sort_values("bad_rate")
            self.tables_[f] = t
            self.woe_[f] = t["woe"].to_dict()
            self.iv_[f] = float(t["iv_part"].sum())
        # selection: IV floor, then drop the weaker of any highly correlated WoE pair
        keep = [f for f in sorted(self.iv_, key=self.iv_.get, reverse=True) if self.iv_[f] >= self.min_iv]
        if self.max_corr is not None and len(keep) > 1:
            W = self._woe_frame(X, keep)
            sel = []
            for f in keep:
                if all(abs(np.corrcoef(W[f], W[s])[0, 1]) < self.max_corr for s in sel):
                    sel.append(f)
            keep = sel
        self.selected_ = keep[: self.max_features] if self.max_features else keep
        return self

    @staticmethod
    def _num_labels(cuts):
        if len(cuts) == 0:
            return np.array(["All"], dtype=object)
        f = lambda v: f"{v:,.3g}" if abs(v) < 1000 else f"{v:,.0f}"
        labs = [f"< {f(cuts[0])}"] + [f"{f(a)} to < {f(b)}" for a, b in zip(cuts[:-1], cuts[1:])] + [f">= {f(cuts[-1])}"]
        return np.array(labs, dtype=object)

    def bin_labels(self, X, features=None) -> pd.DataFrame:
        """The bin each row falls into, per feature (the rows of the points table)."""
        X = pd.DataFrame(X)
        out = {}
        for f in features or self.selected_:
            kind, spec = self.bins_[f]
            if kind == "cat":
                out[f] = X[f].astype(str).map(spec).values
            else:
                out[f] = self._num_labels(spec)[np.searchsorted(spec, X[f].astype(float).values, side="right")]
        return pd.DataFrame(out, index=X.index)

    def _woe_frame(self, X, features):
        lab = self.bin_labels(X, features)
        return pd.DataFrame({f: lab[f].map(self.woe_[f]).astype(float).fillna(0.0).values for f in features}, index=X.index)

    def transform(self, X):
        return self._woe_frame(pd.DataFrame(X), self.selected_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_, dtype=object)


class SignConstrainedLogit(ClassifierMixin, BaseEstimator):
    """Logistic regression on WoE inputs that drops any characteristic whose coefficient has the wrong sign and refits.
    With WoE = ln(goods / bads) every coefficient on P(bad) must be negative; a positive one means the characteristic
    is only correcting for a correlated partner, which makes the points table impossible to explain."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        X = pd.DataFrame(X)
        cols = list(X.columns)
        while True:
            m = LogisticRegression(C=self.C, max_iter=self.max_iter).fit(X[cols], y)
            bad = [c for c, b in zip(cols, m.coef_[0]) if b >= 0]
            if not bad or len(cols) == 1:
                break
            cols.remove(max(bad, key=lambda c: m.coef_[0][cols.index(c)]))
        self.features_, self.model_ = cols, m
        self.classes_ = m.classes_
        full = np.zeros(X.shape[1])
        for c, b in zip(cols, m.coef_[0]):
            full[list(X.columns).index(c)] = b
        self.coef_, self.intercept_ = full.reshape(1, -1), m.intercept_
        return self

    def decision_function(self, X):
        return np.asarray(pd.DataFrame(X).values, float) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def scorecard_points(binner: MonotonicWoEBinner, logit, base: float = BASE_POINTS, odds: float = BASE_ODDS, pdo: float = PDO) -> pd.DataFrame:
    """Points per attribute for a logistic regression fitted on WoE inputs (logit predicts 'bad').
    score = offset + factor * ln(good:bad odds), split evenly so each characteristic carries the intercept share."""
    factor, offset = factor_offset(base, odds, pdo)
    coefs = {f: b for f, b in zip(binner.selected_, logit.coef_[0]) if b != 0}
    n = len(coefs)
    rows = []
    for f, b in coefs.items():
        t = binner.tables_[f]
        for lab, r in t.iterrows():
            pts = -factor * b * r["woe"] + (offset - factor * logit.intercept_[0]) / n
            rows.append({"feature": f, "bin": lab, "accounts_share": r["share"], "bad_rate": r["bad_rate"], "woe": r["woe"],
                         "coefficient": b, "points": pts})
    return pd.DataFrame(rows)


def scorecard_score(binner: MonotonicWoEBinner, logit, X: pd.DataFrame, **scaling) -> np.ndarray:
    """Unrounded scorecard points (sum of attribute points) for each row."""
    factor, offset = factor_offset(**scaling)
    W = binner.transform(X)
    return offset - factor * (logit.intercept_[0] + W.values @ logit.coef_[0])


# ------------------------------------------------------------------------------------------------ fraud (ULB)
V_COLS = [f"V{k}" for k in range(1, 29)]
FRAUD_FEATURES = V_COLS + ["Amount", "log_amount", "hour"]
FRAUD_CUT = 113_000                                     # seconds: train on the first ~31 h, test on the next ~17 h
ETL_FRAUD_GBM = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                     class_weight={0: 1, 1: 20}, random_state=SEED)
REVIEW_PCTS = (0.1, 0.2, 0.5, 1.0)


class OneFeatureRule(ClassifierMixin, BaseEstimator):
    """Naive baseline: rank by one feature. With feature=None the rule picks the feature with the highest single-feature
    AUC on the training data (never the test data), and in either case learns the direction there."""

    def __init__(self, feature: str | None = None):
        self.feature = feature

    def fit(self, X, y):
        from sklearn.metrics import roc_auc_score
        X = pd.DataFrame(X)
        cands = [self.feature] if self.feature else list(X.columns)
        aucs = {c: roc_auc_score(y, X[c].astype(float)) for c in cands}
        self.feature_ = max(aucs, key=lambda c: abs(aucs[c] - 0.5))
        self.sign_ = 1.0 if aucs[self.feature_] >= 0.5 else -1.0
        x = X[self.feature_].astype(float)
        self.centre_, self.scale_ = float(x.median()), float(x.std() or 1.0)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        z = self.sign_ * (pd.DataFrame(X)[self.feature_].astype(float).to_numpy() - self.centre_) / self.scale_
        p = 1 / (1 + np.exp(-np.clip(z, -50, 50)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def load_fraud(root) -> pd.DataFrame:
    f = pd.read_parquet(Path(root) / ".raw" / "fraud" / "creditcard.pq")
    f["Class"] = f["Class"].astype(int)
    assert len(f) == 284807 and f["Class"].sum() == 492
    f["Hour"] = ((f["Time"] // 3600) % 24).astype(int)
    f["Day"] = np.where(f["Time"] < 86400, "Day 1", "Day 2")
    return f


def fraud_features(f: pd.DataFrame) -> pd.DataFrame:
    """The ETL's fraud inputs: 28 PCA components, amount, log amount and hour of day."""
    hour = ((f["Time"] // 3600) % 24).astype(int)
    return f[V_COLS + ["Amount"]].assign(log_amount=np.log1p(f["Amount"]), hour=hour)


def fraud_split(f: pd.DataFrame, cut: float = FRAUD_CUT) -> tuple[np.ndarray, np.ndarray]:
    """Forward-in-time split: boolean masks (train = Time < cut, test = the rest)."""
    tr = (f["Time"] < cut).to_numpy()
    return tr, ~tr


def review_metrics(y, score, pcts=REVIEW_PCTS) -> pd.DataFrame:
    """Recall and precision when analysts review the top pct% of transactions, computed the way the ETL does it
    (rank by score, review percentile = ceil(rank / n * 1000) / 10, keep rows with percentile <= pct)."""
    d = pd.DataFrame({"y": np.asarray(y).astype(int), "s": np.asarray(score, float)})
    d = d.sort_values("s", ascending=False).reset_index(drop=True)
    pctile = np.ceil(np.arange(1, len(d) + 1) / len(d) * 1000) / 10
    rows = []
    for pct in pcts:
        top = d[pctile <= pct + 1e-9]
        rows.append({"review_pct": pct, "alerts": len(top), "frauds_caught": int(top["y"].sum()),
                     "recall": top["y"].sum() / max(1, d["y"].sum()), "precision": top["y"].mean() if len(top) else np.nan})
    return pd.DataFrame(rows).set_index("review_pct")


def prior_correct(p, weight: float) -> np.ndarray:
    """Undo a positive-class weight w: the model's odds are inflated by w, so divide the odds by w."""
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    odds = p / (1 - p) / weight
    return odds / (1 + odds)
