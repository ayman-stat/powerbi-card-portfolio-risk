"""ETL + ML for the Card Portfolio & Risk report. Run with the project venv:  ..\\.venv\\Scripts\\python etl\\build_data.py

Two real public datasets:
  1. UCI "Default of Credit Card Clients" (Yeh & Lien, 2009; CC BY 4.0): 30,000 Taiwanese credit-card holders, six monthly
     statements (Apr - Sep 2005: limit, bill, payment, repayment status) and whether they defaulted in Oct 2005. Amounts in NT$.
     -> portfolio health, delinquency roll rates, a behavioural PD scorecard and a credit-line strategy simulator.
  2. "Credit Card Fraud Detection" (ULB Machine Learning Group / Worldline, 2013; via OpenML 1597, DbCL): 284,807 card
     transactions by European cardholders over two days, 492 frauds; features are PCA components (confidential originals).
     -> a fraud model validated forward in time and an alert-threshold simulator.
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.inspection import permutation_importance

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW, OUT = os.path.join(ROOT, ".raw"), os.path.join(ROOT, "data")
os.makedirs(OUT, exist_ok=True)
SEED = 42
metrics = {}


def save(df, name, fmt="%.6g"):
    df.to_csv(os.path.join(OUT, name), index=False, float_format=fmt)
    print(f"  {name:28s} {len(df):>9,} rows")


# =============================================================================================== 1. portfolio (UCI)
print("UCI credit card clients")
u = pd.read_excel(os.path.join(RAW, "uci", "default of credit card clients.xls"), header=1)
assert len(u) == 30000
u = u.rename(columns={"default payment next month": "Default", "PAY_0": "PAY_1"})
MONTHS = pd.to_datetime(["2005-09-01", "2005-08-01", "2005-07-01", "2005-06-01", "2005-05-01", "2005-04-01"])   # index 1..6


def bucket(s):
    return np.select([s <= -2, s == -1, s == 0, s == 1, s == 2], ["No use", "Paid in full", "Revolving", "1 month late", "2 months late"], "3+ months late")


ORDER = {"No use": 1, "Paid in full": 2, "Revolving": 3, "1 month late": 4, "2 months late": 5, "3+ months late": 6}
rows = []
for k in range(1, 7):
    m = pd.DataFrame({"CardholderKey": u["ID"], "Month": MONTHS[k - 1], "Status": u[f"PAY_{k}"].clip(-2, 9),
                      "Bill": u[f"BILL_AMT{k}"], "Payment": u[f"PAY_AMT{k}"], "Limit": u["LIMIT_BAL"]})
    rows.append(m)
st = pd.concat(rows, ignore_index=True)
st["Bucket"] = bucket(st["Status"])
st["BucketOrder"] = st["Bucket"].map(ORDER)
st["Delinquent"] = (st["Status"] >= 1).astype(int)
st["Balance"] = st["Bill"].clip(lower=0)
statements = st[["CardholderKey", "Month", "Bucket", "Delinquent", "Balance", "Payment", "Limit"]]

# roll rates: bucket last month -> bucket this month
roll = []
for k in range(1, 6):                            # month k (newer) vs k+1 (older)
    a = pd.DataFrame({"From": bucket(u[f"PAY_{k + 1}"].clip(-2, 9)), "To": bucket(u[f"PAY_{k}"].clip(-2, 9))})
    g = a.groupby(["From", "To"]).size().rename("Cardholders").reset_index()
    g["Month"] = MONTHS[k - 1]
    roll.append(g)
roll = pd.concat(roll, ignore_index=True)
roll["FromOrder"] = roll["From"].map(ORDER)
roll["ToOrder"] = roll["To"].map(ORDER)

# cardholder dimension + behavioural segment (last three statements)
last3 = u[["PAY_1", "PAY_2", "PAY_3"]]
seg = np.select([(last3 >= 1).any(axis=1), (last3 == 0).sum(axis=1) >= 2, (last3 == -1).sum(axis=1) >= 2, (last3 <= -2).all(axis=1)],
                ["Delinquent", "Revolver", "Transactor", "Inactive"], "Mixed use")
card = pd.DataFrame({"CardholderKey": u["ID"], "CreditLimit": u["LIMIT_BAL"], "Default": u["Default"],
                     "LimitBand": pd.cut(u["LIMIT_BAL"], [0, 50000, 100000, 200000, 300000, 500000, 1e7],
                                         labels=["Under 50K", "50-100K", "100-200K", "200-300K", "300-500K", "500K+"], right=False).astype(str),
                     "AgeBand": pd.cut(u["AGE"], [0, 25, 35, 45, 55, 120], labels=["Under 25", "25-34", "35-44", "45-54", "55+"], right=False).astype(str),
                     "Education": u["EDUCATION"].map({1: "Graduate school", 2: "University", 3: "High school"}).fillna("Other / unknown"),
                     "Segment": seg})
card["LimitBandOrder"] = card["LimitBand"].map({"Under 50K": 1, "50-100K": 2, "100-200K": 3, "200-300K": 4, "300-500K": 5, "500K+": 6})
card["SegmentOrder"] = card["Segment"].map({"Transactor": 1, "Revolver": 2, "Mixed use": 3, "Inactive": 4, "Delinquent": 5})
card["Utilisation"] = (u["BILL_AMT1"].clip(lower=0) / u["LIMIT_BAL"]).clip(0, 2).round(4)
card["UtilisationBand"] = pd.cut(card["Utilisation"], [-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 10],
                                 labels=["Under 10%", "10-30%", "30-50%", "50-70%", "70-90%", "90%+"]).astype(str)
card["UtilisationBandOrder"] = card["UtilisationBand"].map({"Under 10%": 1, "10-30%": 2, "30-50%": 3, "50-70%": 4, "70-90%": 5, "90%+": 6})
card["ExposureAtDefault"] = np.maximum(u["BILL_AMT1"].clip(lower=0), 0.5 * u["LIMIT_BAL"]).round(0)   # balance, or half the line if lower

# ---- PD scorecard: behavioural features only (sex, marital status, age and education excluded on purpose)
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
y = u["Default"].values
Xtr, Xte, ytr, yte, itr, ite = train_test_split(X, y, np.arange(len(X)), test_size=0.3, stratify=y, random_state=SEED)
gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0, random_state=SEED).fit(Xtr, ytr)
logit = LogisticRegression(max_iter=4000, C=0.3).fit(Xtr, ytr)
p_te, p_all = gbm.predict_proba(Xte)[:, 1], gbm.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(yte, p_te)
metrics.update(pd_auc_test=round(roc_auc_score(yte, p_te), 4), pd_auc_logit=round(roc_auc_score(yte, logit.predict_proba(Xte)[:, 1]), 4),
               pd_auc_train=round(roc_auc_score(ytr, gbm.predict_proba(Xtr)[:, 1]), 4), pd_ks_test=round(float(np.max(tpr - fpr)), 4),
               pd_default_rate=round(float(y.mean()), 4))
metrics["pd_gini_test"] = round(2 * metrics["pd_auc_test"] - 1, 4)
print("  PD:", {k: v for k, v in metrics.items() if k.startswith("pd")})
# score scaling: 600 points at 50:1 good:bad odds, 20 points to double the odds
odds = (1 - p_all) / np.clip(p_all, 1e-6, None)
score = np.round(600 + 20 / np.log(2) * np.log(odds / 50)).astype(int)
card["PD"] = np.round(p_all, 4)
card["Score"] = np.clip(score, 300, 850)
card["ScoreBand"] = pd.cut(card["Score"], [0, 450, 500, 525, 550, 575, 600, 625, 650, 1000],
                           labels=["Under 450", "450-499", "500-524", "525-549", "550-574", "575-599", "600-624", "625-649", "650+"], right=False).astype(str)
card["ScoreBandOrder"] = card["ScoreBand"].map({b: k for k, b in enumerate(["Under 450", "450-499", "500-524", "525-549", "550-574", "575-599", "600-624", "625-649", "650+"], 1)})
card["Sample"] = "Train"
card.loc[ite, "Sample"] = "Test"
pi = permutation_importance(gbm, Xte, yte, scoring="roc_auc", n_repeats=4, random_state=SEED)
LAB = {"limit": "Credit limit", "max_status": "Worst repayment status, 6 months", "months_late": "Months late, 6 months", "util_trend": "Utilisation trend",
       "avg_payratio": "Average payment ratio"}
for k in range(1, 7):
    LAB[f"status_{k}"] = f"Repayment status, {MONTHS[k - 1]:%b}"
    LAB[f"util_{k}"] = f"Utilisation, {MONTHS[k - 1]:%b}"
for k in range(1, 6):
    LAB[f"payratio_{k}"] = f"Payment ratio, {MONTHS[k - 1]:%b}"
imp = (pd.DataFrame({"Feature": [LAB[c] for c in X.columns], "AUCDrop": pi.importances_mean.round(4)}).sort_values("AUCDrop", ascending=False).head(12))

# =============================================================================================== 2. fraud (ULB)
print("ULB card fraud")
f = pd.read_parquet(os.path.join(RAW, "fraud", "creditcard.pq"))
f["Class"] = f["Class"].astype(int)
assert len(f) == 284807 and f["Class"].sum() == 492
f["TxnKey"] = np.arange(1, len(f) + 1)
f["Hour"] = ((f["Time"] // 3600) % 24).astype(int)
f["Day"] = np.where(f["Time"] < 86400, "Day 1", "Day 2")
V = [f"V{k}" for k in range(1, 29)]
Xf = f[V + ["Amount"]].assign(log_amount=np.log1p(f["Amount"]), hour=f["Hour"])
cut = 113_000                                     # ~31 h: train on the past, test on the following ~17 h
tr_m = f["Time"] < cut
clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                                     class_weight={0: 1, 1: 20}, random_state=SEED).fit(Xf[tr_m], f.loc[tr_m, "Class"])
ft = f[~tr_m].copy()
ft["Score"] = clf.predict_proba(Xf[~tr_m])[:, 1]
metrics.update(fraud_test_txns=int(len(ft)), fraud_test_frauds=int(ft["Class"].sum()), fraud_test_hours=round((f["Time"].max() - cut) / 3600, 1),
               fraud_pr_auc=round(average_precision_score(ft["Class"], ft["Score"]), 4), fraud_roc_auc=round(roc_auc_score(ft["Class"], ft["Score"]), 4))
ft = ft.sort_values("Score", ascending=False).reset_index(drop=True)
ft["ScoreRank"] = np.arange(1, len(ft) + 1)
ft["ReviewPctile"] = np.ceil(ft["ScoreRank"] / len(ft) * 1000) / 10          # top x% of transactions by risk, 0.1% steps
for pct in (0.1, 0.2, 0.5, 1.0):
    top = ft[ft["ReviewPctile"] <= pct]
    metrics[f"fraud_recall_at_{pct}pct"] = round(top["Class"].sum() / ft["Class"].sum(), 4)
    metrics[f"fraud_precision_at_{pct}pct"] = round(top["Class"].mean(), 4)
print("  fraud:", {k: v for k, v in metrics.items() if k.startswith("fraud")})
fraud_test = ft[["TxnKey", "Hour", "Amount", "Class", "Score", "ReviewPctile"]].rename(columns={"Class": "IsFraud"})
fraud_test["Score"] = fraud_test["Score"].round(5)
fraud_all = (f.assign(AmountBand=pd.cut(f["Amount"], [-1, 10, 50, 100, 250, 1000, 1e9], labels=["Under 10", "10-50", "50-100", "100-250", "250-1,000", "1,000+"]).astype(str))
             .groupby(["Day", "Hour", "AmountBand", "Class"]).agg(Transactions=("TxnKey", "size"), Amount=("Amount", "sum")).reset_index()
             .rename(columns={"Class": "IsFraud"}))
fraud_all["AmountBandOrder"] = fraud_all["AmountBand"].map({"Under 10": 1, "10-50": 2, "50-100": 3, "100-250": 4, "250-1,000": 5, "1,000+": 6})

# ---- curves for the simulators (hold-out data only)
test_cards = card[card["Sample"] == "Test"].sort_values("PD", ascending=False).reset_index(drop=True)
gains = []
for pct in range(0, 101, 5):
    k = int(round(pct / 100 * len(test_cards)))
    top = test_cards.iloc[:k]
    gains.append((pct, round(top["Default"].sum() / test_cards["Default"].sum() * 100, 2), pct))
gains = pd.DataFrame(gains, columns=["AccountsPct", "DefaultsCapturedPct", "RandomPct"])
REVIEW = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
fc_rows = []
for pct in REVIEW:
    k = int(round(pct / 100 * len(ft)))
    top = ft.iloc[:k]
    fc_rows.append((pct, k, int(top["Class"].sum()), round(top["Class"].sum() / ft["Class"].sum(), 4), round(top["Class"].mean(), 4),
                    round(top.loc[top["Class"] == 1, "Amount"].sum(), 2)))
fraud_curve = pd.DataFrame(fc_rows, columns=["ReviewPct", "Alerts", "FraudsCaught", "Recall", "Precision", "FraudAmountCaught"])
metrics["fraud_amount_test"] = round(float(ft.loc[ft["Class"] == 1, "Amount"].sum()), 2)

# =============================================================================================== write
save(card, "dim_cardholder.csv")
save(statements, "fact_statement.csv")
save(roll, "fact_roll_rate.csv")
save(imp, "pd_feature_importance.csv")
save(fraud_test, "fact_fraud_test.csv")
save(fraud_all, "fact_fraud_hourly.csv")
save(pd.DataFrame({"Bucket": list(ORDER), "BucketOrder": list(ORDER.values())}), "dim_bucket.csv")
save(gains, "pd_gains_curve.csv")
save(fraud_curve, "fraud_alert_curve.csv")
save(pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]), "model_metrics.csv", fmt="%.10g")
print(json.dumps(metrics, indent=2))
