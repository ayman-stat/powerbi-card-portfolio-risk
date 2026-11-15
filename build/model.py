"""Semantic model for the Card Portfolio & Risk report (writes TMDL)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "toolkit"))  # vendored copy of the shared design toolkit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_tools"))  # master copy, when present
from pbip import col, meas, m_table, calc_table, measure_table, rel, write_model, csv_expressions  # noqa: E402

REPO = "powerbi-card-portfolio-risk"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAME = "CardPortfolioRisk"

MONTHS = '''ADDCOLUMNS(FILTER(CALENDAR(DATE(2005, 4, 1), DATE(2005, 9, 30)), DAY([Date]) = 1),
    "Month", FORMAT([Date], "MMM yyyy"), "Month Sort", YEAR([Date]) * 100 + MONTH([Date]))'''

TABLES = [
    calc_table("Month", MONTHS, [col("Date", "dateTime", "mmm yyyy", key=True), col("Month", sort="Month Sort"), col("Month Sort", "int64", "0", hidden=True)],
               desc="Statement months, Apr - Sep 2005."),
    m_table("Cardholder", "dim_cardholder.csv", [
        col("CardholderKey", "int64", "0", hidden=True, key=True), col("CreditLimit", "double", "#,0", hidden=True), col("Default", "int64", "0", hidden=True),
        col("LimitBand", sort="LimitBandOrder", desc="Credit line, NT$"), col("AgeBand"), col("Education"), col("Segment", sort="SegmentOrder",
        desc="Behaviour over the last three statements"), col("LimitBandOrder", "int64", "0", hidden=True), col("SegmentOrder", "int64", "0", hidden=True),
        col("Utilisation", "double", "0%", hidden=True), col("UtilisationBand", sort="UtilisationBandOrder", desc="Sep balance / credit line"),
        col("UtilisationBandOrder", "int64", "0", hidden=True), col("ExposureAtDefault", "double", "#,0", hidden=True), col("PD", "double", "0.0%", hidden=True),
        col("Score", "int64", "0", desc="Behaviour score: 600 at 50:1 good:bad odds, +20 points doubles the odds"),
        col("ScoreBand", sort="ScoreBandOrder"), col("ScoreBandOrder", "int64", "0", hidden=True), col("Sample", desc="Train (70%) or hold-out Test (30%)")],
        desc="30,000 credit-card holders (UCI, Taiwan 2005) with segment, PD score and default in Oct 2005."),
    m_table("Bucket", "dim_bucket.csv", [col("Bucket", sort="BucketOrder", desc="Repayment status on the statement"), col("BucketOrder", "int64", "0", hidden=True)]),
    m_table("Statement", "fact_statement.csv", [
        col("CardholderKey", "int64", "0", hidden=True), col("Month", "dateTime", "mmm yyyy", hidden=True), col("Bucket", hidden=True),
        col("Delinquent", "int64", "0", hidden=True), col("Balance", "double", "#,0", hidden=True), col("Payment", "double", "#,0", hidden=True),
        col("Limit", "double", "#,0", hidden=True)], desc="Fact: one row per cardholder per monthly statement (180,000 rows). NT$."),
    m_table("Roll Rate", "fact_roll_rate.csv", [
        col("From", sort="FromOrder", desc="Status last month"), col("To", sort="ToOrder", desc="Status this month"), col("Cardholders", "int64", "#,0", hidden=True),
        col("Month", "dateTime", "mmm yyyy", hidden=True), col("FromOrder", "int64", "0", hidden=True), col("ToOrder", "int64", "0", hidden=True)],
        desc="Fact: month-to-month transitions between repayment statuses."),
    m_table("PD Drivers", "pd_feature_importance.csv", [col("Feature"), col("AUCDrop", "double", "0.000", hidden=True)],
            desc="Permutation importance on the hold-out sample."),
    m_table("Gains Curve", "pd_gains_curve.csv", [col("AccountsPct", "int64", "0", desc="Riskiest accounts actioned, %"),
                                                 col("DefaultsCapturedPct", "double", "0.0", hidden=True), col("RandomPct", "double", "0.0", hidden=True)]),
    m_table("Fraud Test", "fact_fraud_test.csv", [
        col("TxnKey", "int64", "0", hidden=True), col("Hour", "int64", "0"), col("Amount", "double", "#,0.00", hidden=True), col("IsFraud", "int64", "0", hidden=True),
        col("Score", "double", "0.0000", hidden=True), col("ReviewPctile", "double", "0.0", hidden=True)],
        desc="Fact: the 124,960 transactions in the forward test window (last ~17 hours) with their fraud score."),
    m_table("Fraud Hourly", "fact_fraud_hourly.csv", [
        col("Day"), col("Hour", "int64", "0"), col("AmountBand", sort="AmountBandOrder", desc="Transaction amount, EUR"), col("IsFraud", "int64", "0", hidden=True),
        col("Transactions", "int64", "#,0", hidden=True), col("Amount", "double", "#,0", hidden=True), col("AmountBandOrder", "int64", "0", hidden=True)],
        desc="All 284,807 transactions aggregated by day, hour, amount band and class."),
    m_table("Alert Curve", "fraud_alert_curve.csv", [
        col("ReviewPct", "double", "0.00", desc="Share of transactions sent for review, %"), col("Alerts", "int64", "#,0", hidden=True),
        col("FraudsCaught", "int64", "#,0", hidden=True), col("Recall", "double", "0%", hidden=True), col("Precision", "double", "0%", hidden=True),
        col("FraudAmountCaught", "double", "#,0", hidden=True)], desc="Alert budget vs fraud caught on the forward test window."),
    m_table("Model Metrics", "model_metrics.csv", [col("Metric"), col("Value", "double", "0.####", hidden=True)], hidden=True),
    calc_table("Score Cutoff", 'DATATABLE("Score Cutoff", INTEGER, {{450}, {475}, {500}, {525}, {550}, {575}, {600}})',
               [col("Score Cutoff", "int64", "0", desc="Accounts scoring below this get a line review")]),
    calc_table("Loss Given Default", 'DATATABLE("Loss Given Default", DOUBLE, {{0.6}, {0.75}, {0.9}})', [col("Loss Given Default", "double", "0%")]),
    calc_table("Review Share", 'DATATABLE("Review Share", DOUBLE, {{0.05}, {0.1}, {0.2}, {0.5}, {1}, {2}, {5}})',
               [col("Review Share", "double", "0.00\"%\"", desc="Share of transactions the fraud team can review")]),
    calc_table("Review Cost", 'DATATABLE("Review Cost", DOUBLE, {{2}, {5}, {10}, {20}})', [col("Review Cost", "double", "€0", desc="Cost of reviewing one alert, EUR")]),
]

FP, FD, FS, FL, FF, FM = "1 Portfolio", "2 Delinquency", "3 Scorecard", "4 Line strategy", "5 Fraud", "9 Model metrics"
LATEST = "VAR _m = MAX('Month'[Date])"


def metric(name, key, fmt, desc=None):
    return meas(name, f'LOOKUPVALUE(\'Model Metrics\'[Value], \'Model Metrics\'[Metric], "{key}")', fmt, FM, desc)


MEASURES = [
    meas("Cardholders", "COUNTROWS('Cardholder')", "#,0", FP),
    meas("Credit Lines", "SUM('Cardholder'[CreditLimit])", "#,0", FP, "Total credit lines, NT$."),
    meas("Default Rate", "AVERAGE('Cardholder'[Default])", "0.0%", FP, "Share of cardholders who defaulted in Oct 2005."),
    meas("Balances", f"{LATEST} RETURN CALCULATE(SUM('Statement'[Balance]), 'Month'[Date] = _m)", "#,0", FP, "Statement balances in the latest selected month, NT$."),
    meas("Utilisation", f"{LATEST} RETURN DIVIDE(CALCULATE(SUM('Statement'[Balance]), 'Month'[Date] = _m), CALCULATE(SUM('Statement'[Limit]), 'Month'[Date] = _m))",
         "0.0%", FP, "Balances / credit lines in the latest selected month."),
    meas("Statement Balances", "SUM('Statement'[Balance])", "#,0", FP),
    meas("Payments", "SUM('Statement'[Payment])", "#,0", FP),
    meas("Payment Rate", "DIVIDE([Payments], [Statement Balances])", "0.0%", FP, "Payments received as a share of balances."),
    meas("Statements", "COUNTROWS('Statement')", "#,0", FD),
    meas("Delinquency 30+", "DIVIDE(CALCULATE([Statements], 'Statement'[Delinquent] = 1), [Statements])", "0.0%", FD, "Share of statements 1+ month late."),
    meas("Delinquency 60+", '''DIVIDE(CALCULATE([Statements], 'Statement'[Bucket] IN {"2 months late", "3+ months late"}), [Statements])''', "0.0%", FD),
    meas("Latest 30+", f"{LATEST} RETURN CALCULATE([Delinquency 30+], 'Month'[Date] = _m)", "0.0%", FD),
    meas("Default Rate by Status", f'''{LATEST} RETURN DIVIDE(CALCULATE(SUMX('Statement', RELATED('Cardholder'[Default])), 'Month'[Date] = _m),
       CALCULATE([Statements], 'Month'[Date] = _m))''', "0%", FD, "October default rate by repayment status in the latest selected month."),
    meas("Revolving Share", f'''{LATEST} RETURN DIVIDE(CALCULATE([Statements], 'Statement'[Bucket] = "Revolving", 'Month'[Date] = _m), CALCULATE([Statements], 'Month'[Date] = _m))''',
         "0%", FD, "Share paying less than in full (revolving credit) in the latest month."),
    meas("Bucket Share", "DIVIDE([Statements], CALCULATE([Statements], REMOVEFILTERS('Bucket')))", "0%", FD),
    meas("Roll Share", "DIVIDE(SUM('Roll Rate'[Cardholders]), CALCULATE(SUM('Roll Rate'[Cardholders]), REMOVEFILTERS('Roll Rate'[To], 'Roll Rate'[ToOrder])))",
         "0%", FD, "Share of last month's status that moved to this month's status."),
    meas("Cure Rate", '''DIVIDE(CALCULATE(SUM('Roll Rate'[Cardholders]), 'Roll Rate'[FromOrder] >= 4, 'Roll Rate'[ToOrder] <= 3),
       CALCULATE(SUM('Roll Rate'[Cardholders]), 'Roll Rate'[FromOrder] >= 4))''', "0%", FD, "Past due last month, current this month."),
    meas("Still Late Rate", '''DIVIDE(CALCULATE(SUM('Roll Rate'[Cardholders]), 'Roll Rate'[FromOrder] >= 4, 'Roll Rate'[ToOrder] >= 4),
       CALCULATE(SUM('Roll Rate'[Cardholders]), 'Roll Rate'[FromOrder] >= 4))''', "0%", FD, "Past due last month and still past due (or worse) this month."),
    meas("Entry Rate", '''DIVIDE(CALCULATE(SUM('Roll Rate'[Cardholders]), 'Roll Rate'[FromOrder] <= 3, 'Roll Rate'[ToOrder] >= 4),
       CALCULATE(SUM('Roll Rate'[Cardholders]), 'Roll Rate'[FromOrder] <= 3))''', "0.0%", FD, "Current last month, past due this month: the flow into delinquency."),
    meas("Test Accounts", "CALCULATE([Cardholders], 'Cardholder'[Sample] = \"Test\")", "#,0", FS),
    meas("Test Default Rate", "CALCULATE([Default Rate], 'Cardholder'[Sample] = \"Test\")", "0.0%", FS),
    meas("Test Predicted PD", "CALCULATE(AVERAGE('Cardholder'[PD]), 'Cardholder'[Sample] = \"Test\")", "0.0%", FS),
    meas("Test Defaults", "CALCULATE(SUM('Cardholder'[Default]), 'Cardholder'[Sample] = \"Test\")", "#,0", FS),
    meas("Test Goods", "[Test Accounts] - [Test Defaults]", "#,0", FS),
    meas("Driver Importance", "SUM('PD Drivers'[AUCDrop])", "0.000", FS),
    meas("Defaults Captured", "SUM('Gains Curve'[DefaultsCapturedPct]) / 100", "0%", FS),
    meas("Random Capture", "SUM('Gains Curve'[RandomPct]) / 100", "0%", FS),
    metric("PD AUC", "pd_auc_test", "0.00", "Hold-out AUC of the gradient-boosting scorecard."),
    metric("PD Gini", "pd_gini_test", "0.00"),
    metric("PD KS", "pd_ks_test", "0.00", "Kolmogorov-Smirnov separation on the hold-out sample."),
    metric("Logistic AUC", "pd_auc_logit", "0.00", "Hold-out AUC of a logistic-regression benchmark."),
    meas("Cutoff Value", "SELECTEDVALUE('Score Cutoff'[Score Cutoff], 525)", "0", FL),
    meas("LGD Value", "SELECTEDVALUE('Loss Given Default'[Loss Given Default], 0.75)", "0%", FL),
    meas("Actioned Accounts", '''VAR _c = [Cutoff Value] RETURN CALCULATE(COUNTROWS('Cardholder'), 'Cardholder'[Sample] = "Test", 'Cardholder'[Score] < _c)''', "#,0", FL,
         "Hold-out accounts scoring below the cutoff: candidates for a line decrease or early contact."),
    meas("Actioned Share", "DIVIDE([Actioned Accounts], [Test Accounts])", "0%", FL),
    meas("Defaults Caught", '''VAR _c = [Cutoff Value] RETURN DIVIDE(CALCULATE(SUM('Cardholder'[Default]), 'Cardholder'[Sample] = "Test", 'Cardholder'[Score] < _c), [Test Defaults])''',
         "0%", FL, "Share of next-month defaults inside the actioned group."),
    meas("Bad Rate Actioned", '''VAR _c = [Cutoff Value] RETURN CALCULATE(AVERAGE('Cardholder'[Default]), 'Cardholder'[Sample] = "Test", 'Cardholder'[Score] < _c)''', "0%", FL),
    meas("Bad Rate Kept", '''VAR _c = [Cutoff Value] RETURN CALCULATE(AVERAGE('Cardholder'[Default]), 'Cardholder'[Sample] = "Test", 'Cardholder'[Score] >= _c)''', "0.0%", FL),
    meas("Good Accounts Actioned", '''VAR _c = [Cutoff Value] RETURN CALCULATE(COUNTROWS('Cardholder'), 'Cardholder'[Sample] = "Test", 'Cardholder'[Score] < _c, 'Cardholder'[Default] = 0)''',
         "#,0", FL, "Customers who would not have defaulted but fall below the cutoff: the cost of the policy."),
    meas("Expected Loss", '''CALCULATE(SUMX('Cardholder', 'Cardholder'[PD] * 'Cardholder'[ExposureAtDefault]), 'Cardholder'[Sample] = "Test") * [LGD Value]''', "#,0", FL,
         "PD x exposure at default x LGD, hold-out accounts, NT$."),
    meas("Expected Loss Actioned", '''VAR _c = [Cutoff Value] RETURN CALCULATE(SUMX('Cardholder', 'Cardholder'[PD] * 'Cardholder'[ExposureAtDefault]),
    'Cardholder'[Sample] = "Test", 'Cardholder'[Score] < _c) * [LGD Value]''', "#,0", FL),
    meas("Expected Loss Share Actioned", "DIVIDE([Expected Loss Actioned], [Expected Loss])", "0%", FL),
    meas("Fraud Transactions", "COUNTROWS('Fraud Test')", "#,0", FF),
    meas("Frauds", "SUM('Fraud Test'[IsFraud])", "#,0", FF),
    meas("Fraud Amount", "CALCULATE(SUM('Fraud Test'[Amount]), 'Fraud Test'[IsFraud] = 1)", "€#,0", FF),
    meas("Fraud Rate", "DIVIDE(CALCULATE(SUM('Fraud Hourly'[Transactions]), 'Fraud Hourly'[IsFraud] = 1), SUM('Fraud Hourly'[Transactions]))", "0.00%", FF,
         "Fraud share of transactions, both days."),
    meas("Fraud Basis Points", "[Fraud Rate] * 10000", "0.0", FF, "Frauds per 10,000 transactions."),
    meas("Fraud Share of Amount", "DIVIDE(CALCULATE(SUM('Fraud Hourly'[Amount]), 'Fraud Hourly'[IsFraud] = 1), SUM('Fraud Hourly'[Amount]))", "0.00%", FF),
    metric("Fraud ROC AUC", "fraud_roc_auc", "0.000"),
    metric("Fraud PR AUC", "fraud_pr_auc", "0.00", "Average precision: the right yardstick when fraud is 0.17% of transactions."),
    metric("Recall at 0.1%", "fraud_recall_at_0.1pct", "0%", "Share of fraud caught when reviewing the riskiest 0.1% of transactions."),
    metric("Precision at 0.1%", "fraud_precision_at_0.1pct", "0%"),
    meas("Review Share Value", "SELECTEDVALUE('Review Share'[Review Share], 0.1)", "0.00\"%\"", FF),
    meas("Review Cost Value", "SELECTEDVALUE('Review Cost'[Review Cost], 5)", "€0", FF),
    meas("Sim Alerts", "LOOKUPVALUE('Alert Curve'[Alerts], 'Alert Curve'[ReviewPct], [Review Share Value])", "#,0", FF),
    meas("Sim Frauds Caught", "LOOKUPVALUE('Alert Curve'[FraudsCaught], 'Alert Curve'[ReviewPct], [Review Share Value])", "#,0", FF),
    meas("Sim Recall", "LOOKUPVALUE('Alert Curve'[Recall], 'Alert Curve'[ReviewPct], [Review Share Value])", "0%", FF),
    meas("Sim Precision", "LOOKUPVALUE('Alert Curve'[Precision], 'Alert Curve'[ReviewPct], [Review Share Value])", "0%", FF),
    meas("Sim Fraud Prevented", "LOOKUPVALUE('Alert Curve'[FraudAmountCaught], 'Alert Curve'[ReviewPct], [Review Share Value])", "€#,0", FF),
    meas("Sim Review Cost", "[Sim Alerts] * [Review Cost Value]", "€#,0", FF),
    meas("Sim Net Saving", "[Sim Fraud Prevented] - [Sim Review Cost]", "€#,0;-€#,0", FF),
    meas("Curve Recall", "SUM('Alert Curve'[Recall])", "0%", FF),
    meas("Curve Precision", "SUM('Alert Curve'[Precision])", "0%", FF),
    meas("Curve Net Saving", "SUM('Alert Curve'[FraudAmountCaught]) - SUM('Alert Curve'[Alerts]) * [Review Cost Value]", "€#,0", FF),
]

RELS = [rel("Statement", "CardholderKey", "Cardholder", "CardholderKey"), rel("Statement", "Month", "Month", "Date"),
        rel("Statement", "Bucket", "Bucket", "Bucket"), rel("Roll Rate", "Month", "Month", "Date")]


def build(local=True):
    tables = TABLES + [measure_table("KPI", MEASURES, "All report measures, organised in display folders.")]
    exprs = csv_expressions(REPO, os.path.join(ROOT, "data") + "\\", default="Local" if local else "GitHub")
    write_model(os.path.join(ROOT, f"{NAME}.SemanticModel"), NAME, tables, RELS, exprs,
                desc="Card Portfolio & Risk: UCI credit-card portfolio (Taiwan 2005) and ULB card-fraud transactions (Europe 2013).")


if __name__ == "__main__":
    build(local="--github" not in sys.argv)
    print("model written")
