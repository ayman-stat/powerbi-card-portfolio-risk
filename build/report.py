"""Report definition (PBIR) for the Card Portfolio & Risk report.

Identity: CARD - graphite header with a segmented page control, cobalt for the data in focus, coral for risk and fraud,
KPIs stacked in a left column so charts get the full height. Framed as the analysis a card network's advisory team
delivers to an issuing bank. Not affiliated with, or endorsed by, any card network.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "toolkit"))  # vendored copy of the shared design toolkit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_tools"))  # master copy, when present
from pbip import Report, write_pbip, C, M, proj, b, i, n, s, props, idprops, color, lit  # noqa: E402
import design as D  # noqa: E402
from design import S, GUT, W, use, theme, chrome, kpi_band, bar, combo, line, matrix, table, stacked_column, button_slicer, dropdown, about_page, P, _title, fill_for  # noqa: E402

use(D.CARD)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAME = "CardPortfolioRisk"
TITLE = "Card Portfolio & Risk"
SRC_U = "UCI Default of Credit Card Clients (Yeh & Lien, 2009), CC BY 4.0: 30,000 cardholders, statements Apr - Sep 2005, NT$."
SRC_F = "Credit Card Fraud Detection (ULB Machine Learning Group & Worldline, 2013) via OpenML: 284,807 European card transactions, 492 frauds."
K = lambda m: M("KPI", m)  # noqa: E731
NP = 7


def buttons(p, key, x, y, w, h, field, literal, cols, title):
    ent, prop = field["Column"]["Expression"]["SourceRef"]["Entity"], field["Column"]["Property"]
    return p.add(key, "advancedSlicerVisual", x, y, w, h, query={"Values": [proj(field)]},
                 objects={"layout": props(columnCount=i(cols), rowCount=i(1)),
                          "value": idprops(fontSize=n(12), fontFamily=s(S.font_b), fontColor=color(S.ink)),
                          "fillCustom": [{"properties": {"show": b(True), "fillColor": color(S.panel)}, "selector": {"id": "default"}},
                                         {"properties": {"show": b(True), "fillColor": color(S.sel_fill)}, "selector": {"id": "selected"}}],
                          "outline": [{"properties": {"show": b(True), "lineColor": color(S.line), "weight": n(1)}, "selector": {"id": "default"}},
                                      {"properties": {"show": b(True), "lineColor": color(S.primary), "weight": n(2)}, "selector": {"id": "selected"}}],
                          "selection": props(singleSelect=b(True)),
                          "general": [{"properties": {"filter": {"filter": {"Version": 2, "From": [{"Name": "f", "Entity": ent, "Type": 0}],
                              "Where": [{"Condition": {"In": {"Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "f"}}, "Property": prop}}],
                                                               "Values": [[{"Literal": {"Value": literal}}]]}}}]}}}}]},
                 vco={"title": props(show=b(True), text=s(title), fontSize=n(10), fontColor=color(S.muted), fontFamily=s(S.font)),
                      "background": props(show=b(False)), "border": props(show=b(False)), "dropShadow": props(show=b(False)),
                      "padding": props(top=n(0), bottom=n(0), left=n(0), right=n(0))}, alt=title)


def band_card(p, key, x, y, w, h, measures):
    """One panel with several values side by side (simulator read-out)."""
    D.panel(p, key + "-bg", x, y, w, h)
    p.add(key, "cardVisual", x + 8, y + 10, w - 16, h - 20, query={"Data": [P(m) for m in measures]},
          objects={"value": idprops(fontSize=n(24), fontColor=color(S.ink), fontFamily=s(S.font_num), horizontalAlignment=s("left"), labelDisplayUnits=s("0")),
                   "label": idprops(show=b(True), fontSize=n(11), fontColor=color(S.muted), position=s("aboveValue"), horizontalAlignment=s("left")),
                   "outline": idprops(show=b(False)), "fillCustom": [{"properties": {"show": b(False)}}],
                   "layout": idprops(style=s("Table"), customizeLines=b(True), gridlineWidth=n(1), gridlineColor=color(S.line), gridlineStyle=s("solid")),
                   "padding": idprops(leftMargin=i(16), rightMargin=i(8), topMargin=i(0), bottomMargin=i(0))},
          vco=dict(D.NOBOX), alt="Simulator read-out")


def build():
    r = Report(os.path.join(ROOT, f"{NAME}.Report"), NAME, f"{NAME}.SemanticModel", theme())
    r.n_pages = NP
    bx, bw, by, bh = S.bx, S.bw, S.body_y, S.body_h
    L = 875
    R = bw - L - GUT

    # ------------------------------------------------------------------ 1 Portfolio overview
    p = r.page("portfolio", "Portfolio", bg=S.page)
    chrome(p, TITLE, "1 in 5 cardholders defaulted in October; those already late defaulted at 47%", "30,000 cards, NT$5.0bn of lines, 31% utilised  ·  late on a recent statement: 28% of cardholders, 47% default vs 12-14% for everyone else", f"1 / {NP}", SRC_U)
    dropdown(p, "seg", S.slicer_x, S.title_y + 4, 230, 76, C("Cardholder", "Segment"), "Behaviour segment")
    dropdown(p, "lim", S.slicer_x + 250, S.title_y + 4, 230, 76, C("Cardholder", "LimitBand"), "Credit line")
    kpi_band(p, "kpi", [(K("Cardholders"), "Cardholders"), (K("Credit Lines"), "Credit lines, NT$", 1000000000), (K("Balances"), "Balances, Sep, NT$", 1000000000),
                        (K("Utilisation"), "Utilisation, Sep"), (K("Latest 30+"), "30+ days late, Sep"), (K("Default Rate"), "Defaulted in October")])
    th = 400
    combo(p, "trend", bx, by, L, th, (C("Month", "Month"), "Statement month"), [(K("Statement Balances"), "Balances")], [(K("Payment Rate"), "Payment rate")],
          "Balances and payments, six statements", "Columns: balances, NT$  ·  line: payments as a share of balances", sort_cat=C("Month", "Month Sort"),
          col_colors=("#9DB2F6",), line_colors=(S.primary,))
    bar(p, "seg", bx + L + GUT, by, R, th, (C("Cardholder", "Segment"), "Segment"), (K("Default Rate"), "Default rate"),
        "Default rate by behaviour segment", "Segment from the last three statements", sort_desc=False, sort_cat=C("Cardholder", "SegmentOrder"),
        gradient=(S.soft_neg, S.neg))
    bar(p, "util", bx, by + th + GUT, L, bh - th - GUT, (C("Cardholder", "UtilisationBand"), "Utilisation (Sep)"), (K("Default Rate"), "Default rate"),
        "Default rate rises with utilisation", "Share of each utilisation band that defaulted in October", horizontal=False, sort_desc=False,
        sort_cat=C("Cardholder", "UtilisationBandOrder"), gradient=(S.soft_neg, S.neg))
    matrix(p, "lim", bx + L + GUT, by + th + GUT, R, bh - th - GUT, [(C("Cardholder", "LimitBand"), "Credit line")],
           [(K("Cardholders"), "Cardholders"), (K("Utilisation"), "Utilisation"), (K("Default Rate"), "Default rate")],
           "Credit-line scorecard", "Data bars: cardholders  ·  default rate shaded green (low) to red (high)",
           gradients=[(K("Default Rate"), S.soft_pos, S.soft_neg, S.panel)], databars=[K("Cardholders")], font=11)

    # ------------------------------------------------------------------ 2 Delinquency & roll rates
    p = r.page("delinquency", "Delinquency", bg=S.page)
    chrome(p, TITLE, "Two months late is the point of no return: 69% default the next month", "Only 1 in 4 past-due accounts cures each month  ·  in the source data, one month late is recorded only on the latest statement", f"2 / {NP}", SRC_U)
    dropdown(p, "seg", S.slicer_x + 250, S.title_y + 4, 230, 76, C("Cardholder", "Segment"), "Behaviour segment")
    kpi_band(p, "kpi", [(K("Latest 30+"), "30+ days late, Sep"), (K("Delinquency 60+"), "60+ days late, 6 months"), (K("Cure Rate"), "Past due that cure next month"),
                        (K("Entry Rate"), "Current that fall past due"), (K("Revolving Share"), "Revolving, Sep"), (K("Statements"), "Statements analysed")])
    th = 380
    matrix(p, "roll", bx, by, L, th, [(C("Roll Rate", "From"), "Last month")], [(K("Roll Share"), "Share")],
           "Roll-rate matrix: where last month's status went", "Rows: status last month  ·  columns: status this month  ·  darker = larger share",
           columns=[(C("Roll Rate", "To"), "This month")], gradients=[(K("Roll Share"), S.panel, S.primary)], font=11)
    stacked_column(p, "mix", bx + L + GUT, by, R, th, (C("Month", "Month"), "Month"), [(K("Bucket Share"), "Share")],
                   "Status mix by month", "Share of statements in each repayment status", sort_cat=C("Month", "Month Sort"), series=C("Bucket", "Bucket"), pct=True)
    line(p, "dq", bx, by + th + GUT, L, bh - th - GUT, (C("Month", "Month"), "Month"), [(K("Delinquency 30+"), "30+ days"), (K("Delinquency 60+"), "60+ days")],
         "Delinquency trend", "Share of statements past due", colors=(S.primary, S.accent), sort_cat=C("Month", "Month Sort"), markers=True)
    bar(p, "dstat", bx + L + GUT, by + th + GUT, R, bh - th - GUT, (C("Bucket", "Bucket"), "Status in Sep"), (K("Default Rate by Status"), "Default rate"),
        "What September status says about October", "Default rate in October by September repayment status", sort_desc=False,
        sort_cat=C("Bucket", "BucketOrder"), gradient=(S.soft_neg, S.neg))

    # ------------------------------------------------------------------ 3 PD scorecard (ML)
    p = r.page("scorecard", "Risk scorecard", bg=S.page)
    chrome(p, TITLE, "The behaviour score separates risk 30-fold: 86% default below 450, 3% above 575", "Gradient boosting, hold-out AUC 0.78 vs 0.74 for logistic regression  ·  behaviour only, no demographic inputs", f"3 / {NP}", SRC_U)
    kpi_band(p, "kpi", [(K("PD AUC"), "Hold-out AUC"), (K("PD Gini"), "Gini"), (K("PD KS"), "KS statistic"), (K("Logistic AUC"), "Logistic benchmark AUC"),
                        (K("Test Default Rate"), "Hold-out default rate"), (K("Test Accounts"), "Hold-out accounts")])
    th = 420
    combo(p, "bands", bx, by, L, th, (C("Cardholder", "ScoreBand"), "Score band"), [(K("Test Accounts"), "Accounts")], [(K("Test Default Rate"), "Default rate")],
          "Default rate falls steeply as the score rises", "Hold-out accounts  ·  columns: accounts  ·  line: share defaulting in October",
          sort_cat=C("Cardholder", "ScoreBandOrder"), col_colors=("#C9D5FA",), line_colors=(S.accent,))
    bar(p, "drv", bx + L + GUT, by, R, th, (C("PD Drivers", "Feature"), "Feature"), (K("Driver Importance"), "AUC drop"), "What drives the score",
        "Permutation importance on hold-out data  ·  behaviour only: no sex, marital status, age or education")
    p.add("calib", "clusteredColumnChart", bx, by + th + GUT, L, bh - th - GUT,
          query={"Category": [P((C("Cardholder", "ScoreBand"), "Score band"))],
                 "Y": [P((K("Test Default Rate"), "Actual")), P((K("Test Predicted PD"), "Predicted"))]},
          objects={"dataPoint": [fill_for(K("Test Default Rate"), S.accent), fill_for(K("Test Predicted PD"), S.primary)],
                   "legend": props(show=b(True), position=s("Top")), "labels": props(show=b(False)),
                   "valueAxis": props(show=b(True), gridlineShow=b(True), labelColor=color(S.muted))},
          vco=_title("Calibration: predicted vs actual default rate", "Hold-out accounts by score band, one axis"),
          sort=[(C("Cardholder", "ScoreBandOrder"), "Ascending")], alt="Calibration by score band")
    line(p, "gains", bx + L + GUT, by + th + GUT, R, bh - th - GUT, (C("Gains Curve", "AccountsPct"), "Riskiest accounts reviewed, %"),
         [(K("Defaults Captured"), "Scorecard"), (K("Random Capture"), "Random")], "Gains curve", "Share of October defaults found in the riskiest x% of accounts",
         colors=(S.primary, S.grey), sort_cat=C("Gains Curve", "AccountsPct"), markers=True)

    # ------------------------------------------------------------------ 4 Credit-line strategy (what-if)
    p = r.page("strategy", "Line strategy", bg=S.page)
    chrome(p, TITLE, "Reviewing lines below 525 targets 31% of accounts and catches 63% of defaults", "Hold-out accounts  ·  the actioned group carries 54% of expected loss  ·  change the cutoff and loss-given-default above", f"4 / {NP}", SRC_U)
    kpi_band(p, "kpi", [(K("Actioned Accounts"), "Accounts below cutoff"), (K("Actioned Share"), "Share of accounts"), (K("Defaults Caught"), "October defaults caught"),
                        (K("Bad Rate Actioned"), "Default rate, actioned"), (K("Bad Rate Kept"), "Default rate, kept"),
                        (K("Expected Loss Share Actioned"), "Expected loss in actioned group")])
    D.panel(p, "ctrl", bx, by, bw, 92)
    buttons(p, "cut", bx + 20, by + 8, 900, 76, C("Score Cutoff", "Score Cutoff"), "525L", 7, "Score cutoff: accounts below it get a line review")
    buttons(p, "lgd", bx + 944, by + 8, bw - 964, 76, C("Loss Given Default", "Loss Given Default"), "0.75D", 3, "Loss given default (assumption)")
    cy, ch = by + 92 + GUT, bh - 92 - GUT
    hh = (ch - GUT) // 2
    bar(p, "el", bx, cy, L, hh, (C("Cardholder", "ScoreBand"), "Score band"), (K("Expected Loss"), "Expected loss"),
        "Expected loss by score band", "PD x exposure x LGD, hold-out accounts, NT$", horizontal=False, sort_desc=False,
        sort_cat=C("Cardholder", "ScoreBandOrder"), gradient=(S.soft_neg, S.neg))
    p.add("gb", "columnChart", bx, cy + hh + GUT, L, ch - hh - GUT,
          query={"Category": [P((C("Cardholder", "ScoreBand"), "Score band"))], "Y": [P((K("Test Goods"), "Paid as agreed")), P((K("Test Defaults"), "Defaulted"))]},
          objects={"dataPoint": [fill_for(K("Test Goods"), "#C9D5FA"), fill_for(K("Test Defaults"), S.accent)],
                   "legend": props(show=b(True), position=s("Top")), "labels": props(show=b(False)),
                   "valueAxis": props(show=b(True), gridlineShow=b(True), labelColor=color(S.muted))},
          vco=_title("Who sits in each band", "Hold-out accounts that paid as agreed vs defaulted"),
          sort=[(C("Cardholder", "ScoreBandOrder"), "Ascending")], alt="Goods and defaults by score band")
    matrix(p, "bandtab", bx + L + GUT, cy, R, hh, [(C("Cardholder", "ScoreBand"), "Score band")],
           [(K("Test Accounts"), "Accounts"), (K("Test Default Rate"), "Default rate"), (K("Expected Loss"), "Expected loss")],
           "Score-band table", "Hold-out accounts  ·  default rate shaded green to red  ·  data bars: expected loss",
           gradients=[(K("Test Default Rate"), S.soft_pos, S.soft_neg, S.panel)], databars=[K("Expected Loss")], font=11)
    bar(p, "actseg", bx + L + GUT, cy + hh + GUT, R, ch - hh - GUT, (C("Cardholder", "Segment"), "Segment"), (K("Actioned Accounts"), "Accounts actioned"),
        "Who the policy touches", "Hold-out accounts below the cutoff, by behaviour segment", gradient=(S.light, S.primary))

    # ------------------------------------------------------------------ 5 Fraud detection (ML)
    p = r.page("fraud", "Fraud detection", bg=S.page)
    chrome(p, TITLE, "Reviewing 0.1% of transactions stops 76% of fraud, at 83% precision", "Trained on the first 31 hours, tested on the next 17  ·  fraud is 0.17% of transactions, so precision and recall, not accuracy, are the measures", f"5 / {NP}", SRC_F)
    kpi_band(p, "kpi", [(K("Fraud Transactions"), "Transactions in the test window"), (K("Frauds"), "Frauds in the test window"),
                        (K("Fraud Basis Points"), "Fraud per 10,000 transactions"), (K("Fraud ROC AUC"), "ROC AUC"), (K("Fraud PR AUC"), "PR AUC"),
                        (K("Recall at 0.1%"), "Fraud caught reviewing 0.1%")])
    D.panel(p, "ctrl", bx, by, bw, 92)
    buttons(p, "share", bx + 20, by + 8, 900, 76, C("Review Share", "Review Share"), "0.1D", 7, "Share of transactions the team reviews (%), riskiest first")
    buttons(p, "cost", bx + 944, by + 8, bw - 964, 76, C("Review Cost", "Review Cost"), "5D", 4, "Cost of one review, EUR")
    band_card(p, "sim", bx, by + 92 + GUT, bw, 110,
              [(K("Sim Alerts"), "Alerts"), (K("Sim Frauds Caught"), "Frauds caught"), (K("Sim Recall"), "Recall"), (K("Sim Precision"), "Precision"),
               (K("Sim Fraud Prevented"), "Fraud prevented"), (K("Sim Review Cost"), "Review cost"), (K("Sim Net Saving"), "Net saving")])
    cy = by + 92 + GUT + 110 + GUT
    ch = by + bh - cy
    p.add("pr", "lineChart", bx, cy, L, ch,
          query={"Category": [P((C("Alert Curve", "ReviewPct"), "Transactions reviewed, %"))],
                 "Y": [P((K("Curve Recall"), "Recall")), P((K("Curve Precision"), "Precision"))]},
          objects={"dataPoint": [fill_for(K("Curve Recall"), S.primary), fill_for(K("Curve Precision"), S.accent)],
                   "lineStyles": props(showMarker=b(True), strokeWidth=n(2.5)), "categoryAxis": props(axisType=s("Categorical"))},
          vco=_title("The alert-budget trade-off", "Forward test (last ~17 hours): recall climbs fast, precision collapses after 0.1%"),
          sort=[(C("Alert Curve", "ReviewPct"), "Ascending")], alt="Recall and precision by review share")
    hh = (ch - GUT) // 2
    bar(p, "hour", bx + L + GUT, cy, R, hh, (C("Fraud Hourly", "Hour"), "Hour of day"), (K("Fraud Basis Points"), "Fraud per 10K"),
        "Fraud intensity by hour", "Frauds per 10,000 transactions, both days", horizontal=False, sort_desc=False, sort_cat=C("Fraud Hourly", "Hour"),
        gradient=(S.soft_neg, S.neg))
    bar(p, "amt", bx + L + GUT, cy + hh + GUT, R, ch - hh - GUT, (C("Fraud Hourly", "AmountBand"), "Amount, EUR"), (K("Fraud Basis Points"), "Fraud per 10K"),
        "Fraud intensity by amount", "Frauds per 10,000 transactions", sort_desc=False, sort_cat=C("Fraud Hourly", "AmountBandOrder"), gradient=(S.soft_neg, S.neg))

    # ------------------------------------------------------------------ 6 How card networks make money
    p = r.page("network", "Network economics", bg=S.page)
    chrome(p, TITLE, "How a card network makes money, and where this analysis fits", "The four-party model behind every card payment  ·  a primer, not company data",
           f"6 / {NP}", "Public industry descriptions of the four-party card model. Not affiliated with, or endorsed by, any card network.")
    x0, cw = S.x0, S.cw
    nodes = [("Cardholder", "Pays with the card; repays the issuer"), ("Issuer", "Cardholder's bank: grants the line, approves the purchase, carries credit and fraud risk"),
             ("Card network", "Routes authorisation, clearing and settlement; sets the rules; earns fees on volume and transactions"),
             ("Acquirer", "Merchant's bank: accepts the card for the merchant"), ("Merchant", "Sells the goods; pays the merchant service charge")]
    nw = (cw - 4 * 40) // 5
    ny = 200
    for k_, (head, txt) in enumerate(nodes):
        nx = x0 + k_ * (nw + 40)
        D.panel(p, f"node{k_}", nx, ny, nw, 250, fill=("#E8EEFD" if head == "Card network" else None))
        D.shape(p, f"node{k_}-bar", nx, ny, nw, 5, S.primary if head == "Card network" else S.grey)
        D.textbox(p, f"node{k_}-t", nx + 22, ny + 24, nw - 44, 210, [[(head, 26, S.ink, True)], [(" ", 8, S.ink, False)], [(txt, 18, S.muted, False)]])
        if k_ < 4:
            D.textbox(p, f"arrow{k_}", nx + nw, ny + 100, 40, 40, [["center", ("↔", 24, S.primary, True)]])
    flows = [
        ("Where the money comes from", ["The merchant pays a service charge on each card sale.",
                                        "Most of it is interchange, paid by the acquirer to the issuer.",
                                        "The network charges issuers and acquirers fees on volume and per transaction, plus higher fees on cross-border payments.",
                                        "The issuer also earns interest and fees from cardholders who revolve."]),
        ("What a network watches", ["Gross volume and number of transactions switched.", "Cross-border volume, the highest-margin flow.",
                                    "Authorisation approval rate: every false decline is lost volume.", "Fraud in basis points of volume, and chargebacks.",
                                    "Growth in value-added services: fraud scoring, tokenisation, data and advisory."]),
        ("Where this report fits", ["Pages 1-4 are what a network's advisory team builds for an issuer: portfolio health, roll rates, a PD scorecard and a credit-line policy.",
                                    "Page 5 is the fraud side: a model that catches 76% of fraud while reviewing 0.1% of transactions.",
                                    "Healthier issuer portfolios and fewer false declines mean more good volume on the network."]),
    ]
    fw = (cw - 2 * GUT) // 3
    fy = ny + 250 + 28
    fh = 330
    for k_, (head, pts) in enumerate(flows):
        fx = x0 + k_ * (fw + GUT)
        D.panel(p, f"flow{k_}", fx, fy, fw, fh)
        D.shape(p, f"flow{k_}-a", fx, fy + 20, 4, 60, S.accent if k_ == 2 else S.primary)
        paras = [[(head, 22, S.ink, True)], [(" ", 8, S.ink, False)]]
        for t_ in pts:
            paras += [[("•  " + t_, 16, S.ink, False)], [(" ", 8, S.ink, False)]]
        D.textbox(p, f"flow{k_}-t", fx + 28, fy + 20, fw - 52, fh - 30, paras)
    gy = fy + fh + GUT
    gh = S.foot_y - 20 - gy
    D.panel(p, "gloss", x0, gy, cw, gh)
    D.textbox(p, "gloss-h", x0 + 28, gy + 14, 600, 34, [[("Payments vocabulary", 20, S.ink, True)]])
    terms = [("Authorisation", "Real-time approve or decline of a purchase by the issuer, routed by the network."),
             ("Clearing", "Exchange of final transaction details between acquirer and issuer after the sale."),
             ("Settlement", "The movement of funds between issuer and acquirer, net of interchange."),
             ("Interchange", "Fee the acquirer pays the issuer on each sale, set by network rules and capped by regulators in some markets."),
             ("Chargeback", "A disputed transaction reversed back through the network to the merchant."),
             ("Tokenisation", "Replacing the card number with a token for wallets and online payments, to cut fraud."),
             ("Cross-border", "Issuer and acquirer in different countries: extra fees and currency conversion."),
             ("Approval rate", "Share of authorisations approved; false declines cost volume and customer trust.")]
    cwid = (cw - 56 - 3 * GUT) // 4
    rowh = (gh - 60) // 2
    for k_, (term, txt) in enumerate(terms):
        cx = x0 + 28 + (k_ % 4) * (cwid + GUT)
        cy2 = gy + 54 + (k_ // 4) * rowh
        D.textbox(p, f"term{k_}", cx, cy2, cwid, rowh - 6, [[(term, 15, S.primary, True)], [(txt, 13, S.muted, False)]])

    # ------------------------------------------------------------------ 7 About
    p = r.page("about", "About & method", bg=S.page)
    chrome(p, TITLE, "How this report is built", "Questions, data, models and definitions behind every number", f"7 / {NP}",
           "UCI Default of Credit Card Clients (CC BY 4.0) and ULB Credit Card Fraud Detection (DbCL, via OpenML).")
    about_page(p, [
        ("Business questions", ["How healthy is the card book, and where is risk building?", "How do accounts move between current and past due?",
                                "Who will default next month, and what line policy limits the loss?", "How much fraud can a fixed review budget stop?"]),
        ("Data & ML", ["UCI: 30,000 cardholders, six statements each (180,000 rows); default observed in the following month.",
                       "PD: gradient boosting on behaviour only (no sex, marital status, age or education); hold-out AUC 0.78, Gini 0.55, KS 0.42.",
                       "Fraud: 284,807 transactions; trained on the first ~31 hours, tested forward on the next ~17 hours; ROC AUC 0.98, PR AUC 0.74."]),
        ("Model & DAX", ["Statement fact with Cardholder, Month and Bucket dimensions; roll-rate, gains and alert-curve tables.",
                         "What-if tables for score cutoff, loss given default, review share and review cost.",
                         "Scores scaled to 600 points at 50:1 odds, 20 points to double the odds.",
                         "Source quirk: 'one month late' appears only on the latest statement; roll rates use any past-due status."]),
    ], [("30+ / 60+", "statements one / two or more months past due."), ("Roll rate", "share of accounts moving from one status to another between months."),
        ("PD", "probability of default next month; expected loss = PD x exposure x LGD."),
        ("Gini / KS", "rank-ordering power of the score (Gini = 2 x AUC - 1)."),
        ("Recall / precision", "share of all fraud caught / share of alerts that are fraud.")],
        "Built by Mohamed Ayman, Senior Data Scientist  ·  github.com/ayman-stat  ·  mohamedaymanstat.vercel.app")

    r.write()
    write_pbip(os.path.join(ROOT, f"{NAME}.pbip"), f"{NAME}.Report")


if __name__ == "__main__":
    build()
    print("report written")
