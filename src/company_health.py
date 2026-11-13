"""
company_health.py
=================
A transparent, rule-based financial HEALTH SCORE (0-100).

This is deliberately separate from the machine-learning valuation model:
* it is NOT trained on data and NOT validated against outcomes such as bankruptcies;
* every rule, threshold and weight is written down in HEALTH_RULES below;
* it only uses the company's own financial statements (no stock price, no ML output).
"""

import numpy as np
import pandas as pd

import valuation_pipeline as vp

# poor  -> indicator value that scores 0 points
# good  -> indicator value that scores 100 points (linear in between, clipped to 0..100)
# "poor" may be larger than "good" for indicators where lower is better (e.g. debt).
HEALTH_RULES = {
    "Profitability": {"weight": 0.25, "indicators": {
        "operating_margin": {"poor": 0.00, "good": 0.25, "text": "Operating income / revenue"},
        "net_margin":       {"poor": 0.00, "good": 0.20, "text": "Net income / revenue"},
        "roa":              {"poor": 0.00, "good": 0.10, "text": "Net income / total assets"},
    }},
    "Liquidity": {"weight": 0.15, "indicators": {
        "current_ratio":    {"poor": 0.80, "good": 2.00, "text": "Current assets / current liabilities"},
        "cash_ratio":       {"poor": 0.10, "good": 1.00, "text": "Cash & short-term investments / current liabilities"},
    }},
    "Leverage": {"weight": 0.20, "indicators": {
        "debt_to_equity":   {"poor": 3.00, "good": 0.50, "text": "Total debt / shareholders' equity (lower is better; negative equity scores 0)"},
        "interest_coverage": {"poor": 1.00, "good": 10.0, "text": "Operating income / interest expense"},
    }},
    "Cash flow": {"weight": 0.25, "indicators": {
        "fcf_margin":       {"poor": 0.00, "good": 0.20, "text": "Free cash flow / revenue"},
        "ocf_to_net_income": {"poor": 0.60, "good": 1.20, "text": "Operating cash flow / net income (only when net income > 0)"},
    }},
    "Growth": {"weight": 0.15, "indicators": {
        "revenue_growth":   {"poor": -0.05, "good": 0.15, "text": "Latest year revenue growth"},
        "revenue_cagr":     {"poor": -0.05, "good": 0.15, "text": "Revenue growth per year over up to 3 years"},
    }},
}

GRADE_BANDS = [(75, "Strong"), (55, "Moderate"), (35, "Weak"), (0, "Distressed")]
MIN_CATEGORIES = 3        # need at least this many categories with data, else no score


def rules_table():
    """The scoring rules as a table (useful to show in the notebook or the app)."""
    rows = []
    for category, spec in HEALTH_RULES.items():
        for name, rule in spec["indicators"].items():
            rows.append({"category": category, "category weight": spec["weight"], "indicator": name,
                         "0 points at": rule["poor"], "100 points at": rule["good"],
                         "definition": rule["text"]})
    return pd.DataFrame(rows)


def health_inputs(hist):
    """All indicator values from a fundamentals history that ends at the latest fiscal year."""
    hist = hist.sort_index()
    base = vp.compute_features(hist)
    cur = hist.iloc[-1]
    n = len(hist)
    out = {k: base.get(k, np.nan) for k in ("operating_margin", "net_margin", "roa", "current_ratio",
                                           "fcf_margin", "revenue_growth")}
    out["cash_ratio"] = vp._div(cur["cash"], cur["current_liabilities"])

    equity, debt = cur["equity"], cur["total_debt"]
    if np.isfinite(equity) and np.isfinite(debt):
        out["debt_to_equity"] = debt / equity if equity > 0 else (np.inf if debt > 0 else np.nan)
    else:
        out["debt_to_equity"] = np.nan

    interest = cur["interest_expense"]
    out["interest_coverage"] = (vp._div(cur["operating_income"], interest)
                                if np.isfinite(interest) and interest > 0 else np.nan)

    ni = cur["net_income"]
    out["ocf_to_net_income"] = (vp._div(cur["operating_cash_flow"], ni)
                                if np.isfinite(ni) and ni > 0 else np.nan)

    out["revenue_cagr"] = np.nan
    if n >= 2:
        m = min(3, n - 1)
        start = hist["revenue"].iloc[-1 - m]
        years = vp._years(hist.index[-1 - m], hist.index[-1])
        if np.isfinite(start) and start > 0 and np.isfinite(cur["revenue"]) and cur["revenue"] > 0 and years > 0:
            out["revenue_cagr"] = (cur["revenue"] / start) ** (1 / years) - 1
    return out


def score_indicator(value, poor, good):
    """Linear 0-100 score. Returns None when the value is missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    ratio = (value - poor) / (good - poor)
    return float(np.clip(ratio, 0.0, 1.0) * 100.0)


def grade_for(score):
    for threshold, label in GRADE_BANDS:
        if score >= threshold:
            return label
    return "Distressed"


def score_company(inputs):
    """Turn indicator values into category scores, an overall score and a grade."""
    indicator_rows, category_scores, weights = [], {}, {}
    for category, spec in HEALTH_RULES.items():
        scores = []
        for name, rule in spec["indicators"].items():
            value = inputs.get(name, np.nan)
            s = score_indicator(value, rule["poor"], rule["good"])
            indicator_rows.append({"category": category, "indicator": name,
                                   "value": value, "score": s})
            if s is not None:
                scores.append(s)
        if scores:
            category_scores[category] = float(np.mean(scores))
            weights[category] = spec["weight"]

    result = {"category_scores": category_scores,
              "indicators": pd.DataFrame(indicator_rows),
              "categories_used": len(category_scores)}
    if len(category_scores) < MIN_CATEGORIES:
        result.update(overall_score=None, grade="Insufficient data")
        return result
    total_weight = sum(weights.values())
    overall = sum(category_scores[c] * weights[c] for c in category_scores) / total_weight
    result.update(overall_score=float(overall), grade=grade_for(overall))
    return result


def health_report(hist):
    """Convenience wrapper: fundamentals history -> scored report."""
    report = score_company(health_inputs(hist))
    report["as_of_fiscal_year_end"] = pd.Timestamp(hist.index.max())
    return report