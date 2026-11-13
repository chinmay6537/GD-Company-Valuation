"""
valuation_pipeline.py
=====================
Reusable data + feature code for the AI Company Health & Valuation Predictor.

This file is written by the notebook (cell "%%writefile valuation_pipeline.py") and is
imported both by the notebook and, later, by a Python API. It contains NO model training.

Design rules (see the notebook for the full explanation)
--------------------------------------------------------
* Every feature for a prediction date is computed from financial statements whose
  fiscal year ended at least REPORT_LAG_DAYS before that date (point-in-time).
* Prices are used ONLY to build the target (market capitalization), never as features.
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
REPORT_LAG_DAYS = 90        # assume annual statements become public 90 days after year end
MAX_PRICE_GAP_DAYS = 10     # skip a row if no trading day exists within 10 days of the date
MIN_HISTORY = 2             # need the current and the previous fiscal year

BASE_NUMERIC = [
    "log_revenue", "log_total_assets",
    "gross_margin", "operating_margin", "net_margin", "fcf_margin", "roa",
    "current_ratio", "cash_to_assets", "debt_to_assets", "liabilities_to_assets",
    "revenue_growth",
]
CATEGORICAL = ["sector"]

# yfinance row names differ between companies, so each item has a list of candidates.
SOURCES = {
    "revenue":            ("income",   ["Total Revenue", "Operating Revenue"]),
    "gross_profit":       ("income",   ["Gross Profit"]),
    "operating_income":   ("income",   ["Operating Income", "Total Operating Income As Reported"]),
    "net_income":         ("income",   ["Net Income", "Net Income Common Stockholders"]),
    "interest_expense":   ("income",   ["Interest Expense", "Interest Expense Non Operating"]),
    "total_assets":       ("balance",  ["Total Assets"]),
    "total_liabilities":  ("balance",  ["Total Liabilities Net Minority Interest"]),
    "equity":             ("balance",  ["Stockholders Equity", "Common Stock Equity",
                                        "Total Equity Gross Minority Interest"]),
    "current_assets":     ("balance",  ["Current Assets"]),
    "current_liabilities": ("balance", ["Current Liabilities"]),
    "cash":               ("balance",  ["Cash Cash Equivalents And Short Term Investments",
                                        "Cash And Cash Equivalents"]),
    "total_debt":         ("balance",  ["Total Debt"]),
    "shares":             ("balance",  ["Ordinary Shares Number", "Share Issued"]),
    "operating_cash_flow": ("cashflow", ["Operating Cash Flow",
                                         "Cash Flow From Continuing Operating Activities"]),
    "free_cash_flow":     ("cashflow", ["Free Cash Flow"]),
    "capex":              ("cashflow", ["Capital Expenditure"]),
}


class DataUnavailableError(Exception):
    """Raised when usable data for a company cannot be obtained."""


# --------------------------------------------------------------------------------------
# 1. Downloading and caching raw data
# --------------------------------------------------------------------------------------
def _validate_raw(raw, ticker):
    if raw["prices"] is None or len(raw["prices"]) == 0 or "Close" not in raw["prices"]:
        raise DataUnavailableError(f"{ticker}: no price history returned")
    for key in ("income", "balance"):
        frame = raw[key]
        if frame is None or len(frame) == 0 or frame.dropna(how="all").shape[1] == 0:
            raise DataUnavailableError(f"{ticker}: empty {key} statement")


def fetch_raw(ticker, cache_dir="data_cache", refresh=False, retries=3, pause=1.0):
    """Download prices, statements, splits and basic info for one ticker (cached to disk)."""
    import yfinance as yf

    ticker = ticker.upper().strip()
    cache_path = Path(cache_dir) / f"{ticker}.pkl"
    if cache_path.exists() and not refresh:
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            tk = yf.Ticker(ticker)
            prices = tk.history(period="max", auto_adjust=False)
            income, balance, cashflow = tk.income_stmt, tk.balance_sheet, tk.cashflow
            try:
                splits = tk.splits
            except Exception:
                splits = pd.Series(dtype=float)
            try:
                info = tk.info or {}
            except Exception:
                info = {}
            keep = ("sector", "industry", "currency", "financialCurrency",
                    "marketCap", "shortName", "longName", "quoteType")
            raw = {
                "ticker": ticker,
                "prices": prices,
                "income": income,
                "balance": balance,
                "cashflow": cashflow,
                "splits": splits,
                "info": {k: info.get(k) for k in keep},
                "downloaded_at": pd.Timestamp.now().isoformat(),
            }
            _validate_raw(raw, ticker)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as handle:
                pickle.dump(raw, handle)
            return raw
        except Exception as exc:                       # network, rate limit, empty data ...
            last_error = exc
            time.sleep(pause * attempt)
    raise DataUnavailableError(f"{ticker}: download failed after {retries} attempts ({last_error})")


# --------------------------------------------------------------------------------------
# 2. Cleaning: statements -> one tidy table indexed by fiscal year end
# --------------------------------------------------------------------------------------
def _normalize_statement(df):
    """Rows = line items, columns = fiscal-year-end dates (ascending, timezone-free)."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    out = out[~out.index.duplicated(keep="first")]
    out.columns = pd.DatetimeIndex(pd.to_datetime(out.columns)).tz_localize(None)
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna(axis=1, how="all")
    return out.sort_index(axis=1)


def _pick(frame, names):
    for name in names:
        if name in frame.index and frame.loc[name].notna().any():
            return frame.loc[name].astype(float)
    return pd.Series(np.nan, index=frame.columns, dtype=float)


def fundamentals_table(raw):
    """One row per fiscal year end, one column per financial item (all in raw USD)."""
    frames = {
        "income": _normalize_statement(raw["income"]),
        "balance": _normalize_statement(raw["balance"]),
        "cashflow": _normalize_statement(raw["cashflow"]),
    }
    dates = sorted(set().union(*[set(f.columns) for f in frames.values() if len(f.columns)]))
    if not dates:
        raise DataUnavailableError(f"{raw.get('ticker', '?')}: statements contain no dates")
    table = pd.DataFrame(index=pd.DatetimeIndex(dates))
    for key, (statement, names) in SOURCES.items():
        frame = frames[statement]
        table[key] = _pick(frame, names).reindex(table.index) if len(frame) else np.nan
    # Free cash flow fallback: operating cash flow minus capital expenditure
    fallback = table["operating_cash_flow"] - table["capex"].abs()
    table["free_cash_flow"] = table["free_cash_flow"].fillna(fallback)
    table["interest_expense"] = table["interest_expense"].abs()
    table = table.dropna(subset=["revenue", "total_assets"], how="all").sort_index()
    return table


def close_series(prices):
    """Split-adjusted (not dividend-adjusted) closing prices with a timezone-free date index."""
    s = prices["Close"].dropna().astype(float)
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s[s > 0]


def split_factor_after(splits, date):
    """Product of stock-split ratios that happened AFTER `date`.

    Yahoo prices are adjusted to today's share units, while balance-sheet share counts are
    reported in the units of their own date. Multiplying the old share count by this factor
    puts both in the same units. (Verify with market_cap_consistency below.)
    """
    if splits is None or len(splits) == 0:
        return 1.0
    idx = pd.DatetimeIndex(splits.index).tz_localize(None)
    ratios = pd.Series(np.asarray(splits.values, dtype=float), index=idx)
    ratios = ratios[(ratios > 0) & (ratios.index > date)]
    return float(np.prod(ratios.values)) if len(ratios) else 1.0


def check_currency(raw):
    """Return (ok, message). We only support companies that report in the currency they trade in (USD)."""
    info = raw.get("info", {}) or {}
    reporting, trading = info.get("financialCurrency"), info.get("currency")
    if reporting and reporting != "USD":
        return False, f"reports in {reporting}, not USD"
    if trading and trading != "USD":
        return False, f"trades in {trading}, not USD"
    return True, "ok" if reporting else "currency not stated (assumed USD)"


def market_cap_consistency(raw, adjust_shares_for_splits=True):
    """Ratio of (latest close x latest reported shares) to Yahoo's current marketCap.

    A value far from 1 (say below 0.5 or above 2) usually means a share-count/split unit problem.
    Returns NaN when it cannot be computed.
    """
    try:
        fund = fundamentals_table(raw)
        close = close_series(raw["prices"])
        shares = fund["shares"].dropna()
        reported = raw["info"].get("marketCap")
        if shares.empty or not reported:
            return np.nan
        fye = shares.index[-1]
        factor = split_factor_after(raw.get("splits"), fye) if adjust_shares_for_splits else 1.0
        return float(close.iloc[-1] * shares.iloc[-1] * factor / reported)
    except Exception:
        return np.nan


def max_share_jump(raw, adjust_shares_for_splits=True):
    """Largest year-to-year change (as a multiple, >= 1) in split-adjusted reported share counts.

    A jump of ~10x means a stock split was missed or counted twice. Normal buybacks/issuance
    stay well below 3x, so values above 3 are treated as a data problem.
    """
    try:
        shares = fundamentals_table(raw)["shares"].dropna()
        if len(shares) < 2:
            return np.nan
        factors = np.array([split_factor_after(raw.get("splits"), d) if adjust_shares_for_splits else 1.0
                            for d in shares.index])
        adjusted = shares.values * factors
        ratios = adjusted[1:] / adjusted[:-1]
        return float(np.max(np.maximum(ratios, 1.0 / ratios)))
    except Exception:
        return np.nan


# --------------------------------------------------------------------------------------
# 3. Features (point-in-time: they use ONLY the rows of `hist`, i.e. the past)
# --------------------------------------------------------------------------------------
def _div(a, b):
    try:
        if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
            return np.nan
    except TypeError:
        return np.nan
    return float(a) / float(b)


def _log(v):
    return float(np.log(v)) if np.isfinite(v) and v > 0 else np.nan


def _years(d0, d1):
    return (d1 - d0).days / 365.25


def _trapz(y, x):
    trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapezoid(y, x))


def compute_features(hist):
    """Compute all model features from a fundamentals history that ENDS at the current fiscal year.

    `hist` is a slice of fundamentals_table(): rows are fiscal years up to (and including) the
    one being scored. Nothing after the last row can influence the result.
    """
    hist = hist.sort_index()
    n = len(hist)
    cur = hist.iloc[-1]
    f = {}

    rev, assets = cur["revenue"], cur["total_assets"]
    # ----- Core financial features -----
    f["log_revenue"] = _log(rev)
    f["log_total_assets"] = _log(assets)
    f["gross_margin"] = _div(cur["gross_profit"], rev)
    f["operating_margin"] = _div(cur["operating_income"], rev)
    f["net_margin"] = _div(cur["net_income"], rev)
    f["fcf_margin"] = _div(cur["free_cash_flow"], rev)
    f["roa"] = _div(cur["net_income"], assets)
    f["current_ratio"] = _div(cur["current_assets"], cur["current_liabilities"])
    f["cash_to_assets"] = _div(cur["cash"], assets)
    f["debt_to_assets"] = _div(cur["total_debt"], assets)
    f["liabilities_to_assets"] = _div(cur["total_liabilities"], assets)
    f["revenue_growth"] = np.nan
    if n >= 2:
        prev_rev = hist["revenue"].iloc[-2]
        f["revenue_growth"] = _div(rev - prev_rev, prev_rev)

    return f


def core_data_ok(cur):
    """A row is usable only if the basics needed for the target and main features exist."""
    return all(np.isfinite(cur[k]) and cur[k] > 0 for k in ("revenue", "total_assets", "shares"))


def build_company_rows(raw, ticker=None, report_lag_days=REPORT_LAG_DAYS,
                       adjust_shares_for_splits=True, fund=None):
    """Build one training row per fiscal year for a company.

    Row k describes the situation on `pred_date` = first trading day on/after
    (fiscal year end + report_lag_days). Features use fundamentals up to fiscal year k only.
    The target columns (market_cap, enterprise_value) use the price on pred_date.
    """
    ticker = ticker or raw.get("ticker", "?")
    fund = fundamentals_table(raw) if fund is None else fund
    close = close_series(raw["prices"])
    sector = (raw.get("info") or {}).get("sector") or "Unknown"
    splits = raw.get("splits")

    rows = []
    for k in range(MIN_HISTORY - 1, len(fund)):
        hist = fund.iloc[: k + 1]
        cur = hist.iloc[-1]
        fye = hist.index[-1]
        if not core_data_ok(cur):
            continue
        available = fye + pd.Timedelta(days=report_lag_days)
        pos = close.index.searchsorted(available)
        if pos >= len(close):
            continue                                   # statements not yet "public" in our price data
        pred_date = close.index[pos]
        if (pred_date - available).days > MAX_PRICE_GAP_DAYS:
            continue
        factor = split_factor_after(splits, fye) if adjust_shares_for_splits else 1.0
        shares_used = float(cur["shares"]) * factor
        price = float(close.iloc[pos])
        market_cap = price * shares_used
        debt, cash = cur["total_debt"], cur["cash"]
        enterprise_value = market_cap + debt - cash if np.isfinite(debt) and np.isfinite(cash) else np.nan

        row = {
            "ticker": ticker, "sector": sector,
            "fiscal_year_end": fye, "available_date": available, "pred_date": pred_date,
            "n_history": k + 1, "price": price, "shares_used": shares_used,
            "market_cap": market_cap, "enterprise_value": enterprise_value,
            "total_debt": debt, "cash": cash,
        }
        row.update(compute_features(hist))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 4. Preprocessing (fitted ONLY on training data inside a scikit-learn Pipeline)
# --------------------------------------------------------------------------------------
class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip every column to the [lower, upper] quantiles learned from the training data."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def make_preprocessor(numeric_cols, categorical_cols=("sector",), scale=True):
    """Impute (median) -> clip outliers -> scale numeric columns; one-hot encode categories."""
    steps = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clip", QuantileClipper(0.01, 0.99)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(steps), list(numeric_cols))]
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical_cols)))
    return ColumnTransformer(transformers, remainder="drop")