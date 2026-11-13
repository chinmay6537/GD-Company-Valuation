"""
valuation_predictor.py
======================
Prediction-time code for a future Python API.

    from valuation_predictor import load_bundle, analyze_company
    bundle = load_bundle("artifacts")
    result = analyze_company("AAPL", bundle)      # JSON-friendly dict

It re-uses exactly the same feature code (valuation_pipeline.py) that was used for training,
so training and prediction cannot drift apart.
"""

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import company_health as ch
import valuation_pipeline as vp

DISCLAIMER = ("The model estimates a typical MARKET CAPITALIZATION for a company with these "
              "fundamentals, based on the companies and years it was trained on. It is not an "
              "intrinsic (DCF) value, not a price target and not investment advice.")


def load_bundle(model_dir="artifacts"):
    """Load the fitted pipeline and its metadata."""
    model_dir = Path(model_dir)
    model_path, card_path = model_dir / "valuation_model.joblib", model_dir / "model_card.json"
    for path in (model_path, card_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}. Run the export cell of the notebook first.")
    with open(card_path) as handle:
        card = json.load(handle)
    return {"model": joblib.load(model_path), "card": card}


def to_jsonable(obj):
    """Recursively convert numpy/pandas objects to plain JSON types (NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def snapshot_from_raw(raw, adjust_shares_for_splits=True):
    """Features + current market cap for the LATEST available fiscal year."""
    fund = vp.fundamentals_table(raw)
    cur = fund.iloc[-1]
    if len(fund) < vp.MIN_HISTORY:
        raise vp.DataUnavailableError("need at least two fiscal years of statements")
    if not vp.core_data_ok(cur):
        raise vp.DataUnavailableError("latest statements lack revenue, total assets or share count")
    close = vp.close_series(raw["prices"])
    fye = fund.index[-1]
    factor = vp.split_factor_after(raw.get("splits"), fye) if adjust_shares_for_splits else 1.0
    shares_used = float(cur["shares"]) * factor
    price = float(close.iloc[-1])
    return {
        "fund": fund,
        "features": vp.compute_features(fund),
        "fiscal_year_end": fye,
        "price": price,
        "price_date": close.index[-1],
        "shares_used": shares_used,
        "market_cap": price * shares_used,
        "sector": (raw.get("info") or {}).get("sector") or "Unknown",
    }


def analyze_raw(raw, bundle):
    """Health analysis + valuation estimate from already-downloaded raw data."""
    card, model = bundle["card"], bundle["model"]
    numeric, categorical = card["numeric_features"], card["categorical_features"]
    snap = snapshot_from_raw(raw, card.get("adjust_shares_for_splits", True))

    row = {c: snap["features"].get(c, np.nan) for c in numeric}
    for c in categorical:
        row[c] = snap["sector"]
    X = pd.DataFrame([row], columns=numeric + categorical)
    missing = [c for c in numeric if not np.isfinite(row[c])]

    estimate = float(np.exp(model.predict(X)[0]))
    health = ch.health_report(snap["fund"])

    warnings = []
    if snap["sector"] not in card.get("training_sectors", []):
        warnings.append(f"Sector '{snap['sector']}' was not in the training data.")
    age_days = (pd.Timestamp(snap["price_date"]) - snap["fiscal_year_end"]).days
    if age_days > 540:
        warnings.append(f"Latest annual statements are {age_days} days old.")
    if missing:
        warnings.append("Missing inputs filled with training medians: " + ", ".join(missing))
    if raw["ticker"] not in card.get("training_tickers", []):
        warnings.append("This company was not in the training set; accuracy for unseen companies "
                        "is only indicated by the held-out test results, not guaranteed.")

    return to_jsonable({
        "status": "ok",
        "ticker": raw["ticker"],
        "name": (raw.get("info") or {}).get("longName") or (raw.get("info") or {}).get("shortName"),
        "sector": snap["sector"],
        "in_training_universe": raw["ticker"] in card.get("training_tickers", []),
        "statements_as_of": snap["fiscal_year_end"],
        "price": snap["price"], "price_date": snap["price_date"],
        "current_market_cap_usd": snap["market_cap"],
        "model_estimated_market_cap_usd": estimate,
        "estimate_to_market_ratio": estimate / snap["market_cap"],
        "typical_error": {
            "median_abs_pct_error_on_test": card.get("test_metrics", {}).get("median abs % error"),
            "note": "Median absolute percentage error of the chosen model on a held-out later time period.",
        },
        "health": {
            "overall_score": health["overall_score"], "grade": health["grade"],
            "category_scores": health["category_scores"],
            "indicators": health["indicators"],
        },
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    })


def analyze_company(ticker, bundle, cache_dir="data_cache", refresh=False):
    """Search-by-ticker entry point. Never raises for data problems: returns a status instead."""
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return {"status": "error", "message": "Please provide a ticker symbol."}
    try:
        raw = vp.fetch_raw(ticker, cache_dir=cache_dir, refresh=refresh)
    except vp.DataUnavailableError as exc:
        return {"status": "unavailable", "ticker": ticker, "message": str(exc)}
    ok, message = vp.check_currency(raw)
    if not ok:
        return {"status": "unsupported", "ticker": ticker,
                "message": f"Not supported: company {message}."}
    try:
        return analyze_raw(raw, bundle)
    except vp.DataUnavailableError as exc:
        return {"status": "unavailable", "ticker": ticker, "message": str(exc)}