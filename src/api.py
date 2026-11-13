"""FastAPI sketch from notebook section 5.3 (not tested).
Run from src/ after the notebook has created artifacts/:  uvicorn api:app --reload
"""
import pandas as pd
from fastapi import FastAPI, HTTPException
from valuation_predictor import load_bundle, analyze_company

app = FastAPI(title="AI Company Health & Valuation API")
bundle = load_bundle("artifacts")
supported = pd.read_csv("artifacts/supported_companies.csv")

@app.get("/companies")
def companies(q: str = ""):
    """Search the companies the model was trained on."""
    hits = supported[supported["ticker"].str.contains(q.upper(), na=False)]
    return hits.to_dict(orient="records")

@app.get("/analyze/{ticker}")
def analyze(ticker: str):
    result = analyze_company(ticker, bundle)
    if result["status"] != "ok":
        raise HTTPException(status_code=404, detail=result.get("message", "no data"))
    return result
