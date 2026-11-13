# Learning Company Valuation Weights with Gradient Descent

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chinmay6537/GD-Company-Valuation/blob/main/AI_Company_Health_Valuation_Predictor.ipynb)

This is a CS2227 AI and ML project (Semester III, B.Tech (Hons.), School of Computer Science and Engineering, RV University).

The project estimates a listed company's market capitalization from its annual financial statements. The weight given to each financial ratio comes from minimizing a ridge-regularized squared-error objective. That objective is solved with gradient descent written from scratch in NumPy. The project also includes a rule-based company health score, which is kept separate from the model.

## Team

| Name | USN |
|---|---|
| Deshnaa Suresh | 1RUA25CSE0150 |
| Dindukurti Varshini | 1RUA25CSE0161 |
| Asmita Guptha | 1RUA25CSE0093 |
| Chinmay S | 1RUA25CSE0141 |

## Repository contents

| Path | What it is |
|---|---|
| `AI_Company_Health_Valuation_Predictor.ipynb` | The full project notebook: dataset, models, gradient-descent study, health score and exported artifacts |
| `docs/Project_Report.docx` | Project report |
| `docs/IEEE_Paper.docx` | IEEE-format paper |
| `src/valuation_pipeline.py` | Data download, cleaning, point-in-time features and preprocessing |
| `src/company_health.py` | Rule-based financial health score (0–100) |
| `src/valuation_predictor.py` | Loads the saved model and analyses a ticker |
| `src/api.py` | FastAPI sketch from notebook section 5.3 (not tested) |

The notebook writes the `src/` modules itself with `%%writefile`, so it runs on its own. The copies in `src/` are there so the modules can be read and reused without opening the notebook.

## Method

- **Data:** 117 company-years from 40 large US companies, with prediction dates from December 2023 to October 2026, taken from Yahoo Finance statements. There are 12 point-in-time ratios (size, margins, return on assets, liquidity, leverage, revenue growth) plus a one-hot sector, which gives 19 inputs. The target is ln(market cap).
- **No leakage:** features use only statements that were already public, with a 90-day reporting lag. Prices are never used as inputs. Rows are split by time: train 60 %, validation 20 %, test 20 %.
- **Optimization:** the gradient and Hessian are derived by hand and checked numerically. The objective is convex, and the step size is bounded by η < 2/L. The project implements plain gradient descent, momentum, mini-batch SGD, and projected gradient descent for the constrained form ‖w‖ ≤ c.

## Results (held-out test period)

| Model | RMSE (log) | R² | Median abs. error |
|---|---|---|---|
| Gradient descent (ours) | 0.555 | 0.810 | 22.2 % |
| Closed-form ridge | 0.555 | 0.810 | 22.2 % |
| Revenue-multiple baseline | 0.729 | 0.673 | 47.9 % |
| Training-mean baseline | 1.277 | −0.003 | 53.1 % |

- Momentum needed 48 iterations to converge and plain gradient descent needed 659 (κ ≈ 91).
- The gradient-descent weights match the closed-form solution to within 8.4 × 10⁻¹¹.

**Limitations:** the model is linear, the sample is small and uses annual data only, and the test period covers only 8 rows. The estimate is a typical market value for similar companies. It is not an intrinsic value and it is not investment advice.

## How to run

**Google Colab (recommended):** click the badge above, then choose *Runtime → Run all*. The first run needs internet access to download data from Yahoo Finance and takes a few minutes.

**Locally:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook AI_Company_Health_Valuation_Predictor.ipynb
```

Running the notebook creates `data_cache/`, `artifacts/` (the trained model and model card) and `report_figures/`. Git ignores all three.

**Optional API:** after the notebook has run, copy `artifacts/` into `src/`, then run `cd src && uvicorn api:app --reload`.
