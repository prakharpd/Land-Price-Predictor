from __future__ import annotations

import os
import re
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# =============================
# Config (override via env vars)
# =============================
MODEL_PATH = os.getenv("MODEL_PATH", "./segment_forecast_model.pkl")
CSV_PATH = os.getenv("CSV_PATH", "./clean.csv")
MIN_HISTORY = int(os.getenv("MIN_HISTORY", "5"))

app = FastAPI(
    title="Segment Forecast API",
    version="1.1.0",
    description="Unified Lookup (CSV) -> Predict (Model) API with dropdown endpoints.",
)

_WS_RE = re.compile(r"\s+")


def norm(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = _WS_RE.sub(" ", s)
    return s.lower()


def require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(f"{label} not found at: {os.path.abspath(path)}")


# =============================
# Load model + dataset (once)
# =============================
require_file(MODEL_PATH, "Model artifact")
require_file(CSV_PATH, "CSV dataset")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
pt = artifact["target_transformer"]
feature_cols = artifact["feature_cols"]

quarter_map = artifact.get("quarter_map", {}).copy()
# Add common aliases (safe)
quarter_map.update(
    {
        "April-June": 2,
        "Apr-Jun": 2,
        "Apr - Jun": 2,
        "April - June": 2,
    }
)
quarter_map_n = {norm(k): v for k, v in quarter_map.items()}

df = pd.read_csv(CSV_PATH)

required_cols = ["Locality", "City", "Type", "Months", "Year", "Average Price"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise RuntimeError(f"CSV missing required columns: {missing_cols}")

df["Locality_n"] = df["Locality"].apply(norm)
df["City_n"] = df["City"].apply(norm)
df["Type_n"] = df["Type"].apply(norm)
df["Months_n"] = df["Months"].apply(norm)

df["quarter_id"] = df["Months_n"].map(quarter_map_n)

for col in ["Year", "Average Price", "Min_Price", "Max_Price", "Q-o-Q"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["time_idx"] = df["Year"] * 4 + df["quarter_id"]
df = df.sort_values(["Locality_n", "City_n", "Type_n", "time_idx"]).reset_index(drop=True)


def predict_unified(
    locality: str,
    city: str,
    prop_type: str,
    months: str,
    year: int,
    min_history: int = MIN_HISTORY,
) -> Dict[str, Any]:
    locality_n = norm(locality)
    city_n = norm(city)
    type_n = norm(prop_type)
    months_n = norm(months)

    qid = quarter_map_n.get(months_n)
    if qid is None:
        raise ValueError("Invalid months. Use: Jan-Mar, Apr-Jun, Jul-Sep, Oct-Dec")

    year = int(year)
    req_t = year * 4 + int(qid)

    seg = df[
        (df["Locality_n"] == locality_n)
        & (df["City_n"] == city_n)
        & (df["Type_n"] == type_n)
    ].copy()

    if seg.empty:
        candidates = (
            df[df["City_n"] == city_n][["Locality", "City", "Type"]]
            .drop_duplicates()
            .head(15)
        )
        raise ValueError(
            "Segment not found. Check spelling/case.\n"
            f"Some candidates in the same city:\n{candidates.to_string(index=False)}"
        )

    seg = seg.dropna(subset=["quarter_id"]).sort_values("time_idx")

    # RULE 1: exact row exists -> lookup ALWAYS
    exact = seg[(seg["Year"] == year) & (seg["quarter_id"] == int(qid))]
    if not exact.empty and pd.notna(exact["Average Price"].iloc[0]):
        return {
            "mode": "lookup",
            "average_price": float(exact["Average Price"].iloc[0]),
            "meta": {"year": year, "quarter_id": int(qid)},
        }

    # RULE 2: otherwise -> predict using ONLY past
    past = seg[seg["time_idx"] < req_t].copy()
    if len(past) < min_history:
        raise ValueError(f"Not enough past data (need >= {min_history}, have {len(past)}).")

    needed = ["Average Price", "Min_Price", "Max_Price", "Q-o-Q"]
    missing_needed = [c for c in needed if c not in past.columns]
    if missing_needed:
        raise ValueError(f"CSV missing required numeric columns for prediction: {missing_needed}")

    past_valid = past.dropna(subset=needed).copy()
    if past_valid.empty:
        raise ValueError("Past history exists but Avg/Min/Max/QoQ fields are missing (NaN).")

    last = past_valid.iloc[-1]
    recent = past_valid.tail(4)
    same_q = past_valid[past_valid["quarter_id"] == int(qid)]

    row = {
        "Locality": locality,
        "City": city,
        "Type": prop_type,
        "quarter_id": int(qid),
        "Year": year,
        "time_idx": int(req_t),
        "quarter_id": int(qid),
        "Year": year,
        "time_idx": int(req_t),
        "lag_avg_1": float(last["Average Price"]),
        "lag_min_1": float(last["Min_Price"]),
        "lag_max_1": float(last["Max_Price"]),
        "lag_qoq_1": float(last["Q-o-Q"]),
        "roll_avg_mean_4": float(recent["Average Price"].mean()),
        "roll_avg_std_4": float(recent["Average Price"].std(ddof=1)) if len(recent) > 1 else 0.0,
        "roll_qoq_mean_4": float(recent["Q-o-Q"].mean()),
        "seasonal_avg_qtr": float(same_q["Average Price"].mean())
        if not same_q.empty
        else float(past_valid["Average Price"].mean()),
        "seasonal_qoq_qtr": float(same_q["Q-o-Q"].mean())
        if not same_q.empty
        else float(past_valid["Q-o-Q"].mean()),
        "history_len": float(len(past_valid)),
    }

    X = pd.DataFrame([row])

    missing_feats = [c for c in feature_cols if c not in X.columns]
    if missing_feats:
        raise ValueError(f"Feature mismatch vs trained model. Missing: {missing_feats}")

    X = X[feature_cols]
    pred_t = model.predict(X)[0]
    pred_price = float(pt.inverse_transform([[pred_t]])[0][0])

    mode = "past-missing-predict" if req_t <= float(seg["time_idx"].max()) else "future-forecast"
    return {"mode": mode, "average_price": pred_price, "meta": {"year": year, "quarter_id": int(qid)}}


class PredictRequest(BaseModel):
    locality: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    prop_type: str = Field(..., min_length=1)
    months: str = Field(..., min_length=1)
    year: int = Field(..., ge=1900, le=2200)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/meta")
def meta() -> Dict[str, Any]:
    return {
        "months_allowed": sorted(set(quarter_map.keys())),
        "min_history": MIN_HISTORY,
        "feature_count": len(feature_cols),
    }


@app.get("/options/cities")
def options_cities() -> Dict[str, List[str]]:
    return {"cities": sorted(df["City"].dropna().astype(str).unique().tolist())}


@app.get("/options/localities")
def options_localities(city: str = Query(..., min_length=1)) -> Dict[str, Any]:
    d = df[df["City_n"] == norm(city)]
    return {"city": city, "localities": sorted(d["Locality"].dropna().astype(str).unique().tolist())}


@app.get("/options/types")
def options_types(city: str = Query(..., min_length=1), locality: str = Query(..., min_length=1)) -> Dict[str, Any]:
    d = df[(df["City_n"] == norm(city)) & (df["Locality_n"] == norm(locality))]
    return {"city": city, "locality": locality, "types": sorted(d["Type"].dropna().astype(str).unique().tolist())}


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    try:
        return predict_unified(req.locality, req.city, req.prop_type, req.months, req.year, MIN_HISTORY)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
