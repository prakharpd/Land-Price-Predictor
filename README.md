# Segment Forecast API  
### Quarterly Real Estate Price Prediction using Machine Learning & FastAPI

This project predicts **average real estate prices** for a given:

> **Locality • City • Property Type • Quarter • Year**

The system follows a hybrid logic:

- **Lookup Mode:** If the exact row exists in the dataset, return the real `Average Price`.
- **Forecast Mode:** If missing, predict using a trained ML model based only on past data.

---

## Project Structure

```bash
.
├── app.py                     # FastAPI backend (endpoints + prediction logic)
├── clean.csv                  # Historical dataset (segments + quarterly prices)
├── segment_forecast_model.pkl # Trained ML artifact (model + transformers + feature list)
├── requirements.txt           # Dependencies
└── V2.ipynb                   # Notebook (EDA + feature engineering + training workflow)
```

---

## Dataset

**File:** `clean.csv`  

Each row represents one quarterly real-estate segment.

### Key Columns
| Column | Description |
|--------|-------------|
| Locality | Area name |
| City | City name |
| Type | Property category |
| Months | Quarter (Jan-Mar, Apr-Jun, Jul-Sep, Oct-Dec) |
| Year | Year |
| Average Price | 🎯 Target variable |
| Min_Price | Minimum price |
| Max_Price | Maximum price |
| Q-o-Q | Quarter-over-quarter growth |

Internally, time is converted into `quarter_id` and `time_idx`.

---

## EDA Insights (Summary)

- Prices vary significantly by **City and Property Type**.
- Strong **seasonality across quarters**.
- `Q-o-Q` highlights volatility and growth/decline trends.
- Minimum historical data is required for forecasting.

---

## Target Variable

🎯 **Average Price**

- Returned directly from CSV if available.
- Predicted using ML if missing or future.

---

## Model & Metrics

- **Model:** RandomForestRegressor (scikit-learn pipeline)
- **Features:** Lag prices, rolling mean/std, seasonal averages, growth trends
- **Metrics:** **R²** and **RMSE** on original price scale

---

## API Architecture Flow

```text
Client (UI / Postman / Curl)
          |
          v
     POST /predict
          |
          v
Normalize inputs (Locality/City/Type/Months) + map Months → quarter_id
          |
          v
Filter dataset for (Locality, City, Type)
          |
          +------------------------------+
          |                              |
          v                              v
Exact row exists?                    Exact row missing?
(Year + Quarter match)               (past missing or future)
          |                              |
          v                              v
Return CSV Average Price             Build features from past-only history
(mode = "lookup")                    (lags, rolling stats, seasonality)
                                         |
                                         v
                                  Predict using pretrained ML model
                                  (mode = "forecast")
                                         |
                                         v
                               Return predicted Average Price + meta
```

---

## Run FastAPI (Git Bash / Windows)

### 1) Create & activate virtual environment
```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Start the server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Test API

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

### Prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"locality":"Gachibowli","city":"Hyderabad","prop_type":"Residential - Builder Floor Apartment","months":"Jan-Mar","year":2023}'
```

### Swagger UI
Open in browser:
```
http://127.0.0.1:8000/docs
```

---

## Tech Stack

- **FastAPI**, **Uvicorn**
- **Pandas**, **Joblib**
- **Scikit-learn**

---

## Disclaimer

> Few portions of this codebase were developed with the assistance of AI tools.  
> This was done to meet a **strict 2-day project deadline**.
