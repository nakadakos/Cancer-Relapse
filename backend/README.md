# OncoRelapse — Backend

FastAPI server that exposes the trained ML model as a REST API.

---

## Getting Started

### Windows (easiest)

Double-click **`start_backend.bat`**.  
It will automatically:
1. Create a Python virtual environment (`venv/`) if one does not exist.
2. Install all dependencies from `requirements.txt`.
3. Start the FastAPI server on **http://localhost:8000**.

### Manual (any OS)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
uvicorn main:app --reload
```

---

## Running the Data Pipeline

The model file (`models/model_pipeline.pkl`) must be generated before the server will return predictions. Run these scripts once, in order:

```bash
# 1. Download raw datasets from UCI ML Repository
python download_data.py

# 2. Harmonise and combine all datasets into one CSV
python preprocess_data.py

# 3. Train, tune, and evaluate models; save the best one
python train_model.py
```

Training typically takes **5–15 minutes** depending on your hardware.  
Set `TRAIN_N_JOBS=4` (or any integer) to use more CPU cores during training:

```bash
set TRAIN_N_JOBS=4 && python train_model.py   # Windows
TRAIN_N_JOBS=4 python train_model.py          # macOS / Linux
```

---

## Project Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application and all API endpoints |
| `download_data.py` | Downloads raw datasets from UCI ML Repository |
| `preprocess_data.py` | Harmonises datasets into a unified schema |
| `train_model.py` | Trains and tunes ML models (SMOTE + RandomizedSearchCV) |
| `requirements.txt` | Python dependencies |
| `start_backend.bat` | One-click startup script for Windows |

### Generated directories (not committed)

| Path | Contents |
|------|---------|
| `data/raw/` | Raw downloaded CSVs |
| `data/processed/` | `combined_cancer_data.csv` |
| `models/` | `model_pipeline.pkl`, `evaluation_results.json`, `feature_importances.json` |
| `venv/` | Python virtual environment |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Status message |
| `GET` | `/health` | Readiness probe — returns 503 if model not loaded |
| `POST` | `/predict` | Predict relapse risk (see schema at `/docs`) |
| `GET` | `/model-info` | Model name, metrics, feature importances |
| `GET` | `/visualizations` | Aggregated chart data for the dashboard |

Full interactive docs available at **http://localhost:8000/docs** while the server is running.

---

## Notes

- `venv/` and `venv.rar` should **not** be committed. Virtual environments are machine-specific.
- If you add a new Python package, add it to `requirements.txt` so collaborators pick it up automatically.
- The model is selected by **recall** (not accuracy) because in a medical context, missing a true relapse (false negative) is more dangerous than a false positive.
