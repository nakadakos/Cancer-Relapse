# OncoRelapse Predictor

An AI-powered cancer relapse risk assessment tool built with a Python/FastAPI backend and a React/Vite frontend.

---

## Project Structure

```
Cancer-relapse/
├── backend/          # Python FastAPI ML server
└── frontend/         # React + Vite web application
```

---

## Quick Start

### 1. Backend

```bash
cd backend

# Option A — Windows (recommended)
# Double-click start_backend.bat
# It will auto-create the venv, install dependencies, and start the server.

# Option B — manual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
uvicorn main:app --reload
```

The API will be available at **http://localhost:8000**.  
Interactive API docs: **http://localhost:8000/docs**

> **First-time setup:** Before starting the server you need a trained model.
> Run the data pipeline once:
> ```bash
> python download_data.py      # downloads raw datasets
> python preprocess_data.py    # harmonises into combined_cancer_data.csv
> python train_model.py        # trains models, saves model_pipeline.pkl
> ```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

The app will open at **http://localhost:5173**.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML / Data | scikit-learn, XGBoost, imbalanced-learn (SMOTE), pandas, numpy |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Frontend | React 19, Vite, Recharts |
| Styling | Vanilla CSS (dark mode, glassmorphism) |

---

## Cancer Types Supported

Breast · Lung · Colon · Prostate · Liver · Mouth · Thyroid

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check message |
| `GET` | `/health` | Readiness probe (503 if model not loaded) |
| `POST` | `/predict` | Predict relapse risk from patient data |
| `GET` | `/model-info` | Current model name and metrics |
| `GET` | `/visualizations` | Aggregated chart data for dashboard |

---

## Medical Disclaimer

This tool is for **educational and research purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment.
