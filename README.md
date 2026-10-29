# Imari Health — Financial Health Navigator for Rwanda

**NISR 2026 Big Data Hackathon — Track 2: Financial Inclusion & Poverty Reduction**

## The Problem
Rwanda has achieved 96% financial inclusion. However, FinScope 2024 reports that only **10% of Rwandan adults are financially healthy**. Access is solved; usage and financial health are not.

## The Solution
Imari Health is a two-sided data tool:
1. **For Citizens:** A 14-question tool that uses an XGBoost model to score financial health (0-100) and gives one concrete, cited next action (e.g., "Join Mutuelle de Santé").
2. **For Policymakers:** A dashboard visualizing EICV7 2023/24 poverty and financial gaps by district, aligned with NFIR 2026-2030 targets.

## Data Sources
- **FinScope 2024** (AFR/NISR) — Financial health framework.
- **EICV7 Poverty Profile 2023/24** (NISR) — Poverty and financial gap statistics.

## Tech Stack
- **ML:** XGBoost + SHAP (explainability)
- **Backend:** FastAPI (Python)
- **Frontend:** React (via CDN), Tailwind CSS, Recharts

## How to Run Locally
```bash
# 1. Start the backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# 2. Start the frontend
cd ..
python -m http.server 5500 --directory frontend

# 3. Open http://localhost:5500/index.html