# Australian market Preowned Car Price Estimation

This project predicts preowned car prices using a **retention-based model**:
1) Estimate an approximate **New Price** per Brand/Model from near-new vehicles  
2) Learn **retention = Price / New_Price** as a function of kilometres, age, and vehicle attributes  
3) Convert predicted retention back into price

## Project Structure
- `src/` training + feature pipeline
- `models/` saved model + lookup tables
- `streamlit_app.py` Streamlit UI

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt


## ✅ Model Performance (5-Fold GroupKFold CV)

Evaluation is done using **GroupKFold** split by **Brand + Model** to reduce leakage.

**Cross-validation results (Price prediction):**
- **MAE:** ~ A$8K–A$12K  
- **RMSE:** ~ A$10K–A$18K  
- **MAPE:** ~ 23%–34%  
- **R²:** ~ 0.62–0.75  

✅ The model achieves ~**0.70 R² on average**, indicating strong predictive power for used car pricing.
