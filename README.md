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
- **MAE:** ~ A$2.5K–A$4K  
- **RMSE:** ~ A$5K–A$15K  
- **MAPE:** ~ 8%-14%  
- **R²:** ~ 0.8-0.9  

✅ The model achieves ~**0.86 R² on average**, indicating strong pricing predictive power for pre-owned cars.


<img width="1048" height="445" alt="image" src="https://github.com/user-attachments/assets/75c91c07-18d4-4ca1-b461-3c21e6341f15" />


<img width="1051" height="447" alt="Screenshot 2026-01-26 235222" src="https://github.com/user-attachments/assets/8ab303f2-a39f-4bb4-a72c-8a127a9649cc" />


