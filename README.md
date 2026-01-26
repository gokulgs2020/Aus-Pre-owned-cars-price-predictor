# Australian market Preowned Car Price Estimation

This project predicts preowned car prices using a **retention-based model**:
1) Estimate an approximate **New Price** per Brand/Model from near-new vehicles  
2) Learn **retention = Price / New_Price** as a function of kilometres, age, and vehicle attributes  
3) Convert predicted retention back into price

## Project Structure
- `src/` training + feature pipeline
- `models/` saved model + lookup tables
- `streamlit_app.py` Streamlit UI

## Setup (Local)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt

```
## ✅ Model Performance (5-Fold GroupKFold CV)


Evaluation is done using **GroupKFold** split by **Brand + Model** to reduce leakage.

**Cross-validation results (Price prediction):**
- **MAE:** ~ A$2.5K–A$4K  
- **RMSE:** ~ A$5K–A$15K  
- **MAPE:** ~ 8%-14%  
- **R²:** ~ 0.8-0.9  

✅ The model achieves ~**0.86 R² on average**, indicating strong pricing predictive power for pre-owned cars.

## 📸 Screenshots

### Sample Prediction Validation (Actual vs Predicted)
![Actual vs Predicted 1](https://raw.githubusercontent.com/gokulgs2020/Aus-Pre-owned-cars-price-predictor/main/assets/screenshot_1.png)
![Actual vs Predicted 2](https://raw.githubusercontent.com/gokulgs2020/Aus-Pre-owned-cars-price-predictor/main/assets/screenshot_2.png)

