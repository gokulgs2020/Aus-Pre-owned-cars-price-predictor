import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMRegressor

from .config import (
    RANDOM_STATE, NUM_COLS, CAT_COLS, MODEL_GROUP_COL,
    MAX_PRICE, MAX_KM, MIN_YEAR
)
from .data_prep import clean_raw
from .features import apply_filters, build_new_price_lookups, add_features
from .utils import ensure_dir

def price_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-9)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ],
        remainder="drop"
    )

    model = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE
    )

    return Pipeline([("pre", pre), ("model", model)])

def train(csv_path: str, out_dir: str = "models") -> None:
    ensure_dir(out_dir)

    # ---- load + clean ----
    df_raw = pd.read_csv(csv_path)
    df = clean_raw(df_raw)
    df = apply_filters(df, max_price=MAX_PRICE, max_km=MAX_KM, min_year=MIN_YEAR)

    # ---- build lookups + features ----
    bm_lookup, b_lookup = build_new_price_lookups(df)
    df_feat = add_features(df, bm_lookup, b_lookup)

    X = df_feat[NUM_COLS + CAT_COLS].copy()
    y = df_feat["y"].copy()
    groups = df_feat[MODEL_GROUP_COL].copy()

    pipe = build_pipeline()

    # ---- CV evaluation (GroupKFold) ----
    gkf = GroupKFold(n_splits=5)
    fold_scores = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred_log_ret = pipe.predict(X.iloc[te])

        pred_ret = np.exp(pred_log_ret)
        pred_price = pred_ret * df_feat.iloc[te]["New_Price"].values
        true_price = df_feat.iloc[te]["Price"].values

        scores = price_metrics(true_price, pred_price)
        scores["fold"] = fold
        fold_scores.append(scores)

    scores_df = pd.DataFrame(fold_scores)

    print("\nCV metrics per fold (price):")
    print(scores_df[["fold", "MAE", "RMSE", "MAPE", "R2"]])

    print("\nAverage CV metrics (price):")
    print(scores_df[["MAE", "RMSE", "MAPE", "R2"]].mean().to_dict())


    # ---- Train final on all data ----
    pipe.fit(X, y)

    # ---- Save model + features + lookups ----
    with open(f"{out_dir}/final_price_pipe.pkl", "wb") as f:
        pickle.dump(pipe, f)

    with open(f"{out_dir}/model_features.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)

    bm_lookup.to_csv(f"{out_dir}/new_price_lookup_bm.csv", index=False)
    b_lookup.to_csv(f"{out_dir}/new_price_lookup_b.csv", index=False)

    print(f"\n✅ Saved to {out_dir}/:")
    print(" - final_price_pipe.pkl")
    print(" - model_features.pkl")
    print(" - new_price_lookup_bm.csv")
    print(" - new_price_lookup_b.csv")

if __name__ == "__main__":
    # Example:
    # python -m src.train data/raw/Australian\ Vehicle\ Prices.csv
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", type=str)
    p.add_argument("--out_dir", type=str, default="models")
    args = p.parse_args()
    train(args.csv_path, args.out_dir)
