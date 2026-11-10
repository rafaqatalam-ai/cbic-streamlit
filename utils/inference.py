"""Shared inference utilities for the CBIC Streamlit app."""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "CBIC_final_models.joblib")


@st.cache_resource(show_spinner="Loading CBIC model…")
def load_models():
    """Load the trained CBIC bundle: {'level1', 'level2', 'genes'}."""
    bundle = joblib.load(MODEL_PATH)
    return bundle


def align_to_model(X: pd.DataFrame, genes: list) -> pd.DataFrame:
    """Reindex expression matrix (samples × genes) to the model's gene list.
    Missing genes are filled with 0 (their training mean after z-scoring);
    extra genes are dropped. Gene symbols are uppercased for matching."""
    X = X.copy()
    X.columns = X.columns.astype(str).str.upper()
    missing = [g for g in genes if g not in X.columns]
    if missing:
        for g in missing:
            X[g] = 0.0
    return X[genes]


def predict(bundle, expr_df: pd.DataFrame) -> pd.DataFrame:
    """expr_df: samples × genes (raw log2, NOT z-scored).
    Returns DataFrame with Level 1 & Level 2 probabilities."""
    lvl1, lvl2, genes = bundle["level1"], bundle["level2"], bundle["genes"]
    X = align_to_model(expr_df, genes).values
    p1 = lvl1.predict_proba(X)[:, 1]          # P(IBD)
    p2 = lvl2.predict_proba(X)[:, 1]          # P(CD | IBD)
    return pd.DataFrame(
        {
            "P_IBD_vs_Control": p1,
            "P_CD_vs_UC": p2,
            "Predicted_Level1": np.where(p1 > 0.5, "IBD", "Control"),
            "Predicted_Level2_if_IBD": np.where(
                p1 > 0.5, np.where(p2 > 0.5, "CD", "UC"), "—"
            ),
        },
        index=expr_df.index,
    )