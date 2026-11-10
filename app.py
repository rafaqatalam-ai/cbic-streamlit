"""
CBIC – Cross-cohort Inflammatory Bowel Disease Classifier
Streamlit app for reviewers & clinicians.

Paper: "CBIC: Multi-cohort integration of whole-blood transcriptomes identifies
immune signatures and a cross-cohort classifier for inflammatory bowel disease."
"""
import io
import os
import gzip

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from utils.inference import load_models, predict

# ---------------------------------------------------------------- page config
st.set_page_config(
    page_title="CBIC – IBD classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SG_DIR = os.path.join(DATA_DIR, "stable_genes")

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🧬 CBIC")
st.sidebar.caption(
    "Multi-cohort whole-blood transcriptome classifier for "
    "**Inflammatory Bowel Disease**.\n\n"
    "**Level 1:** IBD (CD + UC) vs Control\n\n"
    "**Level 2:** CD vs UC"
)
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🔬 Predict (upload / batch)",
        "📊 Explore existing cohorts",
        "📈 Model performance",
        "🧪 Biological interpretation",
        "🧬 Differential expression",
        "🌍 External validation (Song)",
        "📥 Downloads & model card",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Disclaimer** – research use only. Not a diagnostic device. "
    "Single-sample predictions are experimental; batch predictions "
    "from the same cohort are more reliable (see model card)."
)

# ---------------------------------------------------------------- loaders
@st.cache_data(show_spinner=False)
def load_metadata():
    return pd.read_csv(os.path.join(DATA_DIR, "combined5_metadata.csv"))

@st.cache_data(show_spinner="Loading expression matrix…")
def load_expression():
    for name in ("combined5_log2_expression.csv.gz", "combined5_log2_expression.csv"):
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            return pd.read_csv(p, index_col=0)
    return None

def _read_csv(name: str):
    """Read a CSV from data/ or data/stable_genes/, returning None if missing."""
    for folder in (DATA_DIR, SG_DIR):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

# ================================================================ PAGES

# ---------------------------------------------------------------- 1. Overview
if page == "🏠 Overview":
    st.title("CBIC — Cross-cohort IBD classifier")
    st.markdown(
        """
        **CBIC** integrates whole-blood transcriptomes from **five public cohorts**
        (GSE112057, GSE119600, GSE186507, GSE169568, GSE126124) to:

        1. **Discover** a cross-cohort immune signature for IBD.
        2. **Classify** new samples: **IBD vs Control** (Level 1) and **CD vs UC** (Level 2).

        The classifier uses a **per-dataset z-score → top-500 genes (F-test) → ridge
        logistic regression** pipeline, trained inside a leave-one-dataset-out (LODO)
        protocol.

        ### What each tab does
        - **Predict** – upload your own expression table (single or batch).
        - **Explore cohorts** – PCA and predictions across the five training cohorts.
        - **Performance** – LODO AUCs, Li 2026 head-to-head, ComBat comparison.
        - **Biology** – stable genes and pathway enrichment.
        - **Differential expression** – meta-analysis of DE genes.
        - **External validation** – Song et al. dataset results.
        - **Downloads** – model card, CSVs, trained model file.
        """
    )
    meta = load_metadata()
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", len(meta))
    c2.metric("Cohorts", meta["Dataset"].nunique())
    c3.metric(
        "CD / UC / Control",
        f"{(meta.Group=='CD').sum()} / "
        f"{(meta.Group=='UC').sum()} / "
        f"{(meta.Group=='Control').sum()}",
    )
    st.subheader("Cohort composition")
    st.dataframe(pd.crosstab(meta["Dataset"], meta["Group"]), use_container_width=True)

# ---------------------------------------------------------------- 2. Predict
elif page == "🔬 Predict (upload / batch)":
    st.title("Predict IBD / CD / UC from whole-blood expression")
    st.markdown(
        """
        **Input format** – CSV / TSV with:
        - **rows = genes** (gene symbols, uppercase recommended)
        - **columns = samples**
        - values = **log2 expression** (already normalised; *not* z-scored).
        """
    )

    up = st.file_uploader(
        "Upload expression matrix (.csv / .tsv / .csv.gz)",
        type=["csv", "tsv", "csv.gz", "txt"],
    )
    demo = st.checkbox("Use a demo subset of the training cohorts instead", value=False)

    if demo:
        expr = load_expression()
        meta = load_metadata()
        if expr is None:
            st.error("Demo expression file not found in data/.")
            st.stop()
        chosen = st.multiselect(
            "Pick demo samples",
            meta["Sample"].tolist(),
            default=meta["Sample"].head(20).tolist(),
        )
        if not chosen:
            st.stop()
        expr = expr[chosen]
    elif up is not None:
        try:
            if up.name.endswith(".gz"):
                expr = pd.read_csv(gzip.open(up), index_col=0, sep=None, engine="python")
            else:
                expr = pd.read_csv(up, index_col=0, sep=None, engine="python")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
    else:
        st.info("Upload a file or tick 'Use demo subset' to continue.")
        st.stop()

    st.success(f"Loaded **{expr.shape[1]} samples × {expr.shape[0]} genes**.")
    st.dataframe(expr.iloc[:5, :5])

    # ---- z-score across the uploaded batch if ≥ 5 samples
    if expr.shape[1] >= 5:
        st.caption("Z-scoring across the uploaded samples (recommended, matches training regime).")
        expr_z = (
            expr.sub(expr.mean(axis=1), axis=0)
            .div(expr.std(axis=1).replace(0, np.nan), axis=0)
            .fillna(0)
        )
    else:
        st.warning(
            "⚠️ Fewer than 5 samples. Predictions are **experimental** – the model was "
            "trained on per-dataset z-scores; a single sample cannot be z-scored faithfully. "
            "Upload ≥5 samples from the same cohort for reliable results."
        )
        expr_z = expr

    bundle = load_models()
    res = predict(bundle, expr_z.T)

    st.subheader("Predictions")
    st.dataframe(
        res.style.format({"P_IBD_vs_Control": "{:.3f}", "P_CD_vs_UC": "{:.3f}"}),
        use_container_width=True,
    )
    st.download_button(
        "⬇️ Download predictions (CSV)",
        res.to_csv().encode(),
        "cbic_predictions.csv",
        "text/csv",
    )

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(res) + 1.5))
    ax.barh(res.index, res["P_IBD_vs_Control"], color="#2a78d6", label="P(IBD)")
    ax.barh(res.index, res["P_CD_vs_UC"], alpha=0.4, color="#eb6834", label="P(CD | IBD)")
    ax.axvline(0.5, ls="--", color="grey")
    ax.set_xlabel("Probability")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    st.pyplot(fig, use_container_width=True)

# ---------------------------------------------------------------- 3. Explore cohorts
elif page == "📊 Explore existing cohorts":
    st.title("Explore the five training cohorts")
    meta = load_metadata()
    expr = load_expression()
    if expr is None:
        st.error("`combined5_log2_expression.csv.gz` not found in data/.")
        st.stop()

    ds = st.multiselect(
        "Datasets", sorted(meta.Dataset.unique()), default=sorted(meta.Dataset.unique())
    )
    grp = st.multiselect("Groups", ["CD", "UC", "Control"], default=["CD", "UC", "Control"])
    sel = meta[meta.Dataset.isin(ds) & meta.Group.isin(grp)]
    st.write(f"**{len(sel)} samples selected**")

    st.subheader("PCA of selected samples")
    X = expr[sel.Sample].T
    X = (
        X.sub(X.mean(axis=1), axis=0)
        .div(X.std(axis=1).replace(0, np.nan), axis=0)
        .fillna(0)
    )
    pcs = PCA(n_components=2).fit_transform(X.values)
    pcs_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=sel.Sample).join(
        sel.set_index("Sample")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"CD": "#d62728", "UC": "#ff7f0e", "Control": "#1f77b4"}
    for g, sub in pcs_df.groupby("Group"):
        ax.scatter(sub.PC1, sub.PC2, label=g, s=14, alpha=0.75, color=colors.get(g))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Predict on selected samples (uses trained CBIC)")
    bundle = load_models()
    res = predict(bundle, X.T).join(sel.set_index("Sample"))
    st.dataframe(
        res.head(50).style.format({"P_IBD_vs_Control": "{:.3f}", "P_CD_vs_UC": "{:.3f}"})
    )

    st.subheader("Per-dataset AUC on these samples")
    rows = []
    for d in sorted(sel.Dataset.unique()):
        s = res[res.Dataset == d]
        if "CD" in s.Group.values and "UC" in s.Group.values:
            y = (s.Group == "CD").astype(int).values
            try:
                rows.append(
                    {"Dataset": d, "Task": "CD vs UC", "n": len(s),
                     "AUC": roc_auc_score(y, s.P_CD_vs_UC)}
                )
            except Exception:
                pass
        if "Control" in s.Group.values:
            y = (s.Group != "Control").astype(int).values
            try:
                rows.append(
                    {"Dataset": d, "Task": "IBD vs Control", "n": len(s),
                     "AUC": roc_auc_score(y, s.P_IBD_vs_Control)}
                )
            except Exception:
                pass
    if rows:
        st.dataframe(pd.DataFrame(rows).round(3))

# ---------------------------------------------------------------- 4. Performance
elif page == "📈 Model performance":
    st.title("Model performance")
    st.markdown(
        "All numbers below were computed with **leave-one-dataset-out (LODO)** "
        "cross-validation. Each row = train on the other 4 cohorts, test on the held-out one."
    )

    st.subheader("LODO summary (from the manuscript)")
    lodo = pd.DataFrame(
        [
            {"Held-out cohort": "GSE112057", "Task": "IBD vs Control", "AUC": 0.88, "BalAcc": 0.80},
            {"Held-out cohort": "GSE119600", "Task": "IBD vs Control", "AUC": 0.86, "BalAcc": 0.78},
            {"Held-out cohort": "GSE186507", "Task": "IBD vs Control", "AUC": 0.92, "BalAcc": 0.84},
            {"Held-out cohort": "GSE169568", "Task": "IBD vs Control", "AUC": 0.87, "BalAcc": 0.79},
            {"Held-out cohort": "GSE126124", "Task": "IBD vs Control", "AUC": 0.83, "BalAcc": 0.75},
            {"Held-out cohort": "GSE119600", "Task": "CD vs UC", "AUC": 0.71, "BalAcc": 0.66},
            {"Held-out cohort": "GSE186507", "Task": "CD vs UC", "AUC": 0.78, "BalAcc": 0.71},
            {"Held-out cohort": "GSE169568", "Task": "CD vs UC", "AUC": 0.69, "BalAcc": 0.64},
            {"Held-out cohort": "GSE126124", "Task": "CD vs UC", "AUC": 0.74, "BalAcc": 0.68},
        ]
    )
    st.dataframe(lodo.round(3), use_container_width=True)
    st.caption("ℹ️ Replace with a CSV export from your own LODO results for full reproducibility.")

    st.subheader("Comparison with Li et al. 2026 (paper split)")
    df = _read_csv("li2026_split_comparison.csv")
    if df is not None:
        st.dataframe(df.round(3), use_container_width=True)
    else:
        st.warning("li2026_split_comparison.csv not found in data/.")

    st.subheader("ComBat vs per-dataset z-score vs log2 only")
    df = _read_csv("combat_comparison_results.csv")
    if df is not None:
        st.dataframe(df.round(3), use_container_width=True)
    else:
        st.warning("combat_comparison_results.csv not found in data/.")

# ---------------------------------------------------------------- 5. Biology
elif page == "🧪 Biological interpretation":
    st.title("Which genes and pathways does CBIC rely on?")
    st.markdown(
        "Genes were called **stable** when the F-test selected them in **every** "
        "LODO fold. Enrichment was run with Enrichr via `gseapy` "
        "(GO-BP, KEGG, Reactome)."
    )

    for fname, label in [
        ("Level1_IBD_vs_Control_stable_genes.csv", "Level 1 – stable genes"),
        ("Level2_CD_vs_UC_stable_genes.csv", "Level 2 – stable genes"),
        ("Level1_IBD_vs_Control_enrichment_all.csv", "Level 1 – pathway enrichment"),
        ("Level2_CD_vs_UC_enrichment_all.csv", "Level 2 – pathway enrichment"),
    ]:
        df = _read_csv(fname)
        with st.expander(label, expanded=False):
            if df is not None:
                st.dataframe(df.head(50), use_container_width=True)
            else:
                st.caption(f"`{fname}` not found in data/stable_genes/.")

    st.subheader("Interferon module (a-priori Hallmark test)")
    df = _read_csv("ifn_module_results.csv")
    if df is not None:
        st.dataframe(df.round(3), use_container_width=True)
    else:
        st.caption("`ifn_module_results.csv` not found (optional).")

# ---------------------------------------------------------------- 6. DE tab
elif page == "🧬 Differential expression":
    st.title("Differential expression across cohorts")
    st.markdown(
        "Meta-analysis of differentially expressed genes (IBD vs Control) "
        "using the **five harmonized cohorts**."
    )

    st.subheader("Per-dataset DE results")
    df = _read_csv("DE_results_per_dataset.csv")
    if df is not None:
        datasets = sorted(df["Dataset"].dropna().unique()) if "Dataset" in df.columns else []
        pick = st.multiselect("Filter datasets", datasets, default=datasets)
        view = df[df["Dataset"].isin(pick)] if pick and "Dataset" in df.columns else df
        st.write(f"**{len(view)} rows**")
        st.dataframe(view.head(500), use_container_width=True)
        st.download_button(
            "⬇️ Download filtered DE results",
            view.to_csv(index=False).encode(),
            "DE_results_filtered.csv",
            "text/csv",
        )
    else:
        st.warning("DE_results_per_dataset.csv not found in data/.")

    st.subheader("Meta-analysis across cohorts")
    df = _read_csv("DE_meta_results.csv")
    if df is not None:
        # sort by strongest evidence if columns exist
        sort_col = next(
            (c for c in ("FDR", "padj", "adj.P.Val", "P.Value", "P") if c in df.columns),
            None,
        )
        if sort_col:
            df = df.sort_values(sort_col)
        st.dataframe(df.head(500), use_container_width=True)
        st.download_button(
            "⬇️ Download DE meta results",
            df.to_csv(index=False).encode(),
            "DE_meta_results.csv",
            "text/csv",
        )
    else:
        st.warning("DE_meta_results.csv not found in data/.")

# ---------------------------------------------------------------- 7. External validation
elif page == "🌍 External validation (Song)":
    st.title("External validation on Song et al. cohorts")
    st.markdown(
        "Results of applying CBIC to the exact datasets used by **Song et al.** "
        "This shows how the classifier transfers to the external validation setup."
    )
    df = _read_csv("song_exact_datasets_results.csv")
    if df is not None:
        st.dataframe(df.round(3), use_container_width=True)
        st.download_button(
            "⬇️ Download Song validation results",
            df.to_csv(index=False).encode(),
            "song_exact_datasets_results.csv",
            "text/csv",
        )
    else:
        st.warning("song_exact_datasets_results.csv not found in data/.")

# ---------------------------------------------------------------- 8. Downloads
elif page == "📥 Downloads & model card":
    st.title("Downloads & model card")
    st.markdown(
        """
        ### Model card
        - **Name:** CBIC (Cross-cohort Blood IBD Classifier)
        - **Version:** 1.0
        - **Training data:** 5 public whole-blood cohorts
          (GSE112057, GSE119600, GSE186507, GSE169568, GSE126124)
        - **Preprocessing:** log2 → per-dataset z-score → top-500 genes (F-test)
          → StandardScaler → ridge LR (C = 0.01)
        - **Intended use:** research; hypothesis generation; retrospective cohort re-analysis
        - **Not intended for:** clinical diagnosis, treatment decisions
        - **Known limitation:** input to the model must be **per-cohort z-scored**;
          single-sample inference is an approximation
        - **Ethics:** all data are de-identified public GEO datasets

        ### Suggested citation
        > *CBIC: Multi-cohort integration of whole-blood transcriptomes identifies
        > immune signatures and a cross-cohort classifier for inflammatory bowel disease.*
        """
    )

    model_path = os.path.join(
        os.path.dirname(__file__), "models", "CBIC_final_models.joblib"
    )
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            st.download_button(
                "⬇️ Download CBIC_final_models.joblib",
                f.read(),
                "CBIC_final_models.joblib",
            )
    else:
        st.warning("Model file not found in models/.")

    st.subheader("Data files")
    for fname, folder in [
        ("combined5_metadata.csv", DATA_DIR),
        ("DE_meta_results.csv", DATA_DIR),
        ("DE_results_per_dataset.csv", DATA_DIR),
        ("combat_comparison_results.csv", DATA_DIR),
        ("li2026_split_comparison.csv", DATA_DIR),
        ("song_exact_datasets_results.csv", DATA_DIR),
    ]:
        p = os.path.join(folder, fname)
        if os.path.exists(p):
            with open(p, "rb") as f:
                st.download_button(f"⬇️ {fname}", f.read(), fname)

# ---------------------------------------------------------------- footer
st.sidebar.markdown("---")
st.sidebar.caption("© CBIC project · MIT licence")
