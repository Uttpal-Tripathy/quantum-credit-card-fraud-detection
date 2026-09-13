"""Page 9 — Research Results: the complete experiment comparison table with
CSV/JSON/PNG export (spec section 21)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from dashboard.components.data import get_registry_df, registry_csv_path
from dashboard.components.theme import inject_theme, render_header

st.set_page_config(page_title="Research Results", page_icon="📊", layout="wide")
inject_theme()
render_header("Research Results", "The complete, unfiltered experiment comparison table.")

df = get_registry_df()
if df.empty:
    st.warning(
        "No experiments recorded yet. Run experiments via Page 8 or "
        "`python scripts/run_benchmark.py` to populate this table.",
        icon="⚠️",
    )
    st.stop()

st.caption(f"Source: `{registry_csv_path()}` — {len(df)} recorded experiment rows.")

datasets_available = sorted(df["dataset"].dropna().unique().tolist())
models_available = sorted(df["model"].dropna().unique().tolist())
col1, col2 = st.columns(2)
dataset_filter = col1.multiselect("Filter by dataset", datasets_available, default=datasets_available)
model_filter = col2.multiselect("Filter by model", models_available, default=models_available)

filtered = df[df["dataset"].isin(dataset_filter) & df["model"].isin(model_filter)]
st.dataframe(filtered, width='stretch', hide_index=True)

st.divider()
st.markdown("#### Export")
exp_col1, exp_col2, exp_col3 = st.columns(3)

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
exp_col1.download_button("⬇ Download CSV", data=csv_bytes, file_name="qgfda_comparison_table.csv", mime="text/csv")

json_bytes = json.dumps(filtered.to_dict(orient="records"), indent=2, default=str).encode("utf-8")
exp_col2.download_button("⬇ Download JSON", data=json_bytes, file_name="qgfda_comparison_table.json", mime="application/json")

if not filtered.empty and "pr_auc" in filtered.columns:
    chart_df = filtered.dropna(subset=["pr_auc"]).sort_values("pr_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(chart_df))))
    ax.barh(chart_df["model"] + " (" + chart_df["dataset"].astype(str) + ")", chart_df["pr_auc"], color="#22d3ee")
    ax.set_xlabel("PR-AUC")
    ax.set_title("PR-AUC by model / dataset")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    exp_col3.download_button("⬇ Download PR-AUC Chart (PNG)", data=buf.getvalue(),
                              file_name="qgfda_pr_auc_chart.png", mime="image/png")
    st.pyplot(fig, width='stretch')
    plt.close(fig)
