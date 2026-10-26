"""Statistical-model page routing and explicit missing-row policy."""

from datetime import datetime, timezone
from html import escape

import streamlit as st

from modules.models.base_models import render_base_model_tab
from modules.models.causal import render_causal_tab
from modules.models.machine_learning import render_machine_learning_tab
from modules.models.mixed_effects import render_mixed_effects_tab
from modules.models.survival import render_survival_tab
from modules.models.time_series import render_time_series_tab


MODEL_OPTIONS = [
    "📊 Base Models  (Linear / Logistic / Poisson / Negative Binomial)",
    "🫀 Survival Analysis  (Kaplan-Meier + Cox PH)",
    "📅 Time Series + Interrupted Time Series",
    "🔀 Mixed Effects Models  (LME + GLME)",
    "⚖️ Causal Inference  (PSM + IPW)",
    "🤖 Machine Learning  (Classification / Regression / Clustering)",
]


def generate_model_html_report(dataframe, selected_model, has_cleaned_data):
    """Build a downloadable HTML summary of the current modeling context."""
    columns_html = dataframe.dtypes.astype(str).rename("Data type").to_frame()
    columns_html.index.name = "Column"
    columns_html = columns_html.to_html(classes="schema", border=0)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cleaned_status = "Available" if has_cleaned_data else "Not available"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Statistical Modeling Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px auto; max-width: 1100px;
         padding: 0 20px; background: #F8FAFC; color: #0F172A; }}
  header {{ background: linear-gradient(135deg, #1E3A8A, #2563EB, #38BDF8);
            color: white; padding: 28px; border-radius: 14px; }}
  h1 {{ margin: 0 0 8px; }}
  section {{ background: white; padding: 20px; margin: 20px 0;
             border: 1px solid #E2E8F0; border-radius: 12px; }}
  h2 {{ color: #1E3A8A; }}
  .metrics {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  .metric {{ background: #EFF6FF; padding: 14px 18px; border-radius: 10px; }}
  .label {{ color: #475569; font-size: 13px; }}
  .value {{ font-weight: bold; font-size: 20px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #CBD5E1; padding: 8px; text-align: left; }}
  th {{ background: #E0F2FE; }}
</style>
</head>
<body>
<header>
  <h1>Statistical Modeling Report</h1>
  <div>Generated {escape(generated_at)}</div>
</header>
<section>
  <h2>Modeling setup</h2>
  <div class="metrics">
    <div class="metric"><div class="label">Selected model</div>
      <div class="value">{escape(selected_model)}</div></div>
    <div class="metric"><div class="label">Rows available</div>
      <div class="value">{len(dataframe):,}</div></div>
    <div class="metric"><div class="label">Columns available</div>
      <div class="value">{len(dataframe.columns):,}</div></div>
    <div class="metric"><div class="label">Cleaned data</div>
      <div class="value">{cleaned_status}</div></div>
  </div>
  <p>Modeling uses the full dataset unless temporary exclusion of ineligible
  rows is explicitly allowed in the application.</p>
</section>
<section>
  <h2>Dataset columns</h2>
  {columns_html}
</section>
</body>
</html>"""


def render_model_page(df, df_cleaned, plot_template):
    """Render model selection while preserving full-data safeguards."""
    st.markdown("## 🤖 Statistical Models")
    st.info(
        f"Full-data modeling is enabled: all {len(df):,} rows are passed "
        "to the selected model. No sampling is applied, and no row is "
        "temporarily excluded unless you explicitly allow it below."
    )
    missing_policy_label = st.radio(
        "If required model values are missing or invalid",
        [
            "Stop and ask me to clean the data",
            "Temporarily exclude ineligible rows for this model",
        ],
        horizontal=True,
        key="model_missing_policy_label",
        help=(
            "Temporary exclusion affects only the current model input. It never "
            "deletes rows from the original or cleaned dataset."
        ),
    )
    st.session_state["model_missing_policy"] = (
        "allow_temporary_exclusion"
        if missing_policy_label.startswith("Temporarily")
        else "stop"
    )
    if st.session_state["model_missing_policy"] == "stop":
        st.caption(
            "Safe default: the model will stop if even one row would need to be excluded."
        )
    else:
        st.warning(
            "You allowed temporary model-only exclusion. The audit will show "
            "the exact number of affected rows before results."
        )

    if st.session_state.get("selected_model_radio_v2") not in MODEL_OPTIONS:
        st.session_state["selected_model_radio_v2"] = MODEL_OPTIONS[0]
    selected_model = st.radio(
        "Select model", MODEL_OPTIONS, key="selected_model_radio_v2"
    )
    st.markdown("---")

    renderers = [
        render_base_model_tab,
        render_survival_tab,
        render_time_series_tab,
        render_mixed_effects_tab,
        render_causal_tab,
        render_machine_learning_tab,
    ]
    renderers[MODEL_OPTIONS.index(selected_model)](df, df_cleaned, plot_template)

    st.markdown("---")
    st.markdown("### Download modeling report")
    st.download_button(
        "📥 Download Modeling Report (HTML)",
        data=generate_model_html_report(
            df, selected_model, df_cleaned is not None
        ),
        file_name="modeling_report.html",
        mime="text/html",
        use_container_width=True,
        key="download_modeling_html",
    )
