from pathlib import Path

import joblib
import kagglehub
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "decision_tree_pipeline.joblib"

st.set_page_config(
    page_title="Hull Tactical Market Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_competition_data() -> pd.DataFrame | None:
    try:
        competition_path = kagglehub.competition_download("hull-tactical-market-prediction")
        data_dir = Path(competition_path)
        csv_files = sorted(data_dir.glob("*.csv"))
        if not csv_files:
            return None
        return pd.read_csv(csv_files[0])
    except Exception as exc:
        st.sidebar.warning(f"Kaggle data unavailable: {exc}")
        return None


@st.cache_data
def make_sample_values(feature_names: list[str], sample_row: pd.Series | None) -> dict[str, float]:
    defaults: dict[str, float] = {}
    for feature in feature_names:
        if sample_row is not None and feature in sample_row:
            value = sample_row[feature]
        elif feature.startswith("D"):
            value = 0
        else:
            value = 0.0
        defaults[feature] = float(value)
    return defaults


@st.cache_data
def build_feature_groups(feature_names: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    section_names = {
        "D": "Binary features",
        "E": "Economic features",
        "I": "Indicator features",
        "M": "Momentum features",
        "P": "Pattern features",
        "S": "Signal features",
        "V": "Volume features",
    }
    for feature in feature_names:
        groups.setdefault(section_names.get(feature[0].upper(), "Custom features"), []).append(feature)
    return dict(sorted(groups.items()))


def render_prediction_panel(model, feature_names: list[str], sample_values: dict[str, float]) -> None:
    st.subheader("Feature input form")
    groups = build_feature_groups(feature_names)
    user_input: dict[str, float] = {}

    for section_name, section_features in groups.items():
        with st.expander(section_name, expanded=section_name == "Binary features"):
            columns = st.columns(3)
            for index, feature in enumerate(section_features):
                default_value = sample_values.get(feature, 0.0)
                is_binary = feature.startswith("D")
                value = columns[index % 3].number_input(
                    feature,
                    value=float(default_value),
                    step=1.0 if is_binary else 0.01,
                    format="%.0f" if is_binary else "%.6f",
                    key=f"feature_{feature}",
                )
                user_input[feature] = value

    input_df = pd.DataFrame([user_input], columns=feature_names)

    if st.button("Generate market forecast", type="primary", use_container_width=True):
        try:
            prediction = float(model.predict(input_df)[0])
        except Exception as exc:
            st.error(f"Prediction error: {exc}")
            return

        direction = "Bullish" if prediction > 0 else "Bearish" if prediction < 0 else "Neutral"
        signal_magnitude = abs(prediction)

        columns = st.columns(3)
        columns[0].metric("Forecasted return", f"{prediction:.4f}")
        columns[1].metric("Market direction", direction)
        columns[2].metric("Signal magnitude", f"{signal_magnitude:.4f}")

        if prediction > 0:
            st.success("The model signals a positive forward return.")
        elif prediction < 0:
            st.warning("The model signals a negative forward return.")
        else:
            st.info("The model signals a neutral forward return.")


def main() -> None:
    st.title("📈 Hull Tactical Market Prediction")
    st.caption("Data-driven market outlook built from the trained decision tree pipeline.")

    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Model error: {exc}")
        st.stop()

    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        st.error("The saved model does not expose feature names.")
        st.stop()

    with st.sidebar:
        st.header("Data source")
        uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
        uploaded_df = None

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.success(f"Uploaded {uploaded_df.shape[0]} rows.")
            except Exception as exc:
                st.error(f"Unable to read CSV: {exc}")

        if st.button("Load Kaggle competition data"):
            with st.spinner("Downloading data..."):
                kaggle_df = load_competition_data()
                if kaggle_df is not None:
                    st.session_state["competition_df"] = kaggle_df
                    st.success(f"Loaded {kaggle_df.shape[0]} rows.")

        st.divider()
        st.subheader("Model summary")
        st.write(f"Features: {len(feature_names)}")
        st.write(f"Pipeline: {type(model).__name__}")

    preview_df = st.session_state.get("competition_df", uploaded_df)
    if preview_df is not None:
        st.subheader("Data preview")
        st.dataframe(preview_df.head(10), use_container_width=True)
        sample_row = preview_df.iloc[0]
    else:
        st.info("No dataset loaded. Upload a CSV or authenticate Kaggle to load competition data.")
        sample_row = None

    sample_values = make_sample_values(feature_names, sample_row)
    render_prediction_panel(model, feature_names, sample_values)

    regressor = model.named_steps.get("regressor") if hasattr(model, "named_steps") else None
    preprocessor = model.named_steps.get("preprocessor") if hasattr(model, "named_steps") else None

    if regressor is not None and hasattr(regressor, "feature_importances_"):
        st.subheader("Top feature importance")
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            transformed_names = list(preprocessor.get_feature_names_out())
            importance_names = transformed_names if len(transformed_names) == len(regressor.feature_importances_) else feature_names
        else:
            importance_names = feature_names

        importances = pd.Series(regressor.feature_importances_, index=importance_names).sort_values(ascending=False)
        st.bar_chart(importances.head(10))


if __name__ == "__main__":
    main()
