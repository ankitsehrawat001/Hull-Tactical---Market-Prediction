# Hull Tactical Market Prediction

A Streamlit-based market prediction dashboard that uses a trained decision-tree pipeline to forecast directional market movement from engineered financial features.

## Overview

This project combines:

- a trained machine learning pipeline saved as `decision_tree_pipeline.joblib`
- a Streamlit UI in `app.py`
- optional CSV upload support for local analysis
- Kaggle competition data loading through `kagglehub`

The application lets you:

- upload your own market dataset as CSV
- load Kaggle competition data
- inspect the dataset preview
- adjust feature values in a structured form
- generate a bullish/bearish/neutral market forecast
- review the most influential features from the model

## Project structure

```text
Hull Tactical - Market Prediction/
├── app.py                       # Streamlit application entry point
├── decision_tree_pipeline.joblib # Trained model artifact
├── frontend/                   # Frontend assets (HTML/CSS)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .venv/                      # Local virtual environment
```

## Features

- Predictive modeling using a scikit-learn decision tree pipeline
- Feature grouping by logical sections such as binary, momentum, volume, and signal inputs
- Interactive input form for forecasting market conditions
- Sidebar data source controls for uploaded CSV files or Kaggle data
- Data preview and feature importance visualization

## Requirements

- Python 3.10+
- pip
- A Kaggle account and API token if you want to download competition data automatically

## Setup

1. Open a terminal in the project root.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

## Usage

1. Launch the app.
2. In the sidebar, either:
   - upload a CSV file, or
   - click `Load Kaggle competition data`
3. Review the preview table.
4. Adjust the feature values in the input form.
5. Click `Generate market forecast` to see the prediction and signal strength.

## Notes

- The model is expected to be saved in the root directory as `decision_tree_pipeline.joblib`.
- If the Kaggle dataset is unavailable, the app still works with a manually uploaded CSV.
- Forecasts are generated from the model's trained feature schema, so uploaded data should match the expected columns.

## Dependencies

The project uses the following packages:

- streamlit
- pandas
- numpy
- scikit-learn
- joblib
- kagglehub

## License

This project does not currently include a license file. Add a license if you plan to distribute or share it publicly.
