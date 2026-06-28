# PricePulse: Amazon Product Price Intelligence Dashboard

PricePulse is a Streamlit dashboard for Amazon product price analysis and prediction.

## Features

- Executive summary with KPIs and market insights
- Category-wise price, discount, rating, and review analysis
- Product insights for expensive, best-rated, most-reviewed, and highest-discount products
- Trend and correlation analysis with Plotly charts
- Random Forest based discounted price predictor
- Searchable data explorer with CSV download

## Project Structure

```text
amazon-price-analysis/
+-- app.py
+-- requirements.txt
+-- README.md
+-- data/
    +-- amazon.csv
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Upload this folder to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub repository.
4. Select `app.py` as the main file.
5. Deploy the app.
