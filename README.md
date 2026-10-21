
# RetailIQ | Retail Analytics & Predictive Insights

RetailIQ is an end-to-end retail analytics application built using Python and Streamlit. It processes historical retail transaction data to uncover sales trends, analyze customer purchasing behavior, generate product recommendations, and forecast daily sales through an interactive dashboard.

**[Live Demo](https://nisha-retailiq.streamlit.app/)** | **[GitHub Repository](https://github.com/Nisha-Lohar04/RetailIQ)**

## Project Overview

Retail businesses generate large volumes of transaction data, but turning this data into useful decisions requires more than basic reporting. RetailIQ brings descriptive analytics, recommendation evaluation, and time-series forecasting into one application, helping users explore historical performance and assess potential future sales trends.

## Key Features

- **Sales Analytics:** Interactive analysis of revenue trends, product performance, and customer purchasing patterns.
- **Product Recommendations:** Generates recommendations from historical purchase behavior and evaluates whether recommended products appear in customers' subsequent purchases.
- **Daily Sales Forecasting:** Uses the Holt-Winters time-series model to forecast sales and compares its performance against a naive baseline.
- **Model Evaluation:** Reports forecasting metrics including MAE, RMSE, WAPE, and sMAPE.
- **Interactive Visualizations & Export:** Explore results through charts and download forecast data as CSV.

## Forecasting Results

Holt-Winters model performance on a historical backtest:

| Metric | Result |
|---|---:|
| MAE | £13,648.70 |
| RMSE | £28,228.07 |
| WAPE | 31.04% |
| MAE improvement over naive baseline | 50.04% |

*Metrics reflect the selected historical backtest and do not guarantee future performance.*

## Technology Stack

**Languages:** Python, SQL  
**Data Analysis:** Pandas, NumPy  
**Machine Learning & Forecasting:** Scikit-learn, Statsmodels  
**Visualization & Application:** Plotly, Streamlit

## Dataset

Uses the [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail), containing transaction records from 2010–2011.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
