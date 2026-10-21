# RetailIQ — Customer Analytics & Recommendation Platform

A portfolio project using the UCI Online Retail dataset to demonstrate data cleaning, EDA, RFM customer segmentation, item-based collaborative filtering, hypothesis testing, forecasting, and interactive analytics.

## Dataset
Download `Online Retail.xlsx` from https://archive.ics.uci.edu/dataset/352/online-retail and save it to `data/raw/Online Retail.xlsx`.
Dataset citation: Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. Licensed CC BY 4.0.

## Run locally
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If you already created `.venv` and installed the libraries, activate it and run `streamlit run app.py`.

## Modules
- EDA: monthly revenue, product and country summaries, visualizations.
- Segmentation: RFM features, log transform, standardization, K-Means, silhouette score.
- Recommendations: item-based collaborative filtering using cosine similarity over customer-item purchase indicators.
- Statistics: Welch's t-test, p-value and confidence interval comparing weekday/weekend order value.
- Forecasting: chronological holdout, Exponential Smoothing, MAE and RMSE.
- Data quality: cancellation/returns/missing customer handling and CSV exports.

## Important limitations
- The data is historical (2010–2011), so results are for methodology demonstration, not current market recommendations.
- This dataset has no randomized A/B test assignment; do not claim causal uplift.
- Precision@K/Recall@K should be added using a chronological user-item holdout before claiming recommender performance.
- Forecast performance should be compared to a simple baseline before claiming improvement.
- AWS and Tableau are optional next steps; do not list them as project experience until you've actually used them.

## Suggested next steps
1. Add reproducible model evaluation scripts and unit tests.
2. Add SQL ingestion/analytics and screenshots.
3. Publish a Tableau dashboard and host the Streamlit app.
4. Add an AWS S3 data storage step if appropriate and cost-safe.
