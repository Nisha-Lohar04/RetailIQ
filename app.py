from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="RetailIQ | Customer Intelligence", page_icon="📊", layout="wide")
st.title("RetailIQ")
st.caption("Retail customer intelligence: segmentation • recommendations • statistics • forecasting")
DEFAULT_PATH = Path("data/raw/Online Retail.xlsx")

@st.cache_data(show_spinner="Loading and preparing transactions…")
def load_data(file_bytes=None, file_name=None):
    if file_bytes is not None:
        from io import BytesIO
        raw = pd.read_excel(BytesIO(file_bytes))
    else:
        if not DEFAULT_PATH.exists():
            return None, None
        raw = pd.read_excel(DEFAULT_PATH)
    raw.columns = [str(c).strip() for c in raw.columns]
    required = {"InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice","CustomerID","Country"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Dataset missing columns: {sorted(required - set(raw.columns))}")

    # Capture source-quality metrics before cleaning changes the data.
    source_row_count = len(raw)
    duplicate_rows_before_cleaning = int(raw.duplicated().sum())
    missing_descriptions_before_cleaning = int(raw["Description"].isna().sum())
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
    raw["InvoiceNo"] = raw["InvoiceNo"].astype(str).str.strip()
    raw["StockCode"] = raw["StockCode"].astype(str).str.strip()
    raw["Description"] = raw["Description"].fillna("Unknown product").astype(str).str.strip()
    raw["Quantity"] = pd.to_numeric(raw["Quantity"], errors="coerce")
    raw["UnitPrice"] = pd.to_numeric(raw["UnitPrice"], errors="coerce")
    raw["CustomerID"] = pd.to_numeric(raw["CustomerID"], errors="coerce")
    raw = raw.drop_duplicates()
    raw = raw.dropna(subset=["InvoiceDate","Quantity","UnitPrice","InvoiceNo","StockCode"])
    raw["Revenue"] = raw["Quantity"] * raw["UnitPrice"]
    raw["IsCancellation"] = raw["InvoiceNo"].str.upper().str.startswith("C")
    # Retain all cleaned rows for audit; sales analysis excludes cancellations, returns and non-positive prices.
    sales = raw.loc[
        (~raw["IsCancellation"]) & (raw["Quantity"] > 0) &
        (raw["UnitPrice"] > 0) & raw["CustomerID"].notna()
    ].copy()
    sales["CustomerID"] = sales["CustomerID"].astype("int64").astype(str)

    raw.attrs["source_row_count"] = source_row_count
    raw.attrs["duplicate_rows_before_cleaning"] = duplicate_rows_before_cleaning
    raw.attrs["missing_descriptions_before_cleaning"] = missing_descriptions_before_cleaning

    return raw, sales

def make_rfm(sales):
    snapshot = sales["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = sales.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        Items=("Quantity", "sum"),
    ).reset_index()
    return rfm

def make_clusters(rfm, k):
    features = rfm[["Recency","Frequency","Monetary"]].copy()
    # Log-transform skewed positive variables before scaling.
    features["Recency"] = np.log1p(features["Recency"].clip(lower=0))
    features["Frequency"] = np.log1p(features["Frequency"].clip(lower=0))
    features["Monetary"] = np.log1p(features["Monetary"].clip(lower=0))
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    result = rfm.copy()
    result["Cluster"] = labels.astype(str)
    score = silhouette_score(X, labels, sample_size=min(2500, len(X)), random_state=42) if len(set(labels)) > 1 else np.nan
    return result, score

def recommend_products(sales, customer_id, top_n=10):
    basket = sales.pivot_table(index="CustomerID", columns="StockCode", values="Quantity",
                               aggfunc="sum", fill_value=0)
    basket = (basket > 0).astype(np.uint8)
    if customer_id not in basket.index:
        return pd.DataFrame(columns=["StockCode","Description","SimilarityScore"])
    # Item-based cosine similarity over customer purchase indicators.
    item_matrix = basket.T
    sim = cosine_similarity(item_matrix)
    sim_df = pd.DataFrame(sim, index=item_matrix.index, columns=item_matrix.index)
    purchased = basket.loc[customer_id]
    bought_codes = set(purchased[purchased > 0].index)
    scores = sim_df.loc[:, list(bought_codes)].sum(axis=1).sort_values(ascending=False)
    scores = scores.drop(labels=list(bought_codes), errors="ignore").head(top_n)
    descriptions = sales.drop_duplicates("StockCode").set_index("StockCode")["Description"]
    out = scores.rename("SimilarityScore").reset_index()
    out.columns = ["StockCode","SimilarityScore"]
    out["Description"] = out["StockCode"].map(descriptions).fillna("Unknown product")
    return out[["StockCode","Description","SimilarityScore"]]

def evaluate_recommender(sales, top_n=10, max_customers=100):
    """Evaluate item-based recommendations against later purchases."""
    from sklearn.metrics.pairwise import cosine_similarity

    dates = sales["InvoiceDate"]
    cutoff = dates.quantile(0.80)
    train = sales.loc[dates <= cutoff].copy()
    test = sales.loc[dates > cutoff].copy()

    if train.empty or test.empty:
        return None

    train_basket = train.pivot_table(
        index="CustomerID", columns="StockCode", values="Quantity",
        aggfunc="sum", fill_value=0,
    )
    train_basket = (train_basket > 0).astype(np.uint8)

    item_matrix = train_basket.T
    if len(item_matrix) < 2:
        return None

    similarities = cosine_similarity(item_matrix)
    similarity_df = pd.DataFrame(
        similarities, index=item_matrix.index, columns=item_matrix.index
    )

    later_purchases = test.groupby("CustomerID")["StockCode"].agg(
        lambda values: set(values)
    )
    eligible = [
        customer for customer in later_purchases.index
        if customer in train_basket.index
        and len(later_purchases[customer]) > 0
        and train_basket.loc[customer].sum() > 0
    ]

    if not eligible:
        return None

    # Deterministic sampling keeps evaluation practical on larger datasets.
    eligible = sorted(eligible)[:max_customers]
    precisions, recalls = [], []

    for customer in eligible:
        purchased = set(train_basket.columns[train_basket.loc[customer] > 0])
        candidate_codes = [c for c in similarity_df.index if c not in purchased]

        if not candidate_codes:
            continue

        history_codes = [c for c in purchased if c in similarity_df.columns]
        if not history_codes:
            continue

        scores = similarity_df.loc[candidate_codes, history_codes].sum(axis=1)
        recommendations = set(scores.nlargest(top_n).index)
        actual = later_purchases[customer]

        hits = len(recommendations & actual)
        precisions.append(hits / top_n)
        recalls.append(hits / len(actual))

    if not precisions:
        return None

    return {
        "customers_evaluated": len(precisions),
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "cutoff_date": cutoff,
        "k": top_n,
    }


def daily_forecast(sales, horizon=30):
    daily = (
        sales.set_index("InvoiceDate")
        .resample("D")["Revenue"]
        .sum()
        .asfreq("D", fill_value=0)
    )

    if len(daily) < 20:
        return daily, None, None

    test_n = min(30, max(7, int(len(daily) * 0.2)))
    train, test = daily.iloc[:-test_n], daily.iloc[-test_n:]

    try:
        # Fit candidate models using training data only.
        candidates = {}

        exp_model = ExponentialSmoothing(
            train, trend="add", seasonal=None,
            initialization_method="estimated"
        ).fit()
        candidates["Exponential Smoothing"] = exp_model.forecast(len(test))

        # Holt-Winters requires at least two complete seasonal cycles.
        if len(train) >= 14:
            try:
                hw_model = ExponentialSmoothing(
                    train,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=7,
                    initialization_method="estimated"
                ).fit()
                candidates["Holt-Winters (7-day seasonality)"] = (
                    hw_model.forecast(len(test))
                )
            except (ValueError, TypeError, np.linalg.LinAlgError):
                pass

        baseline_pred = pd.Series(
            float(train.iloc[-1]), index=test.index, name="NaiveBaseline"
        )
        baseline_mae = float(mean_absolute_error(test, baseline_pred))
        baseline_rmse = float(np.sqrt(mean_squared_error(test, baseline_pred)))

        # Compare candidates on exactly the same chronological holdout.
        results = {}
        actual = np.asarray(test, dtype=float)

        for name, prediction in candidates.items():
            prediction = pd.Series(
                np.asarray(prediction, dtype=float), index=test.index
            )
            predicted = prediction.to_numpy()
            denominator = np.abs(actual) + np.abs(predicted)
            valid = denominator > 0
            total_actual = np.sum(np.abs(actual))

            results[name] = {
                "prediction": prediction,
                "mae": float(mean_absolute_error(test, prediction)),
                "rmse": float(np.sqrt(mean_squared_error(test, prediction))),
                "wape": (
                    float(np.sum(np.abs(actual - predicted)) / total_actual * 100)
                    if total_actual > 0 else np.nan
                ),
                "smape": (
                    float(np.mean(
                        2 * np.abs(actual[valid] - predicted[valid])
                        / denominator[valid]
                    ) * 100)
                    if valid.any() else np.nan
                ),
            }

        best_name = min(results, key=lambda name: results[name]["mae"])
        best = results[best_name]

        # Refit the selected model on all available history for future forecasts.
        if best_name == "Holt-Winters (7-day seasonality)":
            final_model = ExponentialSmoothing(
                daily,
                trend="add",
                seasonal="add",
                seasonal_periods=7,
                initialization_method="estimated"
            ).fit()
        else:
            final_model = ExponentialSmoothing(
                daily, trend="add", seasonal=None,
                initialization_method="estimated"
            ).fit()

        future = final_model.forecast(horizon)

        comparison_rows = []
        for name, metrics in results.items():
            comparison_rows.append({
                "Model": name,
                "MAE (£)": metrics["mae"],
                "RMSE (£)": metrics["rmse"],
                "WAPE (%)": metrics["wape"],
                "sMAPE (%)": metrics["smape"],
                "Selected": name == best_name,
            })

        comparison_df = (
            pd.DataFrame(comparison_rows)
            .sort_values("MAE (£)")
            .reset_index(drop=True)
        )

        return daily, (test, best["prediction"]), (
            best["mae"], best["rmse"], best["wape"], best["smape"],
            future, baseline_mae, baseline_rmse, best_name, comparison_df
        )

    except Exception as exc:
        print(f"Forecast evaluation failed: {exc}")
        return daily, None, None


uploaded = st.sidebar.file_uploader("Optional: upload Online Retail.xlsx", type=["xlsx"])
try:
    if uploaded:
        raw, sales = load_data(uploaded.getvalue(), uploaded.name)
    else:
        raw, sales = load_data()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

if raw is None:
    st.warning("Dataset not found. Place Online Retail.xlsx at data/raw/Online Retail.xlsx, or upload it in the sidebar.")
    st.markdown("[Get the dataset from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online-retail)")
    st.stop()
if sales.empty:
    st.error("No eligible sales rows remain after cleaning. Check the dataset.")
    st.stop()

with st.sidebar:
    st.header("Analysis settings")
    k = st.slider("Customer clusters (K)", min_value=2, max_value=8, value=4)
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=60, value=30)

total_revenue = sales["Revenue"].sum()
customers = sales["CustomerID"].nunique()
orders = sales["InvoiceNo"].nunique()
products = sales["StockCode"].nunique()
a,b,c,d = st.columns(4)
a.metric("Net eligible sales", f"£{total_revenue:,.0f}")
b.metric("Customers", f"{customers:,}")
c.metric("Orders", f"{orders:,}")
d.metric("Products", f"{products:,}")

tabs = st.tabs(["Overview & EDA", "Customer Segmentation", "Recommendations", "Statistics", "Sales Forecast", "Data Quality"])
with tabs[0]:
    st.subheader("Sales and product overview")
    monthly = sales.set_index("InvoiceDate").resample("MS")["Revenue"].sum()
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(8,4))
        monthly.plot(ax=ax, marker="o")
        ax.set_title("Monthly revenue"); ax.set_ylabel("Revenue (£)"); ax.set_xlabel("")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with right:
        top = sales.groupby("Description")["Revenue"].sum().nlargest(10).sort_values()
        fig, ax = plt.subplots(figsize=(8,4))
        top.plot(kind="barh", ax=ax)
        ax.set_title("Top 10 products by revenue"); ax.set_xlabel("Revenue (£)")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    country = sales.groupby("Country")["Revenue"].sum().nlargest(10).sort_values()
    fig, ax = plt.subplots(figsize=(9,4))
    country.plot(kind="barh", ax=ax)
    ax.set_title("Top 10 countries by revenue"); ax.set_xlabel("Revenue (£)")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.dataframe(sales.head(20), use_container_width=True)

with tabs[1]:
    st.subheader("RFM customer segmentation using K-Means")
    rfm = make_rfm(sales)
    if len(rfm) <= k:
        st.warning("Not enough customers for this K value.")
    else:
        clustered, sil = make_clusters(rfm, k)

        # Profile each cluster using its actual RFM behaviour.
        summary = clustered.groupby("Cluster").agg(
            Customers=("CustomerID", "count"),
            MedianRecency=("Recency", "median"),
            MeanOrders=("Frequency", "mean"),
            MeanSpend=("Monetary", "mean"),
        ).reset_index()

        # Relative thresholds adapt to the selected number of clusters.
        spend_high = summary["MeanSpend"].quantile(0.75)
        orders_high = summary["MeanOrders"].quantile(0.75)
        orders_mid = summary["MeanOrders"].median()
        recency_low = summary["MedianRecency"].quantile(0.25)
        recency_high = summary["MedianRecency"].quantile(0.75)

        def name_segment(row):
            if row["MeanSpend"] >= spend_high and row["MeanOrders"] >= orders_mid:
                return "Champions / High Value"
            if row["MeanOrders"] >= orders_high:
                return "Loyal Customers"
            if row["MedianRecency"] >= recency_high:
                return "At Risk / Inactive"
            if row["MedianRecency"] <= recency_low and row["MeanOrders"] < orders_high:
                return "Recent / Promising"
            return "Occasional Customers"

        summary["SegmentName"] = summary.apply(name_segment, axis=1)
        name_lookup = summary.set_index("Cluster")["SegmentName"].to_dict()
        clustered["SegmentName"] = clustered["Cluster"].map(name_lookup)

        x, y = st.columns(2)
        x.metric("Silhouette score", f"{sil:.3f}" if pd.notna(sil) else "N/A")
        y.metric("Customers clustered", f"{len(clustered):,}")

        st.caption(
            "Segment names are assigned from relative cluster profiles. "
            "They are descriptive heuristics, not verified predictions of future behaviour."
        )
        st.dataframe(
            summary[["SegmentName", "Customers", "MedianRecency", "MeanOrders", "MeanSpend"]].round(2),
            use_container_width=True,
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.scatterplot(
            data=clustered, x="Frequency", y="Monetary",
            hue="SegmentName", palette="tab10", ax=ax,
        )
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.set_title("Customer clusters: purchase frequency vs spend")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.download_button("Download customer segments CSV", clustered.to_csv(index=False).encode("utf-8"), "customer_segments.csv", "text/csv")
        st.caption("Cluster numbers are arbitrary labels. Interpret segments using their profiles, not the label number.")

with tabs[2]:
    st.subheader("Item-based collaborative filtering")
    customer_ids = sorted(sales["CustomerID"].unique().tolist())
    chosen = st.selectbox("Choose a customer ID", customer_ids)
    top_n = st.slider("Number of recommendations", 5, 20, 10, key="rec_n")
    if st.button("Generate recommendations"):
        with st.spinner("Computing item similarities…"):
            recs = recommend_products(sales, chosen, top_n)
        if recs.empty:
            st.info("No recommendations available for this customer.")
        else:
            st.dataframe(recs.round(4), use_container_width=True)
            st.caption("Similarity scores rank candidate products; they are not probabilities of purchase.")

    st.divider()
    st.subheader("Recommendation model evaluation")
    st.write(
        "Tests whether products bought later appear in recommendations generated "
        "from earlier purchase history. Results depend on the time split and sample."
    )
    if st.button("Evaluate Recommendation Model"):
        with st.spinner("Evaluating recommendations on held-out purchases..."):
            evaluation = evaluate_recommender(sales, top_n=10, max_customers=100)

        if evaluation is None:
            st.warning("Not enough eligible purchase history for this evaluation.")
        else:
            e1, e2, e3 = st.columns(3)
            e1.metric("Customers evaluated", evaluation["customers_evaluated"])
            e2.metric("Precision@10", f"{evaluation['precision_at_k']:.3f}")
            e3.metric("Recall@10", f"{evaluation['recall_at_k']:.3f}")
            st.caption(
                f"Chronological split cutoff: {evaluation['cutoff_date']:%Y-%m-%d}. "
                "Precision@10 is the share of the 10 recommendation slots that matched "
                "later purchases; Recall@10 is the share of later-purchased products recovered. "
                "This is a simple offline evaluation, not a guarantee of future performance."
            )

with tabs[3]:
    st.subheader("Statistical analysis")
    st.write("Welch's t-test comparing order-level spend on weekdays vs weekends. This is descriptive observational data, not a randomized experiment.")
    order_data = sales.groupby(["InvoiceNo", "InvoiceDate"], as_index=False)["Revenue"].sum()
    order_data["DayType"] = np.where(order_data["InvoiceDate"].dt.dayofweek >= 5, "Weekend", "Weekday")
    weekday = order_data.loc[order_data["DayType"]=="Weekday","Revenue"]
    weekend = order_data.loc[order_data["DayType"]=="Weekend","Revenue"]
    l,r = st.columns(2)
    l.metric("Weekday mean order value", f"£{weekday.mean():,.2f}")
    r.metric("Weekend mean order value", f"£{weekend.mean():,.2f}")
    if len(weekday)>1 and len(weekend)>1:
        test = stats.ttest_ind(weekday, weekend, equal_var=False, nan_policy="omit")
        diff = weekday.mean() - weekend.mean()
        se = np.sqrt(weekday.var(ddof=1)/len(weekday) + weekend.var(ddof=1)/len(weekend))
        dof_num = (weekday.var(ddof=1)/len(weekday) + weekend.var(ddof=1)/len(weekend))**2
        dof_den = (weekday.var(ddof=1)/len(weekday))**2/(len(weekday)-1) + (weekend.var(ddof=1)/len(weekend))**2/(len(weekend)-1)
        dof = dof_num/dof_den if dof_den else np.nan
        crit = stats.t.ppf(0.975, dof) if np.isfinite(dof) else np.nan
        st.write(f"Welch t-statistic: **{test.statistic:.4f}**")
        st.write(f"Two-sided p-value: **{test.pvalue:.6g}**")
        st.write(f"95% CI for weekday minus weekend mean: **£{diff-crit*se:,.2f} to £{diff+crit*se:,.2f}**")
        st.caption("Interpret the p-value with effect size, assumptions, and business context. Multiple comparisons and dependence between orders can affect inference.")
    fig, ax = plt.subplots(figsize=(7,4))
    sns.boxplot(data=order_data, x="DayType", y="Revenue", showfliers=False, ax=ax)
    ax.set_title("Order value by day type"); ax.set_ylabel("Order value (£)")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

with tabs[4]:
    st.subheader("Daily sales forecasting")
    daily, backtest, results = daily_forecast(sales, horizon)
    if results is None:
        st.warning("Not enough usable time-series data or the forecasting model could not be fitted.")
    else:
        (
            mae, rmse, wape, smape, future,
            baseline_mae, baseline_rmse, selected_model, model_comparison
        ) = results
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Backtest MAE", f"£{mae:,.2f}")
        m2.metric("Backtest RMSE", f"£{rmse:,.2f}")
        m3.metric("Backtest WAPE", f"{wape:.2f}%" if np.isfinite(wape) else "N/A")
        m4.metric("Backtest sMAPE", f"{smape:.2f}%" if np.isfinite(smape) else "N/A")
        st.caption(
            f"Selected model: {selected_model} — lowest MAE on the chronological holdout."
        )
        st.subheader("Forecast model comparison")
        st.dataframe(
            model_comparison.round(2),
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(10,4))
        daily.tail(120).plot(ax=ax, label="Historical daily revenue")
        future.plot(ax=ax, label="Forecast", linestyle="--")
        if backtest:
            test, pred = backtest
            test.plot(ax=ax, label="Backtest actual")
            pred.index = test.index
            pred.plot(ax=ax, label="Backtest prediction", linestyle=":")
        ax.set_title("Daily revenue forecast"); ax.set_ylabel("Revenue (£)"); ax.legend()
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        forecast_df = future.rename("ForecastRevenue").reset_index()
        forecast_df.columns = ["Date","ForecastRevenue"]
        st.dataframe(forecast_df, use_container_width=True)
        st.download_button("Download forecast CSV", forecast_df.to_csv(index=False).encode("utf-8"), "sales_forecast.csv", "text/csv")
        st.subheader("Baseline comparison")
        b1, b2 = st.columns(2)
        b1.metric(
            "Naive baseline MAE", f"£{baseline_mae:,.2f}",
            delta=f"{baseline_mae - mae:,.2f} lower error for model" if mae < baseline_mae
            else f"{mae - baseline_mae:,.2f} higher error for model",
            delta_color="normal" if mae < baseline_mae else "inverse",
        )
        b2.metric(
            "Naive baseline RMSE", f"£{baseline_rmse:,.2f}",
            delta=f"{baseline_rmse - rmse:,.2f} lower error for model" if rmse < baseline_rmse
            else f"{rmse - baseline_rmse:,.2f} higher error for model",
            delta_color="normal" if rmse < baseline_rmse else "inverse",
        )
        if mae < baseline_mae and rmse < baseline_rmse:
            st.success(
                f"{selected_model} outperformed the naive baseline on both MAE and RMSE."
            )
        else:
            st.warning(
                "The model did not outperform the naive baseline on both metrics. "
                "Consider this a baseline result and investigate alternative forecasting methods."
            )
        st.caption(
            "The final chronological holdout (up to 30 days) is used for evaluation. "
            "The naive baseline repeats the last training-day revenue for every test day. "
            "MAE and RMSE are in revenue units; WAPE and sMAPE are percentages. "
            "Lower errors are better on the same holdout."
        )

with tabs[5]:
    st.subheader("Data quality audit")
    quality = pd.DataFrame({
        "Check": [
            "Rows in original uploaded dataset",
            "Exact duplicate rows before cleaning",
            "Rows after deduplication/required-field cleaning",
            "Cancellation invoice rows in cleaned raw data",
            "Rows with missing CustomerID in cleaned raw data",
            "Eligible positive sales rows",
            "Missing descriptions before cleaning",
            "Non-positive unit prices in cleaned raw data",
        ],
        "Count": [
            raw.attrs.get("source_row_count", "Unavailable"),
            raw.attrs.get("duplicate_rows_before_cleaning", "Unavailable"),
            len(raw),
            int(raw["IsCancellation"].sum()),
            int(raw["CustomerID"].isna().sum()),
            len(sales),
            raw.attrs.get("missing_descriptions_before_cleaning", "Unavailable"),
            int((raw["UnitPrice"] <= 0).sum()),
        ],
    })
    st.dataframe(quality, use_container_width=True)
    st.download_button("Download cleaned eligible sales CSV", sales.to_csv(index=False).encode("utf-8"), "retail_sales_clean.csv", "text/csv")
    st.info("Cleaning choices are explicit: exact duplicates removed; cancellations, returns, non-positive prices, and anonymous customers excluded from customer modelling. The original cleaned transaction table is retained for auditing.")
