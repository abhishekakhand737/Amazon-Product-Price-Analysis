import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(
    page_title="PricePulse | Amazon Price Intelligence",
    page_icon=":shopping_trolley:",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_DIR = Path(__file__).parent
DATA_CANDIDATES = [
    APP_DIR / "data" / "amazon.csv",
    APP_DIR / "amazon.csv",
    Path.cwd() / "data" / "amazon.csv",
    Path.cwd() / "amazon.csv",
]
COLORWAY = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = COLORWAY


st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.5rem; }
    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        background: #ffffff;
        min-height: 112px;
    }
    .metric-label { color: #64748b; font-size: 0.86rem; margin-bottom: 6px; }
    .metric-value { color: #0f172a; font-size: 1.55rem; font-weight: 750; }
    .metric-help { color: #64748b; font-size: 0.78rem; margin-top: 6px; }
    .section-note {
        border-left: 4px solid #2563eb;
        background: #f8fafc;
        padding: 12px 14px;
        border-radius: 6px;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def money(value: float) -> str:
    if pd.isna(value):
        return "Rs 0"
    return f"Rs {value:,.0f}"


def clean_money(value) -> float:
    if pd.isna(value):
        return np.nan
    cleaned = re.sub(r"[^0-9.]", "", str(value))
    return float(cleaned) if cleaned else np.nan


def clean_percent(value) -> float:
    if pd.isna(value):
        return np.nan
    cleaned = re.sub(r"[^0-9.]", "", str(value))
    return float(cleaned) if cleaned else np.nan


def clean_count(value) -> float:
    if pd.isna(value):
        return np.nan
    cleaned = re.sub(r"[^0-9]", "", str(value))
    return float(cleaned) if cleaned else np.nan


def short_product_name(name: str, limit: int = 58) -> str:
    name = str(name)
    return name if len(name) <= limit else f"{name[:limit].rstrip()}..."


@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = next((path for path in DATA_CANDIDATES if path.exists()), None)
    if data_path is None:
        st.error(
            "Dataset file not found. Please upload amazon.csv either inside "
            "data/amazon.csv or next to app.py in your GitHub repository."
        )
        st.stop()

    df = pd.read_csv(data_path)
    df["actual_price_num"] = df["actual_price"].apply(clean_money)
    df["discounted_price_num"] = df["discounted_price"].apply(clean_money)
    df["discount_percentage_num"] = df["discount_percentage"].apply(clean_percent)
    df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_count_num"] = df["rating_count"].apply(clean_count)
    df["main_category"] = df["category"].fillna("Unknown").str.split("|").str[0]
    df["sub_category"] = df["category"].fillna("Unknown").str.split("|").str[1].fillna("Unknown")
    df["product_short"] = df["product_name"].apply(short_product_name)
    df["savings"] = df["actual_price_num"] - df["discounted_price_num"]
    df["value_score"] = (
        df["rating_num"].fillna(df["rating_num"].median()) * 20
        + df["discount_percentage_num"].fillna(0)
        + np.log1p(df["rating_count_num"].fillna(0))
    )
    df = df.dropna(subset=["actual_price_num", "discounted_price_num", "rating_num"])
    df["rating_count_num"] = df["rating_count_num"].fillna(0)
    df["discount_percentage_num"] = df["discount_percentage_num"].fillna(0)
    return df


@st.cache_resource
def train_price_model(df: pd.DataFrame):
    model_df = df[
        [
            "main_category",
            "sub_category",
            "actual_price_num",
            "discount_percentage_num",
            "rating_num",
            "rating_count_num",
            "discounted_price_num",
        ]
    ].dropna()

    X = model_df.drop(columns=["discounted_price_num"])
    y = model_df["discounted_price_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical = ["main_category", "sub_category"]
    numeric = ["actual_price_num", "discount_percentage_num", "rating_num", "rating_count_num"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=220,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "rows": len(model_df),
    }
    return model, metrics


def metric_card(label: str, value: str, help_text: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str):
    st.title(title)
    st.caption(subtitle)


df = load_data()

with st.sidebar:
    st.title("PricePulse")
    st.caption("Amazon Product Price Intelligence")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Executive Summary",
            "Category Analysis",
            "Product Insights",
            "Trends & Correlations",
            "Price Predictor",
            "Data Explorer",
        ],
    )

    st.divider()
    st.subheader("Global Filters")
    selected_categories = st.multiselect(
        "Filter by Category",
        sorted(df["main_category"].dropna().unique()),
        default=sorted(df["main_category"].dropna().unique()),
    )
    selected_subcategories = st.multiselect(
        "Filter by Subcategory",
        sorted(df["sub_category"].dropna().unique()),
        default=[],
        help="Leave empty to include all subcategories.",
    )

    min_price = int(df["discounted_price_num"].min())
    max_price = int(df["discounted_price_num"].max())
    price_range = st.slider(
        "Discounted Price Range",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=50,
    )

    rating_range = st.slider("Rating Range", 0.0, 5.0, (0.0, 5.0), 0.1)

category_mask = df["main_category"].isin(selected_categories)
subcategory_mask = (
    df["sub_category"].isin(selected_subcategories)
    if selected_subcategories
    else pd.Series(True, index=df.index)
)

filtered_df = df[
    category_mask
    & subcategory_mask
    & df["discounted_price_num"].between(price_range[0], price_range[1])
    & df["rating_num"].between(rating_range[0], rating_range[1])
].copy()

if filtered_df.empty:
    st.warning("No products match the selected filters. Adjust the sidebar filters.")
    st.stop()

with st.sidebar:
    st.divider()
    st.subheader("Quick Stats")
    st.metric("Products", f"{len(filtered_df):,}")
    st.metric("Categories", filtered_df["main_category"].nunique())
    st.metric("Avg Price", money(filtered_df["discounted_price_num"].mean()))


if page == "Executive Summary":
    page_header(
        "Amazon Product Price Intelligence Report",
        "A complete analytical overview of Amazon product pricing, discounts, ratings, and review activity.",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Products", f"{len(filtered_df):,}", "Filtered product records")
    with col2:
        metric_card("Average Price", money(filtered_df["discounted_price_num"].mean()), "Mean discounted price")
    with col3:
        metric_card("Average Discount", f"{filtered_df['discount_percentage_num'].mean():.1f}%", "Across selected products")
    with col4:
        metric_card("Average Rating", f"{filtered_df['rating_num'].mean():.2f}/5", "Customer score")

    st.markdown("### Key Market Insights")
    top_category = filtered_df["main_category"].value_counts().idxmax()
    best_discount = filtered_df.loc[filtered_df["discount_percentage_num"].idxmax()]
    most_reviewed = filtered_df.loc[filtered_df["rating_count_num"].idxmax()]
    st.markdown(
        f"""
        <div class="section-note">
        <b>Market Overview:</b> The largest category in the selected view is <b>{top_category}</b>.
        The strongest observed discount is <b>{best_discount['discount_percentage_num']:.0f}%</b>
        on <b>{best_discount['product_short']}</b>. The most reviewed product is
        <b>{most_reviewed['product_short']}</b> with <b>{most_reviewed['rating_count_num']:,.0f}</b> ratings.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        category_price = (
            filtered_df.groupby("main_category", as_index=False)["discounted_price_num"]
            .mean()
            .sort_values("discounted_price_num", ascending=False)
        )
        fig = px.bar(
            category_price,
            x="main_category",
            y="discounted_price_num",
            title="Average Discounted Price by Category",
            labels={"main_category": "Category", "discounted_price_num": "Average price"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            filtered_df,
            x="discounted_price_num",
            nbins=45,
            title="Discounted Price Distribution",
            labels={"discounted_price_num": "Discounted price"},
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Minimum Price", money(filtered_df["discounted_price_num"].min()))
    with col2:
        st.metric("Median Price", money(filtered_df["discounted_price_num"].median()))
    with col3:
        st.metric("Maximum Price", money(filtered_df["discounted_price_num"].max()))


elif page == "Category Analysis":
    page_header(
        "Category Analysis",
        "Compare category-level prices, discounts, product volume, ratings, and customer attention.",
    )

    category_summary = (
        filtered_df.groupby("main_category")
        .agg(
            products=("product_id", "count"),
            avg_price=("discounted_price_num", "mean"),
            avg_actual_price=("actual_price_num", "mean"),
            avg_discount=("discount_percentage_num", "mean"),
            avg_rating=("rating_num", "mean"),
            total_reviews=("rating_count_num", "sum"),
        )
        .reset_index()
        .sort_values("products", ascending=False)
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            category_summary,
            x="main_category",
            y="products",
            title="Product Count by Category",
            labels={"main_category": "Category", "products": "Products"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(
            category_summary.sort_values("avg_discount", ascending=False),
            x="main_category",
            y="avg_discount",
            title="Average Discount by Category",
            labels={"main_category": "Category", "avg_discount": "Average discount (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            filtered_df,
            x="main_category",
            y="discounted_price_num",
            color="main_category",
            title="Price Spread by Category",
            labels={"main_category": "Category", "discounted_price_num": "Discounted price"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(
            category_summary,
            x="avg_price",
            y="avg_rating",
            size="total_reviews",
            color="main_category",
            title="Category Value Map",
            labels={"avg_price": "Average price", "avg_rating": "Average rating"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Category Statistics")
    st.dataframe(
        category_summary.style.format(
            {
                "avg_price": "Rs {:,.0f}",
                "avg_actual_price": "Rs {:,.0f}",
                "avg_discount": "{:.1f}%",
                "avg_rating": "{:.2f}",
                "total_reviews": "{:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


elif page == "Product Insights":
    page_header(
        "Product Insights",
        "Identify premium products, best-rated products, high-discount deals, and review leaders.",
    )

    top_n = st.slider("Products to display", 5, 25, 10)
    tabs = st.tabs(["Most Expensive", "Highest Discount", "Best Rated", "Most Reviewed"])

    product_columns = [
        "product_short",
        "main_category",
        "actual_price_num",
        "discounted_price_num",
        "discount_percentage_num",
        "rating_num",
        "rating_count_num",
    ]

    with tabs[0]:
        view = filtered_df.nlargest(top_n, "discounted_price_num")[product_columns]
        fig = px.bar(
            view.sort_values("discounted_price_num"),
            x="discounted_price_num",
            y="product_short",
            orientation="h",
            title="Most Expensive Products",
            labels={"discounted_price_num": "Discounted price", "product_short": "Product"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(view, use_container_width=True, hide_index=True)

    with tabs[1]:
        view = filtered_df.nlargest(top_n, "discount_percentage_num")[product_columns]
        fig = px.bar(
            view.sort_values("discount_percentage_num"),
            x="discount_percentage_num",
            y="product_short",
            orientation="h",
            title="Highest Discount Products",
            labels={"discount_percentage_num": "Discount (%)", "product_short": "Product"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(view, use_container_width=True, hide_index=True)

    with tabs[2]:
        qualified = filtered_df[filtered_df["rating_count_num"] >= 100]
        view = qualified.nlargest(top_n, ["rating_num", "rating_count_num"])[product_columns]
        fig = px.bar(
            view.sort_values("rating_num"),
            x="rating_num",
            y="product_short",
            orientation="h",
            title="Best Rated Products with 100+ Reviews",
            labels={"rating_num": "Rating", "product_short": "Product"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(view, use_container_width=True, hide_index=True)

    with tabs[3]:
        view = filtered_df.nlargest(top_n, "rating_count_num")[product_columns]
        fig = px.bar(
            view.sort_values("rating_count_num"),
            x="rating_count_num",
            y="product_short",
            orientation="h",
            title="Most Reviewed Products",
            labels={"rating_count_num": "Review count", "product_short": "Product"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(view, use_container_width=True, hide_index=True)


elif page == "Trends & Correlations":
    page_header(
        "Trends & Correlations",
        "Explore relationships between prices, discounts, ratings, review volume, and product value.",
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            filtered_df,
            x="actual_price_num",
            y="discounted_price_num",
            color="main_category",
            size="rating_count_num",
            hover_name="product_short",
            title="Actual Price vs Discounted Price",
            labels={"actual_price_num": "Actual price", "discounted_price_num": "Discounted price"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            filtered_df,
            x="discount_percentage_num",
            y="rating_num",
            color="main_category",
            size="rating_count_num",
            hover_name="product_short",
            title="Discount vs Rating",
            labels={"discount_percentage_num": "Discount (%)", "rating_num": "Rating"},
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        heatmap_data = pd.pivot_table(
            filtered_df,
            values="discounted_price_num",
            index="main_category",
            columns="sub_category",
            aggfunc="mean",
        )
        fig = px.imshow(
            heatmap_data,
            title="Average Price Heatmap: Category x Subcategory",
            labels={"color": "Avg price"},
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        corr_cols = [
            "actual_price_num",
            "discounted_price_num",
            "discount_percentage_num",
            "rating_num",
            "rating_count_num",
            "savings",
            "value_score",
        ]
        corr = filtered_df[corr_cols].corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            title="Correlation Matrix",
            labels={"color": "Correlation"},
            aspect="auto",
            color_continuous_scale="RdBu_r",
        )
        st.plotly_chart(fig, use_container_width=True)


elif page == "Price Predictor":
    page_header(
        "ML Price Predictor",
        "Estimate a product's discounted price using category, subcategory, actual price, discount, rating, and review volume.",
    )

    model, model_metrics = train_price_model(df)
    st.markdown(
        f"""
        <div class="section-note">
        Random Forest model trained on <b>{model_metrics['rows']:,}</b> products.
        Test MAE: <b>{money(model_metrics['mae'])}</b> | R2 score: <b>{model_metrics['r2']:.3f}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Configure Product Parameters")
    col1, col2 = st.columns(2)
    with col1:
        pred_category = st.selectbox("Category", sorted(df["main_category"].unique()))
        sub_options = sorted(df.loc[df["main_category"] == pred_category, "sub_category"].unique())
        pred_subcategory = st.selectbox("Subcategory", sub_options)
        pred_actual_price = st.number_input(
            "Actual Price",
            min_value=50,
            max_value=250000,
            value=1999,
            step=100,
        )
    with col2:
        pred_discount = st.slider("Discount Percentage", 0, 95, 35)
        pred_rating = st.slider("Rating", 1.0, 5.0, 4.1, 0.1)
        pred_reviews = st.number_input(
            "Rating Count",
            min_value=0,
            max_value=500000,
            value=2500,
            step=100,
        )

    if st.button("Generate Price Prediction", type="primary"):
        input_df = pd.DataFrame(
            [
                {
                    "main_category": pred_category,
                    "sub_category": pred_subcategory,
                    "actual_price_num": pred_actual_price,
                    "discount_percentage_num": pred_discount,
                    "rating_num": pred_rating,
                    "rating_count_num": pred_reviews,
                }
            ]
        )
        predicted_price = float(model.predict(input_df)[0])
        expected_discount_price = pred_actual_price * (1 - pred_discount / 100)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Discounted Price", money(predicted_price))
        with col2:
            st.metric("Rule-Based Discount Price", money(expected_discount_price))
        with col3:
            st.metric("Estimated Savings", money(max(pred_actual_price - predicted_price, 0)))

        fig = go.Figure(
            data=[
                go.Bar(
                    x=["Actual Price", "Discount Formula", "ML Prediction"],
                    y=[pred_actual_price, expected_discount_price, predicted_price],
                    marker_color=["#64748b", "#f59e0b", "#2563eb"],
                )
            ]
        )
        fig.update_layout(title="Price Estimate Comparison", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)


elif page == "Data Explorer":
    page_header(
        "Data Explorer",
        "Browse, search, sort, and download the Amazon product dataset.",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(filtered_df):,}")
    with col2:
        st.metric("Columns", len(filtered_df.columns))
    with col3:
        st.metric("Missing Values", int(filtered_df.isna().sum().sum()))
    with col4:
        st.metric("Categories", filtered_df["main_category"].nunique())

    st.subheader("Search & Filter")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        query = st.text_input("Search in product name or review title")
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            [
                "discounted_price_num",
                "actual_price_num",
                "discount_percentage_num",
                "rating_num",
                "rating_count_num",
                "value_score",
            ],
        )
    with col3:
        descending = st.checkbox("Descending order", value=True)

    explorer_df = filtered_df.copy()
    if query:
        mask = (
            explorer_df["product_name"].str.contains(query, case=False, na=False)
            | explorer_df["review_title"].str.contains(query, case=False, na=False)
        )
        explorer_df = explorer_df[mask]

    explorer_df = explorer_df.sort_values(sort_by, ascending=not descending)

    display_cols = [
        "product_name",
        "main_category",
        "sub_category",
        "actual_price_num",
        "discounted_price_num",
        "discount_percentage_num",
        "rating_num",
        "rating_count_num",
        "product_link",
    ]
    st.dataframe(
        explorer_df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "product_link": st.column_config.LinkColumn("Product Link"),
            "actual_price_num": st.column_config.NumberColumn("Actual Price", format="Rs %.0f"),
            "discounted_price_num": st.column_config.NumberColumn("Discounted Price", format="Rs %.0f"),
            "discount_percentage_num": st.column_config.NumberColumn("Discount %", format="%.1f%%"),
            "rating_num": st.column_config.NumberColumn("Rating", format="%.1f"),
            "rating_count_num": st.column_config.NumberColumn("Rating Count", format="%.0f"),
        },
    )

    csv = explorer_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered Data (CSV)",
        data=csv,
        file_name="amazon_filtered_products.csv",
        mime="text/csv",
    )


st.caption("PricePulse Analytics | Amazon Product Price Intelligence Platform | Built with Streamlit, Plotly and scikit-learn")
