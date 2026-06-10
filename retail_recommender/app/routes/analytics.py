"""
/api/analytics — Statistik & insight untuk dashboard marketplace
"""

from flask import Blueprint, request, jsonify
from retail_recommender.app.model_manager import get_model_state, get_preprocessor, get_engine

analytics_bp = Blueprint("analytics", __name__)


def _check_ready():
    state = get_model_state()
    if not state["is_ready"]:
        return jsonify({"error": "Model belum siap", "model_ready": False}), 503
    return None


@analytics_bp.route("/summary", methods=["GET"])
def summary():
    """Ringkasan dataset dan model."""
    err = _check_ready()
    if err:
        return err

    preprocessor = get_preprocessor()
    engine = get_engine()

    return jsonify(
        {
            "dataset": preprocessor.get_stats(),
            "model": engine.get_summary(),
        }
    )


@analytics_bp.route("/sales-by-country", methods=["GET"])
def sales_by_country():
    """Penjualan per negara."""
    err = _check_ready()
    if err:
        return err

    top_n = min(int(request.args.get("top_n", 15)), 50)
    preprocessor = get_preprocessor()
    df = preprocessor.clean_df

    result = (
        df.groupby("Country")
        .agg(
            transaction_count=("Invoice", "nunique"),
            total_revenue=("Price", lambda x: (x * df.loc[x.index, "Quantity"]).sum()),
            customer_count=("Customer ID", "nunique"),
        )
        .reset_index()
        .sort_values("transaction_count", ascending=False)
        .head(top_n)
    )

    return jsonify(
        {
            "sales_by_country": [
                {
                    "country": row["Country"],
                    "transaction_count": int(row["transaction_count"]),
                    "total_revenue": round(float(row["total_revenue"]), 2),
                    "customer_count": int(row["customer_count"]),
                }
                for _, row in result.iterrows()
            ]
        }
    )


@analytics_bp.route("/sales-trend", methods=["GET"])
def sales_trend():
    """Tren penjualan bulanan."""
    err = _check_ready()
    if err:
        return err

    preprocessor = get_preprocessor()
    df = preprocessor.clean_df.copy()
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    trend = (
        df.groupby("Month")
        .agg(
            transaction_count=("Invoice", "nunique"),
            revenue=("Price", lambda x: (x * df.loc[x.index, "Quantity"]).sum()),
        )
        .reset_index()
        .sort_values("Month")
    )

    return jsonify(
        {
            "monthly_trend": [
                {
                    "month": row["Month"],
                    "transaction_count": int(row["transaction_count"]),
                    "revenue": round(float(row["revenue"]), 2),
                }
                for _, row in trend.iterrows()
            ]
        }
    )


@analytics_bp.route("/itemset-distribution", methods=["GET"])
def itemset_distribution():
    """Distribusi ukuran frequent itemset."""
    err = _check_ready()
    if err:
        return err

    engine = get_engine()
    if engine.frequent_itemsets is None:
        return jsonify({"error": "Frequent itemsets belum tersedia"}), 404

    dist = (
        engine.frequent_itemsets["length"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return jsonify(
        {
            "distribution": [
                {"itemset_size": int(k), "count": int(v)}
                for k, v in dist.items()
            ]
        }
    )
