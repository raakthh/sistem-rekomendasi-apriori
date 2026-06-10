"""
/api/products — Katalog & pencarian produk
"""

from flask import Blueprint, request, jsonify
from retail_recommender.app.model_manager import get_model_state, get_preprocessor

products_bp = Blueprint("products", __name__)


def _check_ready():
    state = get_model_state()
    if not state["is_ready"]:
        return jsonify({"error": "Model belum siap", "model_ready": False}), 503
    return None


# ------------------------------------------------------------------ #
#  GET /api/products/                                                 #
# ------------------------------------------------------------------ #
@products_bp.route("/", methods=["GET"])
def list_products():
    """
    Daftar semua produk unik.

    Query params:
        page (int, default=1)
        per_page (int, default=50, max=200)
        q (str): keyword pencarian nama produk
    """
    err = _check_ready()
    if err:
        return err

    preprocessor = get_preprocessor()
    df = preprocessor.clean_df

    page = max(1, int(request.args.get("page", 1)))
    per_page = min(int(request.args.get("per_page", 50)), 200)
    query = request.args.get("q", "").upper().strip()

    products_df = (
        df.groupby("Description")
        .agg(
            total_sold=("Quantity", "sum"),
            transaction_count=("Invoice", "nunique"),
            avg_price=("Price", "mean"),
            stock_code=("StockCode", "first"),
        )
        .reset_index()
        .rename(columns={"Description": "name"})
    )

    if query:
        products_df = products_df[products_df["name"].str.contains(query, na=False)]

    products_df.sort_values("transaction_count", ascending=False, inplace=True)

    total = len(products_df)
    start = (page - 1) * per_page
    end = start + per_page
    page_data = products_df.iloc[start:end]

    products_list = []
    for _, row in page_data.iterrows():
        products_list.append(
            {
                "name": row["name"],
                "stock_code": row["stock_code"],
                "total_sold": int(row["total_sold"]),
                "transaction_count": int(row["transaction_count"]),
                "avg_price": round(float(row["avg_price"]), 2),
            }
        )

    return jsonify(
        {
            "products": products_list,
            "pagination": {
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": (total + per_page - 1) // per_page,
            },
        }
    )


# ------------------------------------------------------------------ #
#  GET /api/products/top                                              #
# ------------------------------------------------------------------ #
@products_bp.route("/top", methods=["GET"])
def top_products():
    """
    Produk terlaris berdasarkan jumlah transaksi.

    Query params:
        top_n (int, default=20)
    """
    err = _check_ready()
    if err:
        return err

    top_n = min(int(request.args.get("top_n", 20)), 100)
    preprocessor = get_preprocessor()
    df = preprocessor.clean_df

    top = (
        df.groupby("Description")
        .agg(
            transaction_count=("Invoice", "nunique"),
            total_sold=("Quantity", "sum"),
            avg_price=("Price", "mean"),
        )
        .reset_index()
        .rename(columns={"Description": "name"})
        .sort_values("transaction_count", ascending=False)
        .head(top_n)
    )

    return jsonify(
        {
            "top_products": [
                {
                    "rank": i + 1,
                    "name": row["name"],
                    "transaction_count": int(row["transaction_count"]),
                    "total_sold": int(row["total_sold"]),
                    "avg_price": round(float(row["avg_price"]), 2),
                }
                for i, (_, row) in enumerate(top.iterrows())
            ]
        }
    )


# ------------------------------------------------------------------ #
#  GET /api/products/search                                           #
# ------------------------------------------------------------------ #
@products_bp.route("/search", methods=["GET"])
def search_products():
    """
    Pencarian produk berdasarkan keyword.

    Query params:
        q (str, required): keyword
        limit (int, default=20)
    """
    err = _check_ready()
    if err:
        return err

    q = request.args.get("q", "").upper().strip()
    limit = min(int(request.args.get("limit", 20)), 100)

    if not q or len(q) < 2:
        return jsonify({"error": "Parameter 'q' minimal 2 karakter"}), 400

    preprocessor = get_preprocessor()
    df = preprocessor.clean_df

    matched = (
        df[df["Description"].str.contains(q, na=False)]
        .groupby("Description")["Invoice"]
        .nunique()
        .reset_index()
        .rename(columns={"Description": "name", "Invoice": "transaction_count"})
        .sort_values("transaction_count", ascending=False)
        .head(limit)
    )

    return jsonify(
        {
            "query": q,
            "results": matched.to_dict(orient="records"),
            "total": len(matched),
        }
    )
