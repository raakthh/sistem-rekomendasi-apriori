"""
Data Preprocessing Module
Handles: Load → Clean → Transform → Transaction Matrix
"""

import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Memproses dataset Online Retail II menjadi transaction matrix
    siap pakai untuk algoritma Apriori.
    """

    def __init__(self, filepath: str, sheet_name: str = "Year 2010-2011"):
        self.filepath = filepath
        self.sheet_name = sheet_name
        self.raw_df = None
        self.clean_df = None
        self.transaction_df = None
        self.product_catalog = {}

    # ------------------------------------------------------------------ #
    #  STEP 1 – Load Dataset                                               #
    # ------------------------------------------------------------------ #
    def load_data(self) -> pd.DataFrame:
        """Load dataset dari file Excel (semua sheet digabung jika perlu)."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset tidak ditemukan: {self.filepath}")

        logger.info(f"Memuat dataset dari {self.filepath} ...")

        if self.sheet_name == "all":
            sheets = pd.read_excel(self.filepath, sheet_name=None)
            self.raw_df = pd.concat(sheets.values(), ignore_index=True)
            logger.info(f"Semua sheet digabung → {len(self.raw_df):,} baris")
        else:
            self.raw_df = pd.read_excel(self.filepath, sheet_name=self.sheet_name)
            logger.info(f"Sheet '{self.sheet_name}' dimuat → {len(self.raw_df):,} baris")

        return self.raw_df

    # ------------------------------------------------------------------ #
    #  STEP 2 – Cleaning Data                                              #
    # ------------------------------------------------------------------ #
    def clean_data(self) -> pd.DataFrame:
        """
        Membersihkan data:
        - Hapus baris dengan Description / Customer ID null
        - Hapus transaksi cancel (Invoice diawali 'C')
        - Hapus Quantity dan Price ≤ 0
        - Normalisasi tipe data
        """
        if self.raw_df is None:
            raise RuntimeError("Panggil load_data() terlebih dahulu.")

        df = self.raw_df.copy()
        initial_count = len(df)
        logger.info(f"Data awal: {initial_count:,} baris")

        # Hapus null pada kolom kritis
        df.dropna(subset=["Description", "Customer ID"], inplace=True)
        logger.info(f"Setelah hapus null: {len(df):,} baris")

        # Hapus transaksi cancel (Invoice dimulai 'C')
        df = df[~df["Invoice"].astype(str).str.startswith("C")]
        logger.info(f"Setelah hapus cancel: {len(df):,} baris")

        # Hapus Quantity & Price tidak valid
        df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
        logger.info(f"Setelah filter qty/price: {len(df):,} baris")

        # Normalisasi kolom
        df["Description"] = df["Description"].str.strip().str.upper()
        df["Invoice"] = df["Invoice"].astype(str).str.strip()
        df["StockCode"] = df["StockCode"].astype(str).str.strip()
        df["Customer ID"] = df["Customer ID"].astype(str)

        removed = initial_count - len(df)
        logger.info(f"Total dihapus: {removed:,} baris ({removed/initial_count*100:.1f}%)")

        self.clean_df = df

        # Bangun katalog produk {StockCode: Description}
        self.product_catalog = (
            df.drop_duplicates("StockCode")
            .set_index("StockCode")["Description"]
            .to_dict()
        )

        return self.clean_df

    # ------------------------------------------------------------------ #
    #  STEP 3 – Transformasi ke Transaction Matrix                       #
    # ------------------------------------------------------------------ #
    def build_transaction_matrix(self, country: str = None) -> pd.DataFrame:
        """
        Ubah data baris per item menjadi one-hot encoded transaction matrix.
        Baris = Invoice, Kolom = Produk, Value = 0/1.

        Args:
            country: Filter per negara (opsional). None = semua negara.
        """
        if self.clean_df is None:
            raise RuntimeError("Panggil clean_data() terlebih dahulu.")

        df = self.clean_df.copy()

        if country:
            df = df[df["Country"].str.upper() == country.upper()]
            if df.empty:
                raise ValueError(f"Tidak ada data untuk negara: {country}")
            logger.info(f"Filter negara '{country}': {len(df):,} baris")

        # =========================
        # GROUP TRANSAKSI
        # =========================
        basket = (
            df.groupby(["Invoice", "Description"])["Quantity"]
            .sum()
            .unstack(fill_value=0)
        )

        # =========================
        # ONE HOT ENCODING (pandas 2.1+ → .map, bukan .applymap)
        # =========================
        basket = basket.map(lambda x: 1 if x > 0 else 0)

        logger.info(
            f"Sebelum filtering: {basket.shape[0]:,} transaksi × "
            f"{basket.shape[1]:,} produk"
        )

        # =========================
        # FILTER PRODUK
        # =========================
        item_counts = basket.sum()

        # Dataset 20k → threshold rendah
        min_transactions = 2

        popular_items = item_counts[item_counts >= min_transactions].index
        logger.info(f"Produk setelah threshold {min_transactions}: {len(popular_items):,}")

        # Batasi jumlah produk
        if len(popular_items) > 100:
            popular_items = item_counts.nlargest(100).index
            logger.info("Produk terlalu banyak, dibatasi menjadi top 100.")

        basket = basket[popular_items]

        # Fallback kalau kosong
        if basket.shape[1] == 0:
            logger.warning("Semua produk terfilter. Menggunakan top 100 produk.")
            popular_items = item_counts.nlargest(min(100, len(item_counts))).index
            basket = basket[popular_items]

        # Hemat memori
        basket = basket.astype(bool)

        logger.info(
            f"Transaction matrix: {basket.shape[0]:,} transaksi × "
            f"{basket.shape[1]:,} produk"
        )

        self.transaction_df = basket
        return self.transaction_df

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def get_product_name(self, stock_code: str) -> str:
        return self.product_catalog.get(stock_code, stock_code)

    def get_stats(self) -> dict:
        """Statistik ringkas dataset."""
        if self.clean_df is None:
            return {}
        df = self.clean_df
        return {
            "total_transactions": int(df["Invoice"].nunique()),
            "total_products": int(df["Description"].nunique()),
            "total_customers": int(df["Customer ID"].nunique()),
            "total_countries": int(df["Country"].nunique()),
            "date_range": {
                "start": str(df["InvoiceDate"].min()),
                "end": str(df["InvoiceDate"].max()),
            },
            "top_countries": df["Country"]
            .value_counts()
            .head(10)
            .to_dict(),
        }

    def run_pipeline(self, country: str = None):
        """Jalankan seluruh pipeline preprocessing sekaligus."""
        self.load_data()
        self.clean_data()
        self.build_transaction_matrix(country=country)
        return self