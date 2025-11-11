import sys
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QWidget, QTableWidgetItem
from PyQt6 import QtWidgets
from RetailsManager import Ui_App

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from connector import Connector
from Cau9 import data_product, data_category


def df_to_table(tbl, df: pd.DataFrame):
    tbl.clear()
    if df is None or df.empty:
        tbl.setRowCount(0)
        tbl.setColumnCount(0)
        return
    tbl.setRowCount(len(df))
    tbl.setColumnCount(len(df.columns))
    tbl.setHorizontalHeaderLabels([str(c) for c in df.columns])
    for r in range(len(df)):
        for c in range(len(df.columns)):
            tbl.setItem(r, c, QTableWidgetItem(str(df.iat[r, c])))


class App(QWidget, Ui_App):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.conn = Connector().connect()
        self.scaler = None
        self.km = None
        self._cluster_base = None

        # ==============================
        # THỐNG KÊ
        # ==============================
        self.btn_stat_2.clicked.connect(self.stat_2)
        self.btn_stat_3.clicked.connect(self.stat_3)
        self.btn_stat_4.clicked.connect(self.stat_4)
        self.btn_stat_5.clicked.connect(self.stat_5)

        # ==============================
        # KHÁCH HÀNG
        # ==============================
        self.btn_load.clicked.connect(self.load_customer)

        # ==============================
        # PHÂN CỤM
        # ==============================
        self.btn_train.clicked.connect(self.train_cluster)
        self.btn_showc.clicked.connect(self.show_cluster)

        # ==============================
        # DỰ BÁO 1 NĂM (12 tháng)
        # ==============================
        self.btn_cat.clicked.connect(self.forecast_cat)
        self.btn_prod.clicked.connect(self.forecast_prod)

    # ====================== DB QUERY ======================
    def _q(self, sql):
        try:
            cur = self.conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
            cur.close()
            return pd.DataFrame(rows, columns=cols)
        except:
            traceback.print_exc()
            return pd.DataFrame()

    def _fa(self, sql, val_tuple):
        try:
            cur = self.conn.cursor()
            cur.execute(sql, val_tuple)
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
            cur.close()
            return pd.DataFrame(rows, columns=cols)
        except:
            traceback.print_exc()
            return pd.DataFrame()

    # ====================== THỐNG KÊ ======================
    def stat_2(self):
        sql = """
        SELECT p.ProductID,p.Name AS ProductName,
               SUM(od.OrderQty * od.UnitPrice) AS TotalSales
        FROM orderdetails od
        JOIN product p ON p.ProductID = od.ProductID
        GROUP BY p.ProductID, p.Name
        ORDER BY TotalSales DESC;
        """
        df_to_table(self.tbl_stats, self._q(sql))

    def stat_3(self):
        sql = """
        SELECT c.CategoryID,c.Name AS CategoryName,
               SUM(od.OrderQty * od.UnitPrice) AS Revenue
        FROM orderdetails od
        JOIN product p ON p.ProductID = od.ProductID
        JOIN subcategory sc ON sc.SubCategoryID = p.ProductSubcategoryID
        JOIN category c ON c.CategoryID = sc.CategoryID
        GROUP BY c.CategoryID, c.Name
        ORDER BY Revenue DESC;
        """
        df_to_table(self.tbl_stats, self._q(sql))

    def stat_4(self):
        sql = """
        SELECT c.CategoryID,c.Name AS CategoryName,
               YEAR(STR_TO_DATE(o.OrderDate,'%d/%m/%Y')) AS Year,
               MONTH(STR_TO_DATE(o.OrderDate,'%d/%m/%Y')) AS Month,
               SUM(od.OrderQty * od.UnitPrice) AS Revenue
        FROM orders o
        JOIN orderdetails od ON od.OrderID = o.OrderID
        JOIN product p ON p.ProductID = od.ProductID
        JOIN subcategory sc ON sc.SubCategoryID = p.ProductSubcategoryID
        JOIN category c ON c.CategoryID = sc.CategoryID
        GROUP BY c.CategoryID, c.Name,Year,Month
        ORDER BY Year,Month;
        """
        df_to_table(self.tbl_stats, self._q(sql))

    def stat_5(self):
        sql = """
        SELECT o.OrderID,o.CustomerID,
               o.OrderDate,o.DueDate,o.ShipDate,
               DATEDIFF(STR_TO_DATE(o.DueDate,'%d/%m/%Y'),
                        STR_TO_DATE(o.ShipDate,'%d/%m/%Y')) AS DaysEarly
        FROM orders o
        WHERE DATEDIFF(STR_TO_DATE(o.DueDate,'%d/%m/%Y'),
                       STR_TO_DATE(o.ShipDate,'%d/%m/%Y')) >= 3;
        """
        df_to_table(self.tbl_stats, self._q(sql))

    # ====================== KHÁCH HÀNG ======================
    def load_customer(self):
        cid = self.inp_cid.text().strip()
        if not cid:
            return
        df_to_table(self.tbl_cust, self._fa("SELECT * FROM customer WHERE CustomerID=%s", (cid,)))
        df_to_table(self.tbl_orders, self._fa("""
            SELECT o.OrderID,o.OrderDate,o.DueDate,o.ShipDate,
                   SUM(od.OrderQty * od.UnitPrice) AS OrderTotal
            FROM orders o
            JOIN orderdetails od ON od.OrderID = o.OrderID
            WHERE o.CustomerID=%s
            GROUP BY o.OrderID
            ORDER BY STR_TO_DATE(o.OrderDate,'%d/%m/%Y') DESC
        """, (cid,)))

    # ====================== PHÂN CỤM ======================
    def train_cluster(self):
        sql_main = """
        SELECT c.CustomerID,
               COUNT(DISTINCT o.OrderID) AS n_orders,
               SUM(od.OrderQty * od.UnitPrice) AS total_spend,
               MAX(STR_TO_DATE(o.OrderDate,'%d/%m/%Y')) AS last_order_date
        FROM customer c
        LEFT JOIN orders o ON o.CustomerID=c.CustomerID
        LEFT JOIN orderdetails od ON od.OrderID=o.OrderID
        GROUP BY c.CustomerID;
        """
        sql_cat = """
        SELECT o.CustomerID,
               COUNT(DISTINCT sc.CategoryID) AS n_categories_bought
        FROM orders o
        JOIN orderdetails od ON od.OrderID=o.OrderID
        JOIN product p ON p.ProductID=od.ProductID
        JOIN subcategory sc ON sc.SubCategoryID=p.ProductSubcategoryID
        GROUP BY o.CustomerID;
        """
        df = self._q(sql_main).merge(self._q(sql_cat), how="left", on="CustomerID").fillna(0)
        df["last_order_date"] = pd.to_datetime(df["last_order_date"])
        df["days_since_last_order"] = (pd.Timestamp.today() - df["last_order_date"]).dt.days.fillna(9999)
        df["avg_order_value"] = (df["total_spend"]/df["n_orders"]).replace([np.inf,-np.inf],0)

        X = df[["n_orders","total_spend","avg_order_value","days_since_last_order","n_categories_bought"]]
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        k = int(self.cbo_k.currentText())
        self.km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        df["cluster"] = self.km.fit_predict(Xs)
        self._cluster_base = df
        df_to_table(self.tbl_cluster, df.groupby("cluster")["CustomerID"].count().reset_index(name="n_customers"))
        self.cbo_cluster.clear()
        self.cbo_cluster.addItems([str(i) for i in df["cluster"].unique()])

    def show_cluster(self):
        if self._cluster_base is None:
            return
        cid = int(self.cbo_cluster.currentText())
        df_to_table(self.tbl_cluster, self._cluster_base[self._cluster_base["cluster"] == cid])

    # ====================== DỰ BÁO 12 THÁNG ======================
    def forecast_cat(self):
        val = self.inp_cat.text().strip()
        if not val:
            return
        cid = int(val)
        hist = data_category[data_category["CategoryID"] == cid].copy()
        if hist.empty:
            df_to_table(self.tbl_hist, pd.DataFrame([["Không có dữ liệu"]], columns=["Thông báo"]))
            return

        # Hiển thị lịch sử
        df_to_table(self.tbl_hist, hist)

        # Train mô hình
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(hist[["TimeIndex"]], hist["TotalQty"])

        # Dự báo 12 tháng
        last = hist["TimeIndex"].max()
        fut = pd.DataFrame({"TimeIndex": range(last+1, last+13)})
        fut["ForecastQty"] = model.predict(fut[["TimeIndex"]])

        df_to_table(self.tbl_fore, fut)

    def forecast_prod(self):
        val = self.inp_prod.text().strip()
        if not val:
            return
        pid = int(val)
        hist = data_product[data_product["ProductID"] == pid].copy()
        if hist.empty:
            df_to_table(self.tbl_hist, pd.DataFrame([["Không có dữ liệu"]], columns=["Thông báo"]))
            return

        # Hiển thị lịch sử
        df_to_table(self.tbl_hist, hist)

        # Train mô hình
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(hist[["TimeIndex"]], hist["TotalQty"])

        # Dự báo 12 tháng
        last = hist["TimeIndex"].max()
        fut = pd.DataFrame({"TimeIndex": range(last+1, last+13)})
        fut["ForecastQty"] = model.predict(fut[["TimeIndex"]])

        df_to_table(self.tbl_fore, fut)

