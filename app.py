import sys
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QComboBox, QTableWidget, QTableWidgetItem
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import mysql.connector

class Connector:
    def __init__(self,server="localhost", port=3306, database="retails", username="root", password="@Obama123"):
        self.server=server
        self.port=port
        self.database=database
        self.username=username
        self.password=password
        self.conn=None
    def connect(self):
        try:
            self.conn = mysql.connector.connect(host=self.server,port=self.port,database=self.database,user=self.username,password=self.password)
            return self.conn
        except:
            self.conn=None
            traceback.print_exc()
        return None

def df_to_table(tbl: QTableWidget, df: pd.DataFrame):
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

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Retails Manager")
        self.resize(1200, 800)
        self.cn = Connector().connect()
        self.scaler = None
        self.km = None
        self.tabs = QTabWidget()
        self.tabs.addTab(self.tab_stats(), "Thống kê")
        self.tabs.addTab(self.tab_customer(), "Khách hàng")
        self.tabs.addTab(self.tab_cluster(), "Phân cụm")
        self.tabs.addTab(self.tab_forecast(), "Dự báo")
        lay = QVBoxLayout(self)
        lay.addWidget(self.tabs)

    def tab_stats(self):
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        b2 = QPushButton("2) Doanh số theo sản phẩm")
        b3 = QPushButton("3) Doanh thu theo danh mục")
        b4 = QPushButton("4) Doanh thu danh mục theo Tháng+Năm")
        b5 = QPushButton("5) Đơn giao sớm ≥3 ngày")
        row.addWidget(b2); row.addWidget(b3); row.addWidget(b4); row.addWidget(b5)
        v.addLayout(row)
        self.tbl_stats = QTableWidget()
        v.addWidget(self.tbl_stats)

        def run2():
            sql = """
            SELECT
                p.ProductID,
                p.Name AS ProductName,
                SUM(od.OrderQty * od.UnitPrice) AS TotalSales
            FROM orderdetails od
            JOIN product p ON p.ProductID = od.ProductID
            GROUP BY p.ProductID, p.Name
            ORDER BY TotalSales DESC;
            """
            df = self._q(sql)
            df_to_table(self.tbl_stats, df)

        def run3():
            sql = """
            SELECT
                c.CategoryID,
                c.Name AS CategoryName,
                SUM(od.OrderQty * od.UnitPrice) AS Revenue
            FROM orderdetails od
            JOIN product p ON p.ProductID = od.ProductID
            JOIN subcategory sc ON sc.SubCategoryID = p.ProductSubcategoryID
            JOIN category c ON c.CategoryID = sc.CategoryID
            GROUP BY c.CategoryID, c.Name
            ORDER BY Revenue DESC;
            """
            df = self._q(sql)
            df_to_table(self.tbl_stats, df)

        def run4():
            sql = """
            SELECT
                c.CategoryID,
                c.Name AS CategoryName,
                YEAR(STR_TO_DATE(o.OrderDate,'%d/%m/%Y')) AS Year,
                MONTH(STR_TO_DATE(o.OrderDate,'%d/%m/%Y')) AS Month,
                SUM(od.OrderQty * od.UnitPrice) AS Revenue
            FROM orders o
            JOIN orderdetails od ON od.OrderID = o.OrderID
            JOIN product p ON p.ProductID = od.ProductID
            JOIN subcategory sc ON sc.SubCategoryID = p.ProductSubcategoryID
            JOIN category c ON c.CategoryID = sc.CategoryID
            GROUP BY c.CategoryID, c.Name, YEAR(STR_TO_DATE(o.OrderDate,'%d/%m/%Y')), MONTH(STR_TO_DATE(o.OrderDate,'%d/%m/%Y'))
            ORDER BY c.Name, Year, Month;
            """
            df = self._q(sql)
            df_to_table(self.tbl_stats, df)

        def run5():
            sql = """
            SELECT
                o.OrderID,
                o.CustomerID,
                STR_TO_DATE(o.OrderDate,'%d/%m/%Y') AS OrderDate,
                STR_TO_DATE(o.DueDate,'%d/%m/%Y') AS DueDate,
                STR_TO_DATE(o.ShipDate,'%d/%m/%Y') AS ShipDate,
                DATEDIFF(STR_TO_DATE(o.DueDate,'%d/%m/%Y'), STR_TO_DATE(o.ShipDate,'%d/%m/%Y')) AS DaysEarly
            FROM orders o
            WHERE o.ShipDate IS NOT NULL
              AND o.DueDate IS NOT NULL
              AND DATEDIFF(STR_TO_DATE(o.DueDate,'%d/%m/%Y'), STR_TO_DATE(o.ShipDate,'%d/%m/%Y')) >= 3
            ORDER BY DaysEarly DESC;
            """
            df = self._q(sql)
            df_to_table(self.tbl_stats, df)

        b2.clicked.connect(run2)
        b3.clicked.connect(run3)
        b4.clicked.connect(run4)
        b5.clicked.connect(run5)
        return w

    def tab_customer(self):
        w = QWidget()
        v = QVBoxLayout(w)
        r = QHBoxLayout()
        r.addWidget(QLabel("CustomerID"))
        self.inp_cid = QLineEdit()
        btn = QPushButton("Tải")
        r.addWidget(self.inp_cid); r.addWidget(btn)
        v.addLayout(r)
        self.tbl_cust = QTableWidget()
        self.tbl_orders = QTableWidget()
        v.addWidget(QLabel("Thông tin"))
        v.addWidget(self.tbl_cust)
        v.addWidget(QLabel("Đơn hàng"))
        v.addWidget(self.tbl_orders)

        def load():
            cid = self.inp_cid.text().strip()
            if not cid:
                df_to_table(self.tbl_cust, pd.DataFrame())
                df_to_table(self.tbl_orders, pd.DataFrame())
                return
            df1 = self._fa("SELECT * FROM customer WHERE CustomerID=%s", (cid,))
            df2 = self._fa("""
                SELECT
                  o.OrderID,
                  o.OrderDate,
                  o.DueDate,
                  o.ShipDate,
                  SUM(od.OrderQty * od.UnitPrice) AS OrderTotal
                FROM orders o
                JOIN orderdetails od ON od.OrderID = o.OrderID
                WHERE o.CustomerID = %s
                GROUP BY o.OrderID, o.OrderDate, o.DueDate, o.ShipDate
                ORDER BY STR_TO_DATE(o.OrderDate,'%d/%m/%Y') DESC
            """, (cid,))
            df_to_table(self.tbl_cust, df1)
            df_to_table(self.tbl_orders, df2)

        btn.clicked.connect(load)
        return w

    def tab_cluster(self):
        w = QWidget()
        v = QVBoxLayout(w)
        r = QHBoxLayout()
        self.cbo_k = QComboBox()
        self.cbo_k.addItems([str(x) for x in (3,4,5,6)])
        train = QPushButton("Huấn luyện KMeans")
        r.addWidget(QLabel("Số cụm (k)"))
        r.addWidget(self.cbo_k)
        r.addWidget(train)
        v.addLayout(r)
        r2 = QHBoxLayout()
        self.cbo_cluster = QComboBox()
        showc = QPushButton("Xem cụm")
        r2.addWidget(QLabel("Cụm"))
        r2.addWidget(self.cbo_cluster)
        r2.addWidget(showc)
        v.addLayout(r2)
        self.tbl_cluster = QTableWidget()
        v.addWidget(self.tbl_cluster)

        def build_features():
            sql_main = """
            SELECT
              c.CustomerID,
              COUNT(DISTINCT o.OrderID) AS n_orders,
              SUM(od.OrderQty * od.UnitPrice) AS total_spend,
              MAX(STR_TO_DATE(o.OrderDate,'%d/%m/%Y')) AS last_order_date
            FROM customer c
            LEFT JOIN orders o ON o.CustomerID = c.CustomerID
            LEFT JOIN orderdetails od ON od.OrderID = o.OrderID
            GROUP BY c.CustomerID;
            """
            sql_cat = """
            SELECT
              o.CustomerID,
              COUNT(DISTINCT sc.CategoryID) AS n_categories_bought
            FROM orders o
            JOIN orderdetails od ON od.OrderID = o.OrderID
            JOIN product p ON p.ProductID = od.ProductID
            JOIN subcategory sc ON sc.SubCategoryID = p.ProductSubcategoryID
            GROUP BY o.CustomerID;
            """
            df = self._q(sql_main)
            d2 = self._q(sql_cat)
            df = df.merge(d2, how="left", left_on="CustomerID", right_on="CustomerID")
            df["n_categories_bought"] = df["n_categories_bought"].fillna(0)
            df["last_order_date"] = pd.to_datetime(df["last_order_date"])
            today = pd.Timestamp(datetime.today().date())
            df["days_since_last_order"] = (today - df["last_order_date"]).dt.days
            df["days_since_last_order"] = df["days_since_last_order"].fillna(9999)
            df["n_orders"] = df["n_orders"].fillna(0)
            df["total_spend"] = df["total_spend"].fillna(0.0)
            df["avg_order_value"] = (df["total_spend"] / df["n_orders"]).replace([np.inf, -np.inf], 0).fillna(0.0)
            return df

        def do_train():
            k = int(self.cbo_k.currentText())
            feats = build_features()
            X = feats[["n_orders","total_spend","avg_order_value","days_since_last_order","n_categories_bought"]].astype(float)
            self.scaler = StandardScaler()
            Xs = self.scaler.fit_transform(X)
            self.km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            labels = self.km.fit_predict(Xs)
            feats["cluster"] = labels
            self._cluster_base = feats
            clusters = sorted(map(int, pd.Series(labels).unique()))
            self.cbo_cluster.clear()
            self.cbo_cluster.addItems([str(x) for x in clusters])
            summary = feats.groupby("cluster")["CustomerID"].count().reset_index(name="n_customers").sort_values("cluster")
            df_to_table(self.tbl_cluster, summary)

        def show_cluster():
            if self.km is None:
                return
            cid = int(self.cbo_cluster.currentText())
            df = self._cluster_base.loc[self._cluster_base["cluster"]==cid, ["CustomerID","n_orders","total_spend","avg_order_value","n_categories_bought","days_since_last_order"]].sort_values("total_spend", ascending=False)
            df_to_table(self.tbl_cluster, df)

        train.clicked.connect(do_train)
        showc.clicked.connect(show_cluster)
        return w

    def tab_forecast(self):
        w = QWidget()
        v = QVBoxLayout(w)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("CategoryID"))
        self.inp_cat = QLineEdit()
        btn_cat = QPushButton("Dự báo 6 tháng")
        r1.addWidget(self.inp_cat); r1.addWidget(btn_cat)
        v.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("ProductID"))
        self.inp_prod = QLineEdit()
        btn_prod = QPushButton("Dự báo 6 tháng")
        r2.addWidget(self.inp_prod); r2.addWidget(btn_prod)
        v.addLayout(r2)
        self.tbl_hist = QTableWidget()
        self.tbl_fore = QTableWidget()
        v.addWidget(QLabel("Lịch sử"))
        v.addWidget(self.tbl_hist)
        v.addWidget(QLabel("Dự báo"))
        v.addWidget(self.tbl_fore)

        def series_by_category(cid):
            sql = """
            SELECT
              YEAR(STR_TO_DATE(o.OrderDate,'%d/%m/%Y')) AS y,
              MONTH(STR_TO_DATE(o.OrderDate,'%d/%m/%Y')) AS m,
              SUM(od.OrderQty * od.UnitPrice) AS revenue
            FROM orders o
            JOIN orderdetails od ON od.OrderID = o.OrderID
            JOIN product p ON p.ProductID = od.ProductID
            JOIN subcategory sc ON sc.SubCategoryID = p.ProductSubcategoryID
            WHERE sc.CategoryID = %s
            GROUP BY y,m
            ORDER BY y,m;
            """
            df = self._fa(sql, (cid,))
            if df.empty:
                return df
            df["date"] = pd.to_datetime(pd.DataFrame({"year":df["y"].astype(int),"month":df["m"].astype(int),"day":1}))
            return df[["date","revenue"]].sort_values("date").reset_index(drop=True)

        def series_by_product(pid):
            sql = """
            SELECT
              YEAR(STR_TO_DATE(o.OrderDate,'%d/%m/%Y')) AS y,
              MONTH(STR_TO_DATE(o.OrderDate,'%d/%m/%Y')) AS m,
              SUM(od.OrderQty * od.UnitPrice) AS revenue
            FROM orders o
            JOIN orderdetails od ON od.OrderID = o.OrderID
            JOIN product p ON p.ProductID = od.ProductID
            WHERE p.ProductID = %s
            GROUP BY y,m
            ORDER BY y,m;
            """
            df = self._fa(sql, (pid,))
            if df.empty:
                return df
            df["date"] = pd.to_datetime(pd.DataFrame({"year":df["y"].astype(int),"month":df["m"].astype(int),"day":1}))
            return df[["date","revenue"]].sort_values("date").reset_index(drop=True)

        def fit_lr(df, horizon=6):
            if df.empty:
                return df, pd.DataFrame()
            base = df.copy()
            base["t"] = np.arange(1, len(base)+1, dtype=float)
            base["month"] = base["date"].dt.month
            M = pd.get_dummies(base["month"], prefix="m", drop_first=True)
            X = pd.concat([base[["t"]], M], axis=1)
            y = base["revenue"].values
            model = LinearRegression().fit(X, y)
            last_date = base["date"].iloc[-1]
            fut_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
            fut = pd.DataFrame({"date": fut_dates})
            fut["t"] = np.arange(len(base)+1, len(base)+1+horizon, dtype=float)
            fut["month"] = fut["date"].dt.month
            M2 = pd.get_dummies(fut["month"], prefix="m", drop_first=True)
            for col in M.columns:
                if col not in M2.columns:
                    M2[col] = 0
            M2 = M2.reindex(columns=M.columns, fill_value=0)
            Xf = pd.concat([fut[["t"]], M2], axis=1)
            fut["forecast"] = model.predict(Xf)
            return base, fut

        def forecast_cat():
            val = self.inp_cat.text().strip()
            if not val:
                df_to_table(self.tbl_hist, pd.DataFrame())
                df_to_table(self.tbl_fore, pd.DataFrame())
                return
            hist = series_by_category(int(val))
            if hist.empty:
                df_to_table(self.tbl_hist, pd.DataFrame([["Không có dữ liệu"]], columns=["Thông báo"]))
                df_to_table(self.tbl_fore, pd.DataFrame())
                return
            base, fut = fit_lr(hist)
            df_to_table(self.tbl_hist, base)
            df_to_table(self.tbl_fore, fut)

        def forecast_prod():
            val = self.inp_prod.text().strip()
            if not val:
                df_to_table(self.tbl_hist, pd.DataFrame())
                df_to_table(self.tbl_fore, pd.DataFrame())
                return
            hist = series_by_product(int(val))
            if hist.empty:
                df_to_table(self.tbl_hist, pd.DataFrame([["Không có dữ liệu"]], columns=["Thông báo"]))
                df_to_table(self.tbl_fore, pd.DataFrame())
                return
            base, fut = fit_lr(hist)
            df_to_table(self.tbl_hist, base)
            df_to_table(self.tbl_fore, fut)

        btn_cat.clicked.connect(forecast_cat)
        btn_prod.clicked.connect(forecast_prod)
        return w

    def _q(self, sql):
        try:
            cur = self.cn.cursor()
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
            cur = self.cn.cursor()
            cur.execute(sql, val_tuple)
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
            cur.close()
            return pd.DataFrame(rows, columns=cols)
        except:
            traceback.print_exc()
            return pd.DataFrame()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    m = App()
    m.show()
    sys.exit(app.exec())
