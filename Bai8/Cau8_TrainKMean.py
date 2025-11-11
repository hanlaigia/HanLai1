import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
from connector import Connector

def build_customer_features():
    conn = Connector().connect()
    sql = """
    SELECT
        c.CustomerID,
        COUNT(DISTINCT o.OrderID) AS n_orders,
        SUM(od.OrderQty * od.UnitPrice) AS total_spend,
        MAX(STR_TO_DATE(o.OrderDate,'%d/%m/%Y')) AS last_order_date,
        COUNT(DISTINCT cat.CategoryID) AS n_categories_bought
    FROM customer c
    LEFT JOIN orders o ON o.CustomerID = c.CustomerID
    LEFT JOIN orderdetails od ON od.OrderID = o.OrderID
    LEFT JOIN product p ON p.ProductID = od.ProductID
    LEFT JOIN subcategory sc ON sc.SubCategoryID = p.ProductSubcategoryID
    LEFT JOIN category cat ON cat.CategoryID = sc.CategoryID
    GROUP BY c.CustomerID;
    """
    df = pd.read_sql(sql, conn)
    conn.close()
    df["last_order_date"] = pd.to_datetime(df["last_order_date"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    df["days_since_last_order"] = (today - df["last_order_date"]).dt.days.fillna(9999)
    df["avg_order_value"] = (df["total_spend"] / df["n_orders"]).fillna(0)
    df = df.fillna(0)
    return df

def train_and_save(k=4):
    df = build_customer_features()
    X = df[["n_orders","total_spend","avg_order_value","days_since_last_order","n_categories_bought"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    joblib.dump(model, "customer_cluster_model.pkl")
    joblib.dump(scaler, "customer_cluster_scaler.pkl")
    print("Model and scaler saved")

if __name__ == "__main__":
    train_and_save(k=4)
