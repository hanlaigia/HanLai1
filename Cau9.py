import pandas as pd
from sklearn.linear_model import LinearRegression
from connector import Connector

conn = Connector()
conn.connect()

def load(sql):
    return conn.queryDataset(sql)

sql_product = """
SELECT p.ProductID,
       YEAR(STR_TO_DATE(o.OrderDate,'%d/%m/%Y'))*12 
         + MONTH(STR_TO_DATE(o.OrderDate,'%d/%m/%Y')) AS TimeIndex,
       SUM(od.OrderQty) AS TotalQty
FROM orders o
JOIN orderdetails od ON od.OrderID = o.OrderID
JOIN product p ON p.ProductID = od.ProductID
GROUP BY p.ProductID, TimeIndex
ORDER BY TimeIndex;
"""

data_product = load(sql_product).fillna(0)

def predict_product(product_id):
    d = data_product[data_product["ProductID"] == product_id].copy()
    if d.empty:
        return None
    d["TotalQty"] = d["TotalQty"].fillna(0)
    X = d[["TimeIndex"]]
    y = d["TotalQty"]
    model = LinearRegression()
    model.fit(X, y)
    next_time = d["TimeIndex"].max() + 1
    next_df = pd.DataFrame({"TimeIndex":[next_time]})
    return model.predict(next_df)[0]

sql_category = """
SELECT c.CategoryID,
       YEAR(STR_TO_DATE(o.OrderDate,'%d/%m/%Y'))*12 
         + MONTH(STR_TO_DATE(o.OrderDate,'%d/%m/%Y')) AS TimeIndex,
       SUM(od.OrderQty) AS TotalQty
FROM orders o
JOIN orderdetails od ON od.OrderID = o.OrderID
JOIN product p ON p.ProductID = od.ProductID
JOIN subcategory sc ON sc.SubCategoryID = p.ProductSubcategoryID
JOIN category c ON c.CategoryID = sc.CategoryID
GROUP BY c.CategoryID, TimeIndex
ORDER BY TimeIndex;
"""

data_category = load(sql_category).fillna(0)

def predict_category(category_id):
    d = data_category[data_category["CategoryID"] == category_id].copy()
    if d.empty:
        return None
    d["TotalQty"] = d["TotalQty"].fillna(0)
    X = d[["TimeIndex"]]
    y = d["TotalQty"]
    model = LinearRegression()
    model.fit(X, y)
    next_time = d["TimeIndex"].max() + 1
    next_df = pd.DataFrame({"TimeIndex":[next_time]})
    return model.predict(next_df)[0]

product_id = 870
result_product = predict_product(product_id)
print("Du doan ProductID=", product_id, " thang tiep theo:", result_product)

category_id = 1
result_category = predict_category(category_id)
print("Du doan CategoryID=", category_id, " thang tiep theo:", result_category)

conn.disConnect()
