import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error

df_global = None
conn = sqlite3.connect(":memory:")

def upload_file(file):
    global df_global, conn
    content = file.read().decode("utf-8")
    df_global = pd.read_csv(StringIO(content))
    conn = sqlite3.connect(":memory:")
    df_global.to_sql("data", conn, if_exists="replace", index=False)
    return df_global.head().to_string()

def preview_data():
    if df_global is None:
        return "Please upload a dataset first."
    return df_global.head().to_string()

def show_kpis():
    if df_global is None:
        return "Please upload a dataset first."
    rows, cols = df_global.shape
    missing = df_global.isnull().sum().sum()
    return f"✅ Rows: {rows}\n✅ Columns: {cols}\n⚠️ Missing Values: {missing}"

def plot_data(x_col, y_col):
    if df_global is None:
        return None
    plt.figure(figsize=(6,4))
    sns.barplot(x=df_global[x_col], y=df_global[y_col])
    plt.xticks(rotation=45)
    return plt.gcf()

def run_sql(query):
    if df_global is None:
        return "Please upload a dataset first."
    try:
        result = pd.read_sql_query(query, conn)
        return result
    except Exception as e:
        return str(e)

def run_classification(target_col):
    if df_global is None:
        return "Upload data first."
    try:
        X = df_global.dropna()
        y = X[target_col]
        X = pd.get_dummies(X.drop(columns=[target_col]), drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        return f"✅ Logistic Regression Accuracy: {acc:.2f}"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def run_regression(target_col):
    if df_global is None:
        return "Upload data first."
    try:
        X = df_global.dropna()
        y = X[target_col]
        X = pd.get_dummies(X.drop(columns=[target_col]), drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        return f"✅ Linear Regression MSE: {mse:.2f}"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def run_clustering(n_clusters):
    if df_global is None:
        return "Upload data first."
    try:
        X = df_global.dropna()
        X_enc = pd.get_dummies(X, drop_first=True)
        model = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
        clusters = model.fit_predict(X_enc)
        X['Cluster'] = clusters
        return X.head()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("## 📊 Data Explorer + AI/ML Playground")
    with gr.Tab("Upload"):
        file = gr.File(label="Upload CSV", file_types=[".csv"])
        output_upload = gr.Textbox(label="Preview (Head)", interactive=False)
        file.upload(upload_file, inputs=file, outputs=output_upload)
    with gr.Tab("Preview Data"):
        btn_preview = gr.Button("Show Data Preview")
        output_preview = gr.Textbox()
        btn_preview.click(preview_data, outputs=output_preview)
    with gr.Tab("KPIs"):
        btn_kpi = gr.Button("Show KPIs")
        output_kpi = gr.Textbox()
        btn_kpi.click(show_kpis, outputs=output_kpi)
    with gr.Tab("Plot Data"):
        xcol = gr.Textbox(label="X column")
        ycol = gr.Textbox(label="Y column")
        plot_btn = gr.Button("Plot")
        plot_output = gr.Plot()
        plot_btn.click(plot_data, inputs=[xcol, ycol], outputs=plot_output)
    with gr.Tab("SQL Query"):
        query = gr.Textbox(label="SQL Query")
        sql_btn = gr.Button("Run Query")
        sql_output = gr.Dataframe()
        sql_btn.click(run_sql, inputs=query, outputs=sql_output)
    with gr.Tab("AI / ML"):
        gr.Markdown("### 🔮 Choose a Task")
        target_class = gr.Textbox(label="Target Column for Classification")
        class_btn = gr.Button("Run Logistic Regression")
        class_out = gr.Textbox()
        class_btn.click(run_classification, inputs=target_class, outputs=class_out)
        target_reg = gr.Textbox(label="Target Column for Regression")
        reg_btn = gr.Button("Run Linear Regression")
        reg_out = gr.Textbox()
        reg_btn.click(run_regression, inputs=target_reg, outputs=reg_out)
        nclus = gr.Number(label="Number of Clusters")
        clus_btn = gr.Button("Run KMeans Clustering")
        clus_out = gr.Dataframe()
        clus_btn.click(run_clustering, inputs=nclus, outputs=clus_out)

demo.launch(share=True)
