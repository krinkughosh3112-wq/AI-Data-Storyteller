# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from transformers import pipeline
import os
import shutil
import kaleido  # Needed for fig.write_image
import math

# -------------------------- Page Config --------------------------
st.set_page_config(page_title="AI Data Storyteller", layout="wide")

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-image: url("https://img.freepik.com/free-vector/gradient-technology-futuristic-background_23-2149122416.jpg");
    background-size: cover;
    background-attachment: fixed;
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: skyblue;
    text-align: center;
    text-shadow: 2px 2px 4px #000000;
    margin-bottom: 20px;
}
.block-container {
    background: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}
div[data-baseweb="tab-list"] {
    gap: 10px;
}
button[data-baseweb="tab"] {
    background-color: #f0f0f0;
    color: black;
    border-radius: 12px;
    padding: 8px 16px;
    font-weight: 600;
    border: 1px solid #ccc;
    transition: 0.3s;
}
button[data-baseweb="tab"]:hover {
    background-color: #e0e0e0;
    cursor: pointer;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #0078ff;
    color: white !important;
    border: 1px solid #0056b3;
}
</style>
""", unsafe_allow_html=True)

# -------------------------- Title --------------------------
st.markdown('<div class="title"> Smart Analytics Dashboard</div>', unsafe_allow_html=True)

# -------------------------- CSV Upload --------------------------
st.sidebar.header("👩‍💻 Upload CSV Dataset")
uploaded_file = st.sidebar.file_uploader("Choose your CSV file", type="csv")

# -------------------------- Global Vars --------------------------
cleaning_choice = "Not Applied"
outlier_summary = []
prediction_summary = "No prediction model trained."
ai_summary = ""

# Load lightweight LLM model
@st.cache_resource
def load_llm():
    return pipeline("text-generation", model="gpt2")

generator = load_llm()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    cat_cols = df.select_dtypes(include=['object']).columns
    filter_df = df.copy()

    # Sidebar filtering
    if len(cat_cols) > 0:
        filter_col = st.sidebar.selectbox("Filter by Column:", cat_cols)
        filter_val = st.sidebar.multiselect("Select Values:", df[filter_col].unique())
        if filter_val:
            filter_df = df[df[filter_col].isin(filter_val)]

    # -------------------------- KPI Cards --------------------------
    total_missing = filter_df.isnull().sum().sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", filter_df.shape[0])
    col2.metric("Total Columns", filter_df.shape[1])
    col3.metric("Missing Values", total_missing)

    # -------------------------- Tabs --------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Dataset", "🧾 EDA", "🎨 Visualizations",
        "📑 Report", "🚨 Outliers", "🤖 Prediction", "🗺️ Map"
    ])

    # -------------------------- Dataset Tab --------------------------
    with tab1:
        st.markdown("## 📋 Dataset Preview")
        st.info("ℹ️ This section shows the *first few rows* of your dataset.")
        st.dataframe(filter_df.head())

    # -------------------------- EDA Tab --------------------------
    with tab2:
        st.markdown("## 👩‍🔬 Exploratory Data Analysis")
        st.info(
            "ℹ️ Explore *dataset info, data types, summary statistics, and missing values*. "
            "You can also apply simple *data cleaning methods* here."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(" Dataset Info")
            st.write(f"Rows: {filter_df.shape[0]}, Columns: {filter_df.shape[1]}")
            st.write("Column Names:", list(filter_df.columns))
        with col2:
            st.subheader("📊 Column Data Types")
            st.write(filter_df.dtypes)

        st.subheader("🧮 Basic Statistics")
        st.write(filter_df.describe())
        st.subheader("❗ Missing Values")
        st.write(filter_df.isnull().sum())

        # Data Cleaning
        st.markdown("### 🧹 Data Cleaning")
        cleaning_option = st.radio(
            "Choose a missing value handling method:",
            ["Do Nothing", "Fill Numeric with Mean", "Fill Categorical with Mode", "Drop Missing Rows"]
        )
        if cleaning_option == "Fill Numeric with Mean":
            filter_df = filter_df.fillna(filter_df.mean(numeric_only=True))
            cleaning_choice = "Filled numeric columns with mean"
        elif cleaning_option == "Fill Categorical with Mode":
            for col in filter_df.select_dtypes(include='object').columns:
                filter_df[col].fillna(filter_df[col].mode()[0], inplace=True)
            cleaning_choice = "Filled categorical columns with mode"
        elif cleaning_option == "Drop Missing Rows":
            filter_df = filter_df.dropna()
            cleaning_choice = "Dropped rows with missing values"

        # AI Summary (Automated Data Storytelling)
        try:
            num_df = filter_df.select_dtypes(include=['int64','float64'])
            most_correlated_pair = ("N/A", "N/A", 0)
            if len(num_df.columns) > 1:
                corr_matrix = num_df.corr().abs().unstack()
                most_correlated_pair = corr_matrix.sort_values(ascending=False).drop_duplicates().index[1]
                corr_value = num_df.corr().loc[most_correlated_pair[0], most_correlated_pair[1]]
            
            prompt = (
                f"Analyze a dataset with {filter_df.shape[0]} rows, {filter_df.shape[1]} columns, and {total_missing} missing values. "
                f"The most correlated numeric columns are '{most_correlated_pair[0]}' and '{most_correlated_pair[1]}' with a correlation of {corr_value:.2f}. "
                "Write an engaging summary of these findings, along with any other notable observations."
            )
            
            summary_text = generator(prompt, max_length=200, do_sample=True, temperature=0.7)[0]['generated_text']
            ai_summary = summary_text
            st.markdown("### 🤖 AI Insight & Data Storytelling")
            st.info(summary_text)
        except Exception as e:
            st.warning(f"⚠️ AI summary could not be generated. Error: {e}")

    # -------------------------- Visualization Tab --------------------------
    with tab3:
        st.markdown("## 🎨 Visualizations")
        st.info("ℹ️ Create *interactive plots* to understand correlations, patterns, "
                "and distributions in your data.")

        num_df = filter_df.select_dtypes(include=['int64','float64'])
        if not num_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🔥 Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(6,5))
                sns.heatmap(num_df.corr(), annot=True, cmap="Set2", ax=ax)
                st.pyplot(fig)
            with col2:
                st.markdown(f"### 📊 Histogram of {num_df.columns[0]}")
                fig, ax = plt.subplots(figsize=(6,5))
                sns.histplot(num_df[num_df.columns[0]], kde=True, bins=20, ax=ax, color="teal")
                st.pyplot(fig)

            for i, col in enumerate(cat_cols):
                st.markdown(f"### 📊 Bar Chart - {col}")
                bar_data = filter_df[col].value_counts().reset_index()
                bar_data.columns = ['index', 'count']
                fig = px.bar(bar_data, x='index', y='count', color='count', color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True, key=f"bar{i}")

            # Compare Two Features
            st.markdown("### 🔄 Compare Two Features")
            st.info("ℹ️ The *Compare Two Features* option lets you explore relationships "
                    "between two variables. Example: *Age vs Salary* may show if older people earn more.")
            col_x = st.selectbox("Select X-axis feature:", num_df.columns, index=0)
            col_y = st.selectbox("Select Y-axis feature:", num_df.columns, index=1)
            color_col = None
            if len(cat_cols) > 0:
                color_col = st.selectbox("Optional color by category:", [None] + cat_cols.tolist())
            if col_x and col_y:
                fig = px.scatter(filter_df, x=col_x, y=col_y, color=color_col)
                st.plotly_chart(fig, use_container_width=True, key="scatter2")

    # -------------------------- Outlier Tab --------------------------
    with tab5:
        st.markdown("## 🚨 Outlier Detection (IQR Method)")
        st.info("ℹ️ This section detects *outliers in numeric columns* using the "
                "*Interquartile Range (IQR) method* and shows them in boxplots. It also performs *root cause analysis*.")

        for col in num_df.columns:
            Q1 = filter_df[col].quantile(0.25)
            Q3 = filter_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = filter_df[(filter_df[col] < Q1-1.5*IQR) | (filter_df[col] > Q3+1.5*IQR)]
            if not outliers.empty:
                summary_text = f"{len(outliers)} outliers detected in column {col}"
                outlier_summary.append(summary_text)
                st.warning(f"⚠️ {summary_text}")
                
                # Outlier Root Cause Analysis
                st.markdown("#### Root Cause Analysis")
                normal_data = filter_df[~filter_df.index.isin(outliers.index)]
                
                if not normal_data.empty:
                    col_rc1, col_rc2 = st.columns(2)
                    with col_rc1:
                        st.markdown("###### Average values for *Outliers*")
                        outlier_means = outliers[num_df.columns].mean()
                        st.dataframe(outlier_means.to_frame(), use_container_width=True)
                    with col_rc2:
                        st.markdown("###### Average values for *Normal Data*")
                        normal_means = normal_data[num_df.columns].mean()
                        st.dataframe(normal_means.to_frame(), use_container_width=True)
                    
                    st.info("Compare the average values of the numeric features to see which ones are most different for the outliers.")

            fig, ax = plt.subplots(figsize=(6,5))
            sns.boxplot(x=filter_df[col], ax=ax, color="salmon")
            st.pyplot(fig)

    # -------------------------- Prediction Tab --------------------------
    with tab6:
        st.markdown("## 🤖 Simple Prediction Model")
        st.info("ℹ️ Train a basic *Regression* (numeric target) or *Classification* "
                "(categorical target) model directly on your dataset. You can also perform *What-If* analysis.")
        target = st.selectbox("Select target column:", filter_df.columns)
        
        if target:
            try:
                X = pd.get_dummies(filter_df.drop(columns=[target]), drop_first=True)
                y = filter_df[target]
                
                if pd.api.types.is_numeric_dtype(y):
                    model = LinearRegression()
                    model_type = "Regression"
                else:
                    model = LogisticRegression(max_iter=1000)
                    y = pd.factorize(y)[0]
                    model_type = "Classification"
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                if model_type=="Regression":
                    mse = mean_squared_error(y_test, preds)
                    prediction_summary = f"{model_type} model trained. MSE={mse:.2f}"
                else:
                    acc = accuracy_score(y_test, preds)
                    prediction_summary = f"{model_type} model trained. Accuracy={acc:.2f}"
                
                st.success(f"✅ {prediction_summary}")

                # What-If Analysis
                st.markdown("### 🔮 What-If Analysis")
                st.info("ℹ️ Adjust the values below to see how they impact the prediction.")
                
                input_data = {}
                for col in filter_df.drop(columns=[target]).columns:
                    if pd.api.types.is_numeric_dtype(filter_df[col]):
                        min_val, max_val = float(filter_df[col].min()), float(filter_df[col].max())
                        input_data[col] = st.slider(f'Select {col}:', min_val, max_val, float(filter_df[col].mean()))
                    else:
                        input_data[col] = st.selectbox(f'Select {col}:', filter_df[col].unique())

                if st.button("Generate Prediction"):
                    input_df = pd.DataFrame([input_data])
                    input_df = pd.get_dummies(input_df, drop_first=True)
                    
                    missing_cols = set(X.columns) - set(input_df.columns)
                    for c in missing_cols:
                        input_df[c] = 0
                    
                    input_df = input_df[X.columns]
                    
                    prediction = model.predict(input_df)[0]
                    
                    if model_type == "Regression":
                        st.success(f"The predicted value is: *{prediction:.2f}*")
                    else:
                        predicted_class = df[target].unique()[prediction]
                        st.success(f"The predicted class is: *{predicted_class}*")
            
            except Exception as e:
                st.error(f"❌ An error occurred during model training or prediction: {e}")

    # -------------------------- PDF Report Tab --------------------------
    with tab4:
        st.markdown("## 📑 Generate Full PDF Report")
        st.info("ℹ️ Export a structured report with summary, insights, visualizations, outliers, and prediction.")

        def create_professional_pdf(df, cleaning_choice, outlier_summary, prediction_summary, ai_summary):
            pdf_file = "AI_Data_Storyteller_Professional_Report.pdf"
            c = canvas.Canvas(pdf_file, pagesize=letter)
            width, height = letter
            y_pos = height - 50
            line_height = 15

            def add_line(text, bold=False):
                nonlocal y_pos
                if y_pos < 100:  # New page
                    c.showPage()
                    y_pos = height - 50
                c.setFont("Helvetica-Bold" if bold else "Helvetica", 12)
                c.drawString(50, y_pos, text)
                y_pos -= line_height

            # Title
            c.setFont("Helvetica-Bold", 18)
            c.drawString(120, y_pos, "AI Data Storyteller Report")
            y_pos -= 30

            # Sections
            add_line("1. Executive Summary", bold=True)
            add_line(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
            add_line(f"Missing values: {df.isnull().sum().sum()}")
            if ai_summary:
                wrapped_summary = "\n".join([ai_summary[i:i+80] for i in range(0, len(ai_summary), 80)])
                for line in wrapped_summary.split('\n'):
                    add_line(line)
            y_pos -= line_height

            add_line("2. Introduction", bold=True)
            add_line("This report analyzes the uploaded dataset, providing insights, visualizations,")
            add_line("detects outliers, and trains a simple prediction model.")
            add_line(f"Columns: {', '.join(df.columns)}")

            add_line("3. Main Body", bold=True)
            add_line(f"Data Cleaning Applied: {cleaning_choice}")
            add_line(f"Column Types: {df.dtypes.to_dict()}")
            add_line(f"Missing Values: {df.isnull().sum().to_dict()}")

            # Plots
            plot_dir = "temp_plots"
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            numeric_cols = df.select_dtypes(include=['int64','float64']).columns
            categorical_cols = df.select_dtypes(include=['object']).columns

            # Correlation
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots(figsize=(6,5))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="Set2", ax=ax)
                path = f"{plot_dir}/correlation.png"
                plt.savefig(path); plt.close(fig)
                add_line("Correlation Heatmap:")
                c.drawImage(ImageReader(path), 50, y_pos-250, width=500, height=250)
                y_pos -= 270

            # Histogram
            if len(numeric_cols) >= 1:
                col = numeric_cols[0]
                fig, ax = plt.subplots(figsize=(5,4))
                sns.histplot(df[col], kde=True, bins=20, ax=ax, color="teal")
                path = f"{plot_dir}/hist.png"
                plt.savefig(path); plt.close(fig)
                add_line(f"Histogram of {col}:")
                c.drawImage(ImageReader(path), 50, y_pos-200, width=500, height=200)
                y_pos -= 220

            # Bar chart
            if len(categorical_cols) >= 1:
                col = categorical_cols[0]
                bar_data = df[col].value_counts().reset_index()
                bar_data.columns = ['index', 'count']
                fig = px.bar(bar_data, x='index', y='count', color='count', color_continuous_scale="Viridis")
                path = f"{plot_dir}/bar.png"
                fig.write_image(path, engine="kaleido")
                add_line(f"Bar Chart of {col}:")
                c.drawImage(ImageReader(path), 50, y_pos-200, width=500, height=200)
                y_pos -= 220

            # Outliers
            add_line("3.1 Outlier Detection", bold=True)
            for summary in outlier_summary:
                add_line(summary)

            # Prediction
            add_line("3.2 Prediction Model", bold=True)
            for line in prediction_summary.split('\n'):
                add_line(line[:100])

            # Conclusions
            add_line("4. Conclusions", bold=True)
            add_line("Analysis highlights key patterns, correlations, and outliers.")
            add_line("Insights from visualizations and models guide decisions.")

            # Recommendations
            add_line("5. Recommendations", bold=True)
            add_line("- Address missing values properly")
            add_line("- Explore feature relationships in detail")
            add_line("- Use predictive models for further analysis")

            # Appendices
            add_line("6. Appendices", bold=True)
            add_line("First 5 rows of the dataset:")
            for i in range(min(5, len(df))):
                row_text = ", ".join([str(x) for x in df.iloc[i].values])
                add_line(row_text[:100])

            c.save()
            shutil.rmtree(plot_dir, ignore_errors=True)
            return pdf_file

        if st.button("📥 Download Full Report"):
            pdf_path = create_professional_pdf(filter_df, cleaning_choice, outlier_summary, prediction_summary, ai_summary)
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Click to Download Report", f, file_name="Smart Analytics Dashboard.pdf")

    # -------------------------- Map Tab --------------------------
    with tab7:
        st.markdown("## 🗺️ Map Visualization")
        st.info("ℹ️ This section shows your dataset on a map if it contains latitude and longitude columns.")

        # Try to auto-detect latitude/longitude columns
        lat_cols = [col for col in filter_df.columns if 'lat' in col.lower()]
        lon_cols = [col for col in filter_df.columns if 'lon' in col.lower() or 'lng' in col.lower()]

        if lat_cols and lon_cols:
            lat_col = lat_cols[0]
            lon_col = lon_cols[0]
            fig = px.scatter_mapbox(
                filter_df,
                lat=lat_col,
                lon=lon_col,
                hover_name=filter_df.columns[0],
                zoom=3
            )
            fig.update_layout(mapbox_style='carto-positron')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🌍 No latitude/longitude columns detected in this dataset. If you want a map, please add 'lat' and 'lon' columns.")

else:
    st.info("📥 Please upload a CSV file to access the dashboard features.")
