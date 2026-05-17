import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import urllib.parse
import numpy as np
from sqlalchemy import create_engine, text

from data_processing import clean_data, preprocess_data, get_train_test_split
from model import ChurnModeling
from agent import AIBusinessAnalyst


st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1E3A8A; font-family: 'Inter', sans-serif; }
    .stButton>button {
        background-color: #2563EB; color: white; border-radius: 8px; border: none; padding: 0.5rem 1rem; transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1D4ED8; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def get_sqlserver_engine(server, database,table, password):
    try:
        conn_str = (
          #  f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"PASSWORD = {password};"
            f"Trusted_Connection=yes;"
            f"TrustServerCertificate=yes;"
        )
        params = urllib.parse.quote_plus(conn_str)
        # engine = create_engine(f"mysql+pymysql://root:helloworld@123@127.0.0.1:3306/churn_db")
        engine = create_engine(f"mysql+pymysql://root:helloworld@127.0.0.1:3306/customer_data")
        return engine
    except Exception as e:
        return str(e)

def load_data_from_sqlserver(server, database,table, password):
    try:
        engine = get_sqlserver_engine(server, database,table, password)

        if isinstance(engine, str):
            return engine

        with engine.connection() as conn:
            conn.execute(text("SELECT 1"))

        query = text(f"SELECT * FROM {table}")
        df = pd.read_sql(query, engine)
        return df

    except Exception as e:
        return f"SQL Server error: {str(e)}"


if 'data' not in st.session_state:
    st.session_state.data = None
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = None


st.sidebar.title("🔌Database & API Configuration")

st.sidebar.subheader("🤖 Groq API Key")
api_key_input = st.sidebar.text_input("Enter Groq API Key:", type="password", key="api_key_input")
if api_key_input:
    st.session_state.groq_api_key = api_key_input
    os.environ["GROQ_API_KEY"] = api_key_input
    st.sidebar.success("✅ Groq API Key configured!")
else:
    st.sidebar.warning("⚠️ Please enter your Groq API Key to enable AI features")


with st.sidebar.form("db_form"):
    db_host = st.text_input("SQL Server Host", "localhost\\SQLEXPRESS")
    db_name = st.text_input("Database Name", "")   # default fixed
    db_table = st.text_input("Table Name", "")
    db_pass = st.text_input("Password", "")  

    connect_btn = st.form_submit_button("Connect & Load Data")


if connect_btn:
    with st.spinner("Processing..."):
        try:
            safe_host = db_host.strip()
            safe_db = db_name.strip()
            safe_table = db_table.strip()
            safe_pass = db_pass.strip()

            if not safe_host:
                st.sidebar.error("❌ SQL Server Host is required.")
                st.stop()

            if not safe_db:
                st.sidebar.error("❌ Database Name is required.")
                st.stop()

            if not safe_table:
                st.sidebar.error("❌ Table Name is required.")
                st.stop()

            if not safe_pass:
                st.sidebar.error("❌ Password is required.")
                st.stop()


        
            engine = get_sqlserver_engine(safe_host, safe_db,safe_table, safe_pass)

            if isinstance(engine, str):
                st.sidebar.error(f"❌ Engine creation failed: {engine}")
                st.stop()

            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                st.sidebar.success("✅ SQL Server connection successful!")
            except Exception as e:
                st.sidebar.error(f"❌ Connection failed: {e}")
                st.stop()
        
            result = load_data_from_sqlserver(safe_host, safe_db, safe_table,safe_pass)

            if isinstance(result, pd.DataFrame):
                st.session_state.data = result
                st.sidebar.success(f"✅ Loaded {len(result)} rows from database!")
            else:
                st.sidebar.error(f"❌ Connection failed: {result}")

        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.header("📋 Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard (EDA)", "Model Insights", "Customer Predictor", "AI Analyst"])


if st.session_state.data is None:
    st.info("👈 Please connect to your SQL Server database in the sidebar to load data. You can also upload a CSV.")
    st.stop()


def check_groq_api():
    if st.session_state.groq_api_key is None or not st.session_state.groq_api_key.strip():
        st.warning("⚠️ **Groq API Key required** for AI Analyst features. Please configure it in the sidebar.")
        return False
    return True


@st.cache_resource
def train_and_get_models(data):
    clean_df = clean_data(data)
    X, y, encoders, scaler = preprocess_data(clean_df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    modeling = ChurnModeling()
    modeling.train_models(X_train, y_train)
    return modeling, X_test, y_test, encoders, scaler, X_train

df = clean_data(st.session_state.data)

try:
    modeling, X_test, y_test, encoders, scaler, X_train = train_and_get_models(df)
except Exception as e:
    st.error(f"❌ Error training models: {e}. Ensure dataset has 'Churn' column and features.")
    st.stop()


if page == "Dashboard (EDA)":
    st.title("📊 Customer Churn Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    total_customers = len(df)
    churn_rate = (df['Customer_Status'] == 'Churned').mean() * 100 if 'Customer_Status' in df.columns else 0
    avg_lifetime = df['Tenure_in_Months'].mean() if 'Tenure_in_Months' in df.columns else 0
    monthly_rev = df['Monthly_Charge'].sum() if 'Monthly_Charge' in df.columns else 0

    col1.markdown(f"<div class='metric-card'><h3 style='color:#2F2F2F'>Total Customers</h3><h2 style='color:#3A81F6'>{total_customers:,}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h3 style='color:#2F2F2F'>Churn Rate</h3><h2 style='color:#EE4434'>{churn_rate:.1f}%</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><h3 style='color:#2F2F2F'>Avg Tenure</h3><h2 style='color:#10A941'>{avg_lifetime:.1f}</h2></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><h3 style='color:#2F2F2F'>MRR</h3><h2 style='color:#8B5CFF'>${monthly_rev:,.0f}</h2></div>", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        if 'Contract' in df.columns and 'Customer_Status' in df.columns:
            st.subheader("Churn by Contract Type")
            contract_churn = df.groupby(['Contract', 'Customer_Status']).size().reset_index(name='Count')
            fig = px.bar(contract_churn, x="Contract", y="Count", color="Customer_Status", barmode="group",
                         color_discrete_map={"Churned": "#EF4444", "Stayed": "#10B981"})
            fig.update_traces(marker=dict(line=dict(color='black', width=2)))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Missing Contract or Churn columns for chart.")

    with c2:
        if 'Monthly_Charge' in df.columns and 'Customer_Status' in df.columns:
            st.subheader("Monthly Charges Distribution by Churn")
            fig2 = px.box(df, x="Customer_Status", y="Monthly_Charge", color="Customer_Status",
                          color_discrete_map={"Churned": "#EF4444", "Stayed": "#10B981"})
            st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    if 'Internet_Service' in df.columns and 'Total_Charges' in df.columns:
        grp_by_internet = (df.groupby('Internet_Service')['Total_Charges']
                           .mean()
                           .sort_values(ascending=False)
                           .head(5)
                           .reset_index())

        fig3 = px.bar(grp_by_internet, x="Internet_Service", y="Total_Charges",
                      title="Top 5 Internet Service by Avg Total Charges")
        fig3.update_traces(marker_color='#EF4444', marker=dict(line=dict(color='black', width=3)))
        c3.plotly_chart(fig3, use_container_width=True)

    if 'Total_Charges' in df.columns:
        fig4 = px.histogram(df, x="Total_Charges", nbins=30, marginal="box",
                            title="Distribution of Total Charges")
        fig4.update_traces(marker_color='#10B981', marker=dict(line=dict(color='black', width=3)))
        c4.plotly_chart(fig4, use_container_width=True)

    c5, c6 = st.columns(2)

    if 'Monthly_Charge' in df.columns:
        fig5 = px.histogram(df, x="Monthly_Charge", nbins=30, marginal="box",
                            title="Distribution of Monthly Charges")
        fig5.update_traces(marker_color='#10B981', marker=dict(line=dict(color='black', width=3)))
        c5.plotly_chart(fig5, use_container_width=True)

    corr = df.corr(numeric_only=True)
    fig6 = px.imshow(corr, text_auto=True, aspect="auto",
                     color_continuous_scale="RdBu_r",
                     title="Correlation Matrix of Features")
    c6.plotly_chart(fig6, use_container_width=True)
    

elif page == "Model Insights":
   st.title("🤖 Intelligent Churn Predictions")


   st.subheader("Model Performance")
   eval_df = modeling.evaluate_models(X_test, y_test)
   st.dataframe(eval_df.style.highlight_max(axis=0, color="#102080"), use_container_width=True)

   st.markdown("---")
   c1, c2 = st.columns([1, 2])
   with c1:
       selected_model = st.selectbox("Select Model for Feature Importance", list(modeling.trained_models.keys()))
    
   st.subheader(f"Top 10 Feature Importances ({selected_model})")
   feat_imp = modeling.get_feature_importance(selected_model).head(10)

   fig = px.bar(feat_imp, x="Importance", y="Feature", orientation='h',
             color="Importance", color_continuous_scale="Blues")
   fig.update_layout(yaxis={'categoryorder':'total ascending'})
   st.plotly_chart(fig, use_container_width=True)
 
 
elif page == "Customer Predictor":
    st.title("🎯 Customer Predictor")
    st.markdown("Fill in customer details to predict churn probability.")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("👤 Customer Profile")

        with st.form("customer_form"):

            def get_options(col, fallback):
                return df[col].dropna().unique().tolist() if col in df.columns else fallback

            gender_options = get_options("Gender", ["Male", "Female"])
            contract_options = get_options("Contract", ["Month-to-month", "One year", "Two year"])
            internet_type_options = get_options("Internet_Type", ["DSL", "Fiber optic", "Cable", "No"])
            value_deal_options = get_options("Value_Deal", ["Yes", "No"])
            yes_no_options = ["Yes", "No"]

            gender = st.selectbox("Gender", gender_options)
            age = st.slider("Age", 18, 100, 35)
            number_of_referrals = st.number_input("Number of Referrals", min_value=0, max_value=50, value=0)
            tenure = st.slider("Tenure (Months)", 0, 72, 12)

            value_deal = st.selectbox("Value Deal", value_deal_options)
            phone_service = st.selectbox("Phone Service", yes_no_options)
            internet_type = st.selectbox("Internet Type", internet_type_options)

            online_security = st.selectbox("Online Security", yes_no_options)
            online_backup = st.selectbox("Online Backup", yes_no_options)
            device_protection_plan = st.selectbox("Device Protection Plan", yes_no_options)
            premium_support = st.selectbox("Premium Support", yes_no_options)

            streaming_tv = st.selectbox("Streaming TV", yes_no_options)
            streaming_movies = st.selectbox("Streaming Movies", yes_no_options)
            streaming_music = st.selectbox("Streaming Music", yes_no_options)
            unlimited_data = st.selectbox("Unlimited Data", yes_no_options)
            contract = st.selectbox("Contract", contract_options)

            monthly_charges = st.number_input("Monthly Charge ($)", min_value=0.0, max_value=1000.0, value=70.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=100000.0, value=float(monthly_charges * max(tenure, 1)))
            total_refunds = st.number_input("Total Refunds ($)", min_value=0.0, max_value=50000.0, value=0.0)
            total_extra_data_charges = st.number_input("Total Extra Data Charges ($)", min_value=0.0, max_value=50000.0, value=0.0)
            total_long_distance_charges = st.number_input("Total Long Distance Charges ($)", min_value=0.0, max_value=50000.0, value=0.0)
            total_revenue = st.number_input("Total Revenue ($)", min_value=0.0, max_value=200000.0, value=float(total_charges + total_extra_data_charges + total_long_distance_charges - total_refunds))

            churn_category = st.text_input("Churn Category (Optional - not recommended)", "")
            churn_reason = st.text_area("Churn Reason (Optional - not recommended)", "")

            selected_model = st.selectbox("Select Prediction Model", list(modeling.trained_models.keys()))

            submit = st.form_submit_button("🔮 Predict Churn Risk")

    with c2:
        st.subheader("📈 Risk Assessment")

        if submit:
            try:
                input_dict = {
                    'Gender': [gender],
                    'Age': [age],
                    'Number_of_Referrals': [number_of_referrals],
                    'Tenure_in_Months': [tenure],
                    'Value_Deal': [value_deal],
                    'Phone_Service': [phone_service],
                    'Internet_Type': [internet_type],
                    'Online_Security': [online_security],
                    'Online_Backup': [online_backup],
                    'Device_Protection_Plan': [device_protection_plan],
                    'Premium_Support': [premium_support],
                    'Streaming_TV': [streaming_tv],
                    'Streaming_Movies': [streaming_movies],
                    'Streaming_Music': [streaming_music],
                    'Unlimited_Data': [unlimited_data],
                    'Contract': [contract],
                    'Monthly_Charge': [monthly_charges],
                    'Total_Charges': [total_charges],
                    'Total_Refunds': [total_refunds],
                    'Total_Extra_Data_Charges': [total_extra_data_charges],
                    'Total_Long_Distance_Charges': [total_long_distance_charges],
                    'Total_Revenue': [total_revenue],
                    'Churn_Category': [churn_category if churn_category else np.nan],
                    'Churn_Reason': [churn_reason if churn_reason else np.nan]
                }

                input_df = pd.DataFrame(input_dict)

                for col in X_train.columns:
                    if col not in input_df.columns:
                        if col in df.columns:
                            if df[col].dtype == 'object':
                                input_df[col] = df[col].mode()[0]
                            else:
                                input_df[col] = df[col].median()
                        else:
                            input_df[col] = np.nan

                input_df = input_df[X_train.columns]

                prob = modeling.predict_churn_prob(input_df, model_name=selected_model)
                pred_label = modeling.predict_churn_label(
                    input_df,
                    model_name=selected_model,
                    threshold=0.35
                )

                
                if prob >= 0.8:
                    risk_color = "red"
                    risk_label = "🔴 Critical Risk"
                elif prob >= 0.5:
                    risk_color = "orangered"
                    risk_label = "🟠 High Risk"
                elif prob >= 0.2:
                    risk_color = "orange"
                    risk_label = "🟡 Medium Risk"
                else:
                    risk_color = "green"
                    risk_label = "🟢 Low Risk"
                

                st.markdown(
                    f"### Churn Probability: <span style='color:{risk_color}; font-size: 2em; font-weight: bold;'>{prob*100:.1f}%</span>",
                    unsafe_allow_html=True
                )

                st.markdown(f"### Risk Level: {risk_label}")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "rgba(0,0,0,0)"},
                        'steps': [
                            {'range': [0, 20], 'color': "lightgreen"},
                            {'range': [20, 50], 'color': "moccasin"},
                            {'range': [50, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': prob * 100
                        }
                    }
                ))

                fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

                st.info(f"**💡 Retention Strategy:** {modeling.get_retention_recommendations(prob)}")

                with st.expander("📋 Show Input Used for Prediction"):
                    st.dataframe(input_df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")


elif page == "AI Analyst":
    st.title("🧠 AI Business Analyst")
    
    if not check_groq_api():
        st.stop()
    
    try:
        agent = AIBusinessAnalyst(df)
    except Exception as e:
        st.error(f"❌ AI Agent initialization failed: {e}")
        st.info("💡 Ensure your `agent.py` file is properly configured with Groq API integration.")
        st.stop()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI Business Analyst. Ask me anything about your customer churn data, such as:\n\n• 'What are the top churn drivers?'\n• 'Show me high-value customers at risk'\n• 'What retention strategies should we prioritize?'\n• 'Compare churn across demographics'"}
        ]
    

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about your churn data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤖 AI is analyzing your data..."):
                try:
                    response = agent.query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"❌ AI response failed: {e}")
                    st.info("💡 Check your Groq API key and internet connection.")
