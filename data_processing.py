import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import urllib.parse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

def load_data_from_sqlserver(server, database, table_name, trusted_connection=True, username=None, password=None):
    """Load data from SQL Server / SSMS"""

    server = server.strip()
    database = database.strip()
    table_name = table_name.strip()

    try:
        params = urllib.parse.quote_plus(
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            f"TrustServerCertificate=yes;"
        )
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
        return engine
    except Exception as e:
        return str(e)

def load_data(server, username, password, database, table):
    try:
        engine = load_data_from_sqlserver(server, database, username, password)

        if isinstance(engine, str):
            return engine

        query = text(f"SELECT * FROM [{table}]")
        df = pd.read_sql(query, engine)
        return df

    except Exception as e:
        return f"SQL Server error: {str(e)}"



def standardize_columns(df):
    """Convert column names to lowercase snake_case"""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def clean_data(df):
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()              
        .str.replace(" ", "_")   
        .str.replace(r"[^\w]", "", regex=True)  
    )

    print("Cleaned columns:", df.columns.tolist())

    if 'Total_Charges' in df.columns:
        df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce').fillna(0)
    elif 'TotalCharges' in df.columns:
        df['Total_Charges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    else:
        raise KeyError(
            f"'Total_Charges' column not found. Available columns: {df.columns.tolist()}"
        )

    return df

    if 'Customer_Status' in df.columns:
        df['Customer_Status'] = (
            df['Customer_Status']
            .astype(str)
            .str.strip()
            .str.lower()
            .map({'yes': 'Yes', 'no': 'No', '1': 'Yes', '0': 'No'})
        )
        df['Customer_Status'] = df['Customer_Status'].fillna('No')

    print(f"✅ Cleaned data: {len(df)} rows, {len(df.columns)} columns")
    return df


def preprocess_data(df):
    """
    Return CLEAN DataFrame + target only.
    DO NOT transform into numpy array here.
    Let model.py handle preprocessing.
    """
    df = df.copy()
    df = clean_data(df)

    if 'Customer_Status' not in df.columns:
        raise ValueError("❌ Missing 'Customer_Status' column. Dataset must have churn target.")

    feature_cols = [col for col in df.columns if col not in ['Customer_ID', 'Customer_ID', 'Customer_Status']]
    X = df[feature_cols].copy()

    y = df['Customer_Status'].astype(str).map({
        'Churned': 1, 'Stated': 0, 'yes': 1, 'no': 0
    }).fillna(0).astype(int)

    print(f"📊 Features: {X.shape[1]}, Target encoded: {y.value_counts().to_dict()}")

    encoders = {}
    scaler = None

    return X, y, encoders, scaler


def get_train_test_split(X, y, test_size=0.2, random_state=42):
    """Standard train/test split"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"✅ Train/Test split: {X_train.shape[0]}/{X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def legacy_preprocess_data(df):
    """Legacy version for backward compatibility"""
    print("⚠️ Using legacy preprocessing")

    df = clean_data(df)

    X = df.drop(['Customer_ID', 'Customer_Status'], axis=1, errors='ignore')
    y = df['Customer_Status'].apply(lambda x: 1 if str(x).lower() == 'Churned' else 0)

    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=['number']).columns

    encoders = {}
    for col in cat_cols:
        X[col] = X[col].fillna('Unknown')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(0)
        X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y, encoders, scaler

