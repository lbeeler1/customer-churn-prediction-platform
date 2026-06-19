"""
Customer Churn Prediction Platform
Streamlit app for looking up a customer's churn risk and seeing a
plain-language explanation of what is driving the prediction.

Run from the project root with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import shap

st.set_page_config(page_title="Customer Churn Prediction Platform", layout="wide")

MODELS_DIR = "models/"
DATA_PATH = "data/customer_features.csv"


# ---------------------------------------------------------------------
# Load model, explainer, and data once and cache them so the app stays
# fast on every rerun instead of reloading everything on each interaction
# ---------------------------------------------------------------------

@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(MODELS_DIR + "xgb_model.json")
    return model


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def load_feature_cols():
    with open(MODELS_DIR + "feature_cols.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_customer_data():
    return pd.read_csv(DATA_PATH, index_col="customerID")


model = load_model()
explainer = load_explainer(model)
feature_cols = load_feature_cols()
df = load_customer_data()


# ---------------------------------------------------------------------
# Labeling dictionaries and explanation functions for the SHAP plots
# ---------------------------------------------------------------------

BINARY_LABELS = {
    'gender': ('being female', 'being male'),
    'SeniorCitizen': ('being a senior citizen', 'not being a senior citizen'),
    'Partner': ('having a partner', 'not having a partner'),
    'Dependents': ('having dependents', 'not having dependents'),
    'PhoneService': ('having phone service', 'not having phone service'),
    'MultipleLines': ('having multiple phone lines', 'not having multiple phone lines'),
    'OnlineSecurity': ('having online security', 'not having online security'),
    'OnlineBackup': ('having online backup', 'not having online backup'),
    'DeviceProtection': ('having device protection', 'not having device protection'),
    'TechSupport': ('having tech support', 'not having tech support'),
    'StreamingTV': ('having streaming TV', 'not having streaming TV'),
    'StreamingMovies': ('having streaming movies', 'not having streaming movies'),
    'PaperlessBilling': ('using paperless billing', 'not using paperless billing'),
    'InternetService_Fiber optic': ('having fiber optic internet', 'not having fiber optic internet'),
    'InternetService_No': ('having no internet service', 'having internet service'),
    'Contract_One year': ('being on a one year contract', 'not being on a one year contract'),
    'Contract_Two year': ('being on a two year contract', 'not being on a two year contract'),
    'PaymentMethod_Credit card (automatic)': ('paying by automatic credit card', 'not paying by automatic credit card'),
    'PaymentMethod_Electronic check': ('paying by electronic check', 'not paying by electronic check'),
    'PaymentMethod_Mailed check': ('paying by mailed check', 'not paying by mailed check'),
}

CONTINUOUS_LABELS = {
    'MonthlyCharges': 'a monthly charge of ${value:.2f}',
    'TotalCharges': 'a total spend of ${value:.2f}',
}


def describe_feature(feature, value):
    if feature in BINARY_LABELS:
        yes_label, no_label = BINARY_LABELS[feature]
        return yes_label if value == 1 else no_label
    elif feature == 'tenure':
        month_word = 'month' if round(value) == 1 else 'months'
        return f'a tenure of {value:.0f} {month_word}'
    elif feature in CONTINUOUS_LABELS:
        return CONTINUOUS_LABELS[feature].format(value=value)
    else:
        return f'{feature} = {value}'


def explain_prediction(shap_row_values, feature_values, feature_names, top_n=3):
    """Build a plain-language explanation from the top SHAP contributors for one customer."""
    contributions = list(zip(feature_names, shap_row_values, feature_values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top = contributions[:top_n]

    phrases = []
    for feature, shap_val, value in top:
        direction = 'increases' if shap_val > 0 else 'decreases'
        phrases.append(f'{describe_feature(feature, value)} ({direction} risk)')

    return 'The biggest factors in this prediction are ' + ', '.join(phrases) + '.'


def plot_explanation(shap_row_values, feature_values, feature_names, title=None, top_n=5):
    """Build a horizontal bar chart of the top SHAP contributors with plain-language labels."""
    contributions = list(zip(feature_names, shap_row_values, feature_values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top = contributions[:top_n]

    labels = [describe_feature(f, v) for f, _, v in top]
    values = [s for _, s, _ in top]
    colors = ['crimson' if v > 0 else 'steelblue' for v in values]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Impact on churn risk')
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def encode_customer(raw):
    row = {
        'gender': 1 if raw['gender'] == 'Female' else 0,
        'SeniorCitizen': raw['SeniorCitizen'],
        'Partner': 1 if raw['Partner'] == 'Yes' else 0,
        'Dependents': 1 if raw['Dependents'] == 'Yes' else 0,
        'tenure': raw['tenure'],
        'PhoneService': 1 if raw['PhoneService'] == 'Yes' else 0,
        'MultipleLines': 1 if raw['MultipleLines'] == 'Yes' else 0,
        'OnlineSecurity': 1 if raw['OnlineSecurity'] == 'Yes' else 0,
        'OnlineBackup': 1 if raw['OnlineBackup'] == 'Yes' else 0,
        'DeviceProtection': 1 if raw['DeviceProtection'] == 'Yes' else 0,
        'TechSupport': 1 if raw['TechSupport'] == 'Yes' else 0,
        'StreamingTV': 1 if raw['StreamingTV'] == 'Yes' else 0,
        'StreamingMovies': 1 if raw['StreamingMovies'] == 'Yes' else 0,
        'PaperlessBilling': 1 if raw['PaperlessBilling'] == 'Yes' else 0,
        'MonthlyCharges': raw['MonthlyCharges'],
        'TotalCharges': raw['TotalCharges'],
        'InternetService_Fiber optic': 1 if raw['InternetService'] == 'Fiber optic' else 0,
        'InternetService_No': 1 if raw['InternetService'] == 'No' else 0,
        'Contract_One year': 1 if raw['Contract'] == 'One year' else 0,
        'Contract_Two year': 1 if raw['Contract'] == 'Two year' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if raw['PaymentMethod'] == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if raw['PaymentMethod'] == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if raw['PaymentMethod'] == 'Mailed check' else 0,
    }
    return pd.DataFrame([row])[feature_cols]


# ---------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------

st.title("Customer Churn Prediction Platform")
st.write(
    "Look up an existing customer or enter custom values to see a churn "
    "risk score and a plain-language explanation of what is driving it."
)

# ---------------------------------------------------------------------
# Sidebar: choose how to provide a customer
# ---------------------------------------------------------------------

mode = st.sidebar.radio("Choose input method", ["Look up existing customer", "Enter custom values"])

if mode == "Look up existing customer":
    customer_id = st.sidebar.selectbox("Select a customer", df.index.tolist())
    customer_row = df.loc[[customer_id]].drop(columns=["Churn"])

else:
    st.sidebar.subheader("Customer Details")

    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior = st.sidebar.checkbox("Senior citizen")
    partner = st.sidebar.selectbox("Has partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Has dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone = st.sidebar.selectbox("Phone service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple lines", ["Yes", "No"])
    internet = st.sidebar.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online security", ["Yes", "No"])
    online_backup = st.sidebar.selectbox("Online backup", ["Yes", "No"])
    device_protection = st.sidebar.selectbox("Device protection", ["Yes", "No"])
    tech_support = st.sidebar.selectbox("Tech support", ["Yes", "No"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.sidebar.selectbox("Streaming movies", ["Yes", "No"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.sidebar.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.sidebar.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    
    # UPDATE: Configured step=1.0 to increment charges by whole dollars instead of pennies
    monthly_charges = st.sidebar.number_input(
        "Monthly charges ($)", min_value=0.0, max_value=150.0, value=65.0, step=1.0
    )
    total_charges = st.sidebar.number_input(
        "Total charges ($)", min_value=0.0, max_value=10000.0,
        value=float(monthly_charges * tenure), step=1.0
    )

    raw = {
        "gender": gender,
        "SeniorCitizen": 1 if senior else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "PaperlessBilling": paperless,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "InternetService": internet,
        "Contract": contract,
        "PaymentMethod": payment,
    }
    customer_row = encode_customer(raw)

# ---------------------------------------------------------------------
# Run the prediction and explanation
# ---------------------------------------------------------------------

probability = model.predict_proba(customer_row)[0, 1]

if probability < 0.3:
    risk_label, risk_color = "Low Risk", "green"
elif probability < 0.6:
    risk_label, risk_color = "Medium Risk", "orange"
else:
    risk_label, risk_color = "High Risk", "red"

shap_row = explainer(customer_row)
explanation = explain_prediction(shap_row.values[0], customer_row.values[0], feature_cols)

# ---------------------------------------------------------------------
# Row 1: Key Metrics Bar (Horizontal)
# ---------------------------------------------------------------------
metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.metric("Churn Probability", f"{probability * 100:.1f}%")

with metric_col2:
    st.markdown(
        f"**Risk Level:** <br><span style='font-size:26px; font-weight:bold; color:{risk_color};'>{risk_label}</span>", 
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------
# Row 2: Full-Width Prediction Drivers & SHAP Visualization
# ---------------------------------------------------------------------
st.write("")  
st.subheader("Prediction Drivers")
st.write(explanation)

fig = plot_explanation(shap_row.values[0], customer_row.values[0], feature_cols)
st.pyplot(fig, use_container_width=True)

# ---------------------------------------------------------------------
# Wide Layout Dropdown: Clean, Decoded Feature Grid
# ---------------------------------------------------------------------
st.write("---")
with st.expander("View full customer details", expanded=False):
    # Extract row values cleanly
    gender_val = customer_row.iloc[0]['gender']
    senior_val = customer_row.iloc[0]['SeniorCitizen']
    partner_val = customer_row.iloc[0]['Partner']
    dependents_val = customer_row.iloc[0]['Dependents']
    tenure_val = customer_row.iloc[0]['tenure']
    phone_val = customer_row.iloc[0]['PhoneService']
    multiple_val = customer_row.iloc[0]['MultipleLines']
    security_val = customer_row.iloc[0]['OnlineSecurity']
    backup_val = customer_row.iloc[0]['OnlineBackup']
    device_val = customer_row.iloc[0]['DeviceProtection']
    tech_val = customer_row.iloc[0]['TechSupport']
    tv_val = customer_row.iloc[0]['StreamingTV']
    movies_val = customer_row.iloc[0]['StreamingMovies']
    paperless_val = customer_row.iloc[0]['PaperlessBilling']
    monthly_val = customer_row.iloc[0]['MonthlyCharges']
    total_val = customer_row.iloc[0]['TotalCharges']
    
    # Re-combine One-Hot Encoded Categories cleanly back to single answers
    if customer_row.iloc[0]['InternetService_No'] == 1:
        internet_val = "No Internet Service"
    elif customer_row.iloc[0]['InternetService_Fiber optic'] == 1:
        internet_val = "Fiber optic"
    else:
        internet_val = "DSL"
        
    if customer_row.iloc[0]['Contract_Two year'] == 1:
        contract_val = "Two year"
    elif customer_row.iloc[0]['Contract_One year'] == 1:
        contract_val = "One year"
    else:
        contract_val = "Month-to-month"
        
    if customer_row.iloc[0]['PaymentMethod_Electronic check'] == 1:
        payment_val = "Electronic check"
    elif customer_row.iloc[0]['PaymentMethod_Mailed check'] == 1:
        payment_val = "Mailed check"
    elif customer_row.iloc[0]['PaymentMethod_Credit card (automatic)'] == 1:
        payment_val = "Credit card (automatic)"
    else:
        payment_val = "Bank transfer (automatic)"

    # Construct clean profile rows
    profile_data = [
        {"Customer Attribute": "Gender", "Value": "Female" if gender_val == 1 else "Male"},
        {"Customer Attribute": "Senior Citizen Status", "Value": "Yes" if senior_val == 1 else "No"},
        {"Customer Attribute": "Has Partner", "Value": "Yes" if partner_val == 1 else "No"},
        {"Customer Attribute": "Has Dependents", "Value": "Yes" if dependents_val == 1 else "No"},
        {"Customer Attribute": "Account Tenure", "Value": f"{tenure_val:.0f} month" if round(tenure_val) == 1 else f"{tenure_val:.0f} months"},
        {"Customer Attribute": "Phone Service Plan", "Value": "Yes" if phone_val == 1 else "No"},
        {"Customer Attribute": "Multiple Phone Lines", "Value": "Yes" if multiple_val == 1 else "No"},
        {"Customer Attribute": "Internet Service Type", "Value": internet_val},
        {"Customer Attribute": "Online Security Add-on", "Value": "Yes" if security_val == 1 else "No"},
        {"Customer Attribute": "Cloud Backup Add-on", "Value": "Yes" if backup_val == 1 else "No"},
        {"Customer Attribute": "Device Protection Plan", "Value": "Yes" if device_val == 1 else "No"},
        {"Customer Attribute": "Premium Tech Support Plan", "Value": "Yes" if tech_val == 1 else "No"},
        {"Customer Attribute": "TV Streaming Subscription", "Value": "Yes" if tv_val == 1 else "No"},
        {"Customer Attribute": "Movie Streaming Subscription", "Value": "Yes" if movies_val == 1 else "No"},
        {"Customer Attribute": "Contract Agreement", "Value": contract_val},
        {"Customer Attribute": "Paperless Billing Method", "Value": "Yes" if paperless_val == 1 else "No"},
        {"Customer Attribute": "Selected Payment Method", "Value": payment_val},
        {"Customer Attribute": "Monthly Recurring Charge", "Value": f"${monthly_val:.2f}"},
        {"Customer Attribute": "Cumulative Total Spend", "Value": f"${total_val:.2f}"}
    ]
    
    details_df = pd.DataFrame(profile_data)
    st.dataframe(details_df, use_container_width=True, hide_index=True)