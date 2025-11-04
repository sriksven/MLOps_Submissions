import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------------
# Must come first
# --------------------------------------------------------
st.set_page_config(page_title="Job Salary Prediction App", layout="centered")

# --------------------------------------------------------
# Load model, data, and encoders
# --------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/salary_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_data.csv")

@st.cache_resource
def load_encoders():
    return joblib.load("models/label_encoders.pkl")

model = load_model()
data = load_data()
encoders = load_encoders()

# --------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------
st.title("💼 Job Salary Prediction App")
st.markdown("Predict estimated salaries based on job attributes using a trained ML model.")
st.divider()

st.sidebar.header("About this App")
st.sidebar.write("""
This app uses a Random Forest model trained on the
[Data Science Job Salaries 2023 dataset](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023).
""")

# --------------------------------------------------------
# Helper function to encode
# --------------------------------------------------------
def encode_value(column, label):
    """Return encoded numeric value for a category using saved encoders"""
    le = encoders.get(column)
    if le:
        try:
            return le.transform([label])[0]
        except ValueError:
            return 0
    return label

# --------------------------------------------------------
# Input Form
# --------------------------------------------------------
st.subheader("Input Job Details")

col1, col2 = st.columns(2)

with col1:
    job_title_text = st.selectbox("Job Title", encoders["job_title"].classes_)
    experience_text = st.selectbox("Experience Level", encoders["experience_level"].classes_)
    employment_text = st.selectbox("Employment Type", encoders["employment_type"].classes_)

with col2:
    company_loc_text = st.selectbox("Company Location", encoders["company_location"].classes_)
    company_size_text = st.selectbox("Company Size", encoders["company_size"].classes_)
    remote_ratio = st.slider("Remote Ratio (0 = Onsite, 100 = Fully Remote)", 0, 100, 50)

# --------------------------------------------------------
# Predict Salary
# --------------------------------------------------------
if st.button("🔍 Predict Salary"):
    input_data = pd.DataFrame({
        "work_year": [2023],
        "experience_level": [encode_value("experience_level", experience_text)],
        "employment_type": [encode_value("employment_type", employment_text)],
        "job_title": [encode_value("job_title", job_title_text)],
        "employee_residence": [encode_value("company_location", company_loc_text)],
        "remote_ratio": [remote_ratio],
        "company_location": [encode_value("company_location", company_loc_text)],
        "company_size": [encode_value("company_size", company_size_text)]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"💰 **Estimated Salary:** ${prediction:,.0f} USD")

# --------------------------------------------------------
# Visuals
# --------------------------------------------------------
st.divider()
st.subheader("📊 Salary Insights")

tab1, tab2 = st.tabs(["Average Salary by Job Title", "Remote Work Impact"])

with tab1:
    avg_salary_by_job = data.groupby("job_title")["salary_in_usd"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(avg_salary_by_job)

with tab2:
    avg_remote = data.groupby("remote_ratio")["salary_in_usd"].mean()
    st.line_chart(avg_remote)

st.caption("© 2025 Job Salary Prediction | Streamlit Demo App")
