import streamlit as st
import pandas as pd
import joblib
st.set_page_config(page_title="Income Classifier", layout="wide", page_icon="💵")
st.markdown("""
<style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    [data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }

    h1, h2, h3 {
        color: #10B981 !important;
    }

    .stSelectbox label, .stSlider label, .stRadio label {
        color: #CBD5E1 !important;
        font-weight: 600;
    }

   
    [data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 10px;
    }

    div.stSlider > div[data-baseweb="slider"] > div {
        background-color: #10B981 !important;
    }
    .stProgress > div > div > div > div {
        background-color: #10B981 !important;
    }
</style>
""",
    unsafe_allow_html=True,

)
st.write("""
 # Income classifier using XGBoost
A machine learning model will predict whether a person earns morethan $50K """ )
st.sidebar.header('User Input Parameters')
def user_input_features():
    age = st.sidebar.slider('age', 1, 100, 30)
    workclass = st.sidebar.selectbox(
       "Work_class",[
       "Private",
            "Government",
            "Self-employed",
            "Other"
       ]

    )
    education = st.sidebar.selectbox("Education",['11th', 'hs-grad', 'assoc-acdm', 'some-college', '10th', 
                              'prof-school', '7th-8th', 'bachelors', 'masters', 'doctorate', '5th-6th', 'assoc-voc', '9th', '12th', '1st-4th', 'preschool'])
    education_mapping = {
    "preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "hs-grad": 9,
    "some-college": 10,
    "assoc-voc": 11,
    "assoc-acdm": 12,
    "bachelors": 13,
    "masters": 14,
    "prof-school": 15,
    "doctorate": 16
}
    education_num = education_mapping[education]
    marital_status = st.sidebar.selectbox("Marital-status",["Single", "Married", "Previously-Married"])
    occupation = st.sidebar.selectbox("Occupation", [
            "Professional",
            "Office-Service",
            "Skilled-Manual",
            "Specialized"
        ])
    relationship= st.sidebar.selectbox('Relationship', ["Spouse",
            "Child",
            "Other"])
    race = st.sidebar.selectbox('Race', ['Black', 'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo','Other'])
    gender = st.sidebar.radio("Gender",["Male","Female"])
    native_country = st.sidebar.selectbox("Native_country",["US", "Non-US"])
    capital_gain = st.sidebar.slider('Capital gain', 0, 100000, 10000)
    capital_loss = st.sidebar.slider('Capital loss', 0, 100000, 6000)
    hours_per_week = st.sidebar.slider('Hours per week', 0, 140, 40)
    capital_net = capital_gain - capital_loss
    has_capital_activity = int(capital_gain > 0 or capital_loss > 0)
    data = {
         
        "age": [age],
        "workclass": [workclass],
        "education-num": [education_num],
        "marital-status": [marital_status],
        "occupation": [occupation],
        "relationship": [relationship],
        "race": [race],
        "gender": [gender],
        "native-country": [native_country],
        "capital_net": [capital_net],
        "has_capital_activity": [has_capital_activity],
        "hours-per-week": [hours_per_week]
    
    }
    features = pd.DataFrame(data)
    return features
df = user_input_features()
st.subheader('User Input Parameters')
st.write(df)
model = joblib.load("model.pkl")
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)
prob_lessthan50 = prediction_proba[0][0]
prob_morethan50 = prediction_proba[0][1]
st.header("Prediction Results")
if prediction[0] == 1:
    st.success(f"### Output: **>= $50K / year**")
else:
    st.info(f"### Output: **< $50K / year**")
st.markdown("---")
st.subheader("Prediction Confidence")
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label = "Probability of earning < $50K",
        value = f"{prob_lessthan50 * 100:.1f}%"
    )
    st.progress(float(prob_lessthan50))
with col2:
     st.metric(
            label = "Probability of earning >= $50K",
            value = f"{prob_morethan50 * 100:.1f}%"
        )
     st.progress(float(prob_morethan50))