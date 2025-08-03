import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# --- Page Configuration ---
st.set_page_config(
    page_title="Mental Health Prediction App",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Model Loading ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_my_model():
    """Loads the pre-trained CatBoost model."""
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    return model

model = load_my_model()

# --- Application Interface ---
st.title("🧠 Mental Health Condition Prediction")
st.markdown("""
This app uses a CatBoost machine learning model to predict the likelihood of a mental health condition
based on lifestyle and demographic factors. Please provide your information below.
""")

# --- Input Fields in Sidebar---
st.sidebar.header("👤 User Input Features")

# Create two columns for a more compact layout
col1, col2 = st.sidebar.columns(2)

# --- Feature Inputs ---
# Numerical Inputs
age = col1.number_input("Age", 18, 100, 30)
sleep_hours = col2.number_input("Sleep Hours (per day)", 0.0, 24.0, 7.0, 0.5)
work_hours = col1.number_input("Work Hours (per week)", 0, 100, 40)
physical_activity_hours = col2.number_input("Physical Activity (hours per week)", 0, 50, 3)
social_media_usage = col1.number_input("Social Media Usage (hours per day)", 0.0, 24.0, 2.0, 0.5)

# Categorical Inputs
gender = st.sidebar.selectbox("Gender", ('Female', 'Male', 'Non-binary', 'Prefer not to say'))
occupation = st.sidebar.selectbox("Occupation", ('Engineering', 'Education', 'Finance', 'IT', 'Sales', 'Healthcare', 'Other'))
country = st.sidebar.selectbox("Country", ('USA', 'Canada', 'UK', 'Australia', 'Germany', 'India', 'Other'))
stress_level = st.sidebar.selectbox("Stress Level", ('Low', 'Medium', 'High'))
diet_quality = st.sidebar.selectbox("Diet Quality", ('Healthy', 'Average', 'Unhealthy'))
smoking_habit = st.sidebar.selectbox("Smoking Habit", ('Non-Smoker', 'Regular Smoker', 'Heavy Smoker'))
alcohol_consumption = st.sidebar.selectbox("Alcohol Consumption", ('Non-Drinker', 'Regular Drinker', 'Heavy Drinker'))


# --- Prediction Logic ---
if st.sidebar.button("🔮 Predict Condition"):
    # Create a DataFrame from the user's inputs
    # The column order must exactly match the training data
    features = {
        'Age': age,
        'Gender': gender,
        'Occupation': occupation,
        'Country': country,
        'Stress_Level': stress_level,
        'Sleep_Hours': sleep_hours,
        'Work_Hours': work_hours,
        'Physical_Activity_Hours': physical_activity_hours,
        'Social_Media_Usage': social_media_usage,
        'Diet_Quality': diet_quality,
        'Smoking_Habit': smoking_habit,
        'Alcohol_Consumption': alcohol_consumption
    }
    input_df = pd.DataFrame([features])
    
    # Define categorical features for the model
    # This should match the list used during training
    cat_features = ['Gender', 'Occupation', 'Country', 'Diet_Quality',
                    'Smoking_Habit', 'Alcohol_Consumption', 'Stress_Level']
    
    # Ensure categorical features are of 'category' dtype
    for col in cat_features:
        input_df[col] = input_df[col].astype('category')

    # Make a prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # --- Display Results ---
    st.subheader("📊 Prediction Result")
    
    # Interpret the prediction
    if prediction[0] == 1:
        st.error("The model predicts a high likelihood of a mental health condition.", icon="😟")
    else:
        st.success("The model predicts a low likelihood of a mental health condition.", icon="😊")

    # Display prediction probabilities with more context
    st.write("Confidence Score:")
    prob_df = pd.DataFrame({
        'Condition': ['Low Likelihood', 'High Likelihood'],
        'Probability': prediction_proba[0]
    })
    st.bar_chart(prob_df.set_index('Condition'))
    
    st.info("Disclaimer: This prediction is based on a machine learning model and is not a substitute for a professional medical diagnosis. Please consult a healthcare provider for any health concerns.")


# --- Footer ---
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit & CatBoost.")