import streamlit as st
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Smart Student Performance AI")
st.write("An end-to-end Machine Learning dashboard predicting student success.")

# Sidebar controls for user input
st.sidebar.header("Input Student Features")
study_hours = st.sidebar.slider("Study Hours (per day)", 0.0, 12.0, 6.5, 0.5)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 87, 1)
previous_score = st.sidebar.slider("Previous Exam Score", 0, 100, 72, 1)
sleep_hours = st.sidebar.slider("Sleep Hours (per night)", 0.0, 12.0, 7.0, 0.5)
assignments = st.sidebar.slider("Assignments Completed (%)", 0, 100, 90, 1)
extracurricular = st.sidebar.slider("Extracurricular Hours (per week)", 0.0, 20.0, 4.0, 0.5)

# Prediction execution on button click
if st.button("PREDICT PERFORMANCE", type="primary"):
    # Simulated inference based on our trained feature relationships
    predicted_score = 15 + (study_hours * 3.8) + (attendance * 0.25) + (previous_score * 0.32) + (sleep_hours * 1.5) + (assignments * 0.1)
    predicted_score = float(np.clip(predicted_score, 0, 100))
    
    # Pass probability using our sigmoid concept
    z = (predicted_score - 55) / 8.0
    pass_prob = 1 / (1 + np.exp(-z))
    
    # Risk calculation
    if pass_prob >= 0.8:
        risk = "LOW"
        risk_color = "green"
    elif pass_prob >= 0.5:
        risk = "MEDIUM"
        risk_color = "orange"
    else:
        risk = "HIGH"
        risk_color = "red"

    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Score", f"{predicted_score:.1f}")
    col2.metric("Pass Probability", f"{pass_prob * 100:.1f}%")
    col3.markdown(f"**Risk Level:**\n\n :{risk_color}[**{risk}**]")