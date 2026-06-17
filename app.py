import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(
    page_title="IPL Chase Predictor",
    page_icon="🏏",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1 {
    text-align:center;
    color:#FF4B4B;
}
.stButton>button {
    width:100%;
    background-color:#FF4B4B;
    color:white;
    font-size:18px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("model.pkl")

st.title("🏏 IPL Chase Win Predictor")

st.markdown("### Predict the probability of chasing the target")

target = st.number_input(
    "🎯 Target Score",
    min_value=30,
    value=180
)

current_runs = st.number_input(
    "🏃 Current Runs",
    min_value=30,
    value=100
)

balls_bowled = st.number_input(
    "⚾ Balls Bowled",
    min_value=30,
    max_value=300,
    value=120
)

wickets_fallen = st.number_input(
    "❌ Wickets Fallen",
    min_value=0,
    max_value=10,
    value=3
)

if st.button("Predict Winning Probability"):

    balls_left = 300 - balls_bowled
    wickets_left = 10 - wickets_fallen
    req_runs = target - current_runs

    if balls_left > 0:
        rr = round((req_runs * 6) / balls_left, 2)
        cr = round((current_runs * 6) / balls_bowled, 2)
    else:
        rr = 0
        cr = 0

    features = pd.DataFrame({
        'target':[target],
        'rr':[rr],
        'cr':[cr],
        'req_runs':[req_runs],
        'balls_left':[balls_left],
        'wickets_left':[wickets_left]
    })

    prob = model.predict_proba(features)[0]

    lose_prob = round(prob[0] * 100, 2)
    win_prob = round(prob[1] * 100, 2)

    st.success(f"🏆 Winning Chance : {win_prob}%")
    st.error(f"💀 Losing Chance : {lose_prob}%")

    st.progress(int(win_prob))

    st.markdown("---")

    st.metric("Required Runs", req_runs)
    st.metric("Balls Left", balls_left)
    st.metric("Current Run Rate", cr)
    st.metric("Required Run Rate", rr)
