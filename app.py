import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Beat the AI | Student Predictor", page_icon="🎯", layout="wide")

DATA_FILE = "student_performance.csv"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            "student_performance.csv is missing. Keep it in the same folder as app.py."
        )
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def train_model():
    data = load_data()
    features = ["study_hours", "attendance", "previous_score", "sleep_hours"]
    X = data[features]
    y = data["actual_marks"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = np.clip(model.predict(X_test), 0, 100)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return model, mae, r2, X_test.reset_index(drop=True), y_test.reset_index(drop=True)

data = load_data()
model, mae, r2, challenge_X, challenge_y = train_model()

if "round_index" not in st.session_state:
    st.session_state.round_index = 0
if "total_score" not in st.session_state:
    st.session_state.total_score = 0
if "played" not in st.session_state:
    st.session_state.played = 0
if "revealed" not in st.session_state:
    st.session_state.revealed = False
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🎯 Beat the AI: Student Performance Predictor")
st.write(
    "Predict a student's exam marks from their study habits. Then compare your guess "
    "with a Machine Learning model and the hidden actual score."
)
st.info(
    "Learning-project note: this app uses an illustrative synthetic dataset. "
    "It is for demonstrating regression, not for making real academic decisions."
)

with st.sidebar:
    st.header("How to play")
    st.markdown(
        "1. Study the student's profile.  \n"
        "2. Enter your predicted marks.  \n"
        "3. Click **Lock my prediction**.  \n"
        "4. Reveal the actual marks and compare your score with the AI."
    )
    st.divider()
    st.metric("Your total points", st.session_state.total_score)
    st.metric("Rounds played", st.session_state.played)
    st.caption("Scoring: 100 points minus the absolute error, with a minimum of 0.")

st.subheader("Your challenge")
idx = st.session_state.round_index % len(challenge_X)
profile = challenge_X.iloc[idx]
actual = float(challenge_y.iloc[idx])
ai_prediction = float(np.clip(model.predict(pd.DataFrame([profile]))[0], 0, 100))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Study hours / day", f"{profile['study_hours']:.1f} hrs")
c2.metric("Attendance", f"{profile['attendance']:.1f}%")
c3.metric("Previous exam score", f"{profile['previous_score']:.1f}/100")
c4.metric("Sleep / night", f"{profile['sleep_hours']:.1f} hrs")

if not st.session_state.revealed:
    guess = st.slider("What do you think this student's exam mark will be?", 0, 100, 65)
    if st.button("🔒 Lock my prediction", type="primary", use_container_width=True):
        st.session_state.current_guess = int(guess)
        st.session_state.current_ai = round(ai_prediction, 1)
        st.session_state.current_actual = round(actual, 1)
        st.session_state.revealed = True
        st.rerun()
else:
    st.success("Predictions locked! Here are the results.")
    r1, r2, r3 = st.columns(3)
    r1.metric("Your prediction", f"{st.session_state.current_guess}/100")
    r2.metric("AI prediction", f"{st.session_state.current_ai}/100")
    r3.metric("Actual marks", f"{st.session_state.current_actual}/100")

    player_error = abs(st.session_state.current_guess - st.session_state.current_actual)
    ai_error = abs(st.session_state.current_ai - st.session_state.current_actual)
    points = max(0, int(round(100 - player_error)))

    a, b = st.columns(2)
    a.metric("Your error", f"{player_error:.1f} marks")
    b.metric("AI error", f"{ai_error:.1f} marks")

    if player_error < ai_error:
        st.balloons()
        st.success(f"You beat the AI! You earned {points} points.")
    elif player_error == ai_error:
        st.info(f"It's a tie with the AI! You earned {points} points.")
    else:
        st.warning(f"The AI was closer this round. You earned {points} points.")

    if st.button("➡️ Next round", type="primary", use_container_width=True):
        st.session_state.total_score += points
        st.session_state.played += 1
        st.session_state.history.append({
            "Round": st.session_state.played,
            "Your guess": st.session_state.current_guess,
            "AI guess": st.session_state.current_ai,
            "Actual marks": st.session_state.current_actual,
            "Your error": round(player_error, 1),
            "Points": points
        })
        st.session_state.round_index += 1
        st.session_state.revealed = False
        st.rerun()

st.divider()
st.subheader("📊 Model performance")
m1, m2, m3 = st.columns(3)
m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f} marks")
m2.metric("R² score", f"{r2:.3f}")
m3.metric("Training records", f"{len(data):,}")

st.caption(
    "MAE tells us the average prediction error in marks (lower is better). "
    "R² describes how much variation the model explains (closer to 1 is generally better)."
)

with st.expander("Explore the dataset"):
    st.write("A small preview of the synthetic training dataset:")
    st.dataframe(data.head(15), use_container_width=True)
    st.write("Average marks by study-hour group:")
    plot_data = data.copy()
    plot_data["study_group"] = pd.cut(
        plot_data["study_hours"], bins=[0, 2, 4, 6, 8, 10],
        labels=["0–2", "2–4", "4–6", "6–8", "8–10"]
    )
    summary = plot_data.groupby("study_group", observed=False)["actual_marks"].mean()
    fig, ax = plt.subplots()
    summary.plot(kind="bar", ax=ax)
    ax.set_xlabel("Study hours per day")
    ax.set_ylabel("Average marks")
    ax.set_title("Average marks by study-hour group")
    st.pyplot(fig)
    plt.close(fig)

if st.session_state.history:
    st.subheader("🏆 Your game history")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
    st.metric("Average points per round", f"{history_df['Points'].mean():.1f}")
    if st.button("Reset game"):
        for key in ["round_index", "total_score", "played", "revealed", "history"]:
            st.session_state[key] = [] if key == "history" else (False if key == "revealed" else 0)
        st.rerun()
