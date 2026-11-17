from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Beat the AI | Live Student Predictor",
    page_icon="🎯",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "student_performance.csv"
FEATURES = ["study_hours", "attendance", "previous_score", "sleep_hours"]


# ------------------------------------------------------------
# Load data and train the model once
# ------------------------------------------------------------
@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "student_performance.csv was not found. "
            "Upload it to the same GitHub folder as app.py."
        )

    data = pd.read_csv(DATA_PATH)
    required = FEATURES + ["actual_marks"]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"The dataset is missing these columns: {missing}")

    data = data[required].dropna()
    if len(data) < 20:
        raise ValueError("The dataset needs at least 20 complete records.")
    return data


@st.cache_resource
def train_model():
    data = load_dataset()
    X = data[FEATURES]
    y = data["actual_marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    test_predictions = np.clip(model.predict(X_test), 0, 100)
    mae = mean_absolute_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)

    return model, mae, r2, len(X_train), len(X_test)


data = load_dataset()
model, mae, r2, train_count, test_count = train_model()


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
defaults = {
    "rounds": 0,
    "total_points": 0,
    "history": [],
    "last_result": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("🎯 Beat the AI: Live Student Performance Predictor")
st.write(
    "Enter a student's details, make your own prediction, and see how close "
    "you are to the Linear Regression model."
)
st.info(
    "This is an educational demonstration. The included dataset is synthetic. "
    "Predictions are estimates, not guaranteed exam results."
)

# ------------------------------------------------------------
# Sidebar score panel
# ------------------------------------------------------------
with st.sidebar:
    st.header("🏆 Your game score")
    st.metric("Total points", st.session_state.total_points)
    st.metric("Rounds played", st.session_state.rounds)
    if st.button("Reset game score", use_container_width=True):
        st.session_state.rounds = 0
        st.session_state.total_points = 0
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()

    st.divider()
    st.caption(
        "Scoring: up to 100 points for a round, based on how close your guess "
        "is to the AI's estimate. This measures agreement with the model, "
        "not real-world prediction accuracy."
    )


# ------------------------------------------------------------
# Live prediction form
# ------------------------------------------------------------
st.header("🧑‍🎓 Enter student information")
st.caption("Change the values below to generate a new prediction.")

with st.form("live_prediction_form"):
    c1, c2 = st.columns(2)

    with c1:
        study_hours = st.slider(
            "Study hours per day", min_value=0.0, max_value=12.0,
            value=6.0, step=0.5
        )
        attendance = st.slider(
            "Attendance (%)", min_value=0, max_value=100,
            value=85, step=1
        )

    with c2:
        previous_score = st.slider(
            "Previous exam marks", min_value=0, max_value=100,
            value=70, step=1
        )
        sleep_hours = st.slider(
            "Sleep hours per night", min_value=0.0, max_value=12.0,
            value=7.0, step=0.5
        )

    st.subheader("🎮 Make your prediction")
    player_guess = st.slider(
        "What marks do you think this student will score?",
        min_value=0, max_value=100, value=75, step=1
    )

    submitted = st.form_submit_button(
        "🚀 Predict marks and compare with AI",
        type="primary",
        use_container_width=True,
    )


# ------------------------------------------------------------
# Generate live prediction
# ------------------------------------------------------------
if submitted:
    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours,
    }])

    ai_prediction = float(np.clip(model.predict(input_data)[0], 0, 100))
    player_error = abs(float(player_guess) - ai_prediction)
    points = max(0, int(round(100 - player_error)))

    st.session_state.last_result = {
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours,
        "player_guess": int(player_guess),
        "ai_prediction": round(ai_prediction, 1),
        "difference": round(player_error, 1),
        "points": points,
    }

    st.session_state.rounds += 1
    st.session_state.total_points += points
    st.session_state.history.append({
        "Round": st.session_state.rounds,
        "Your guess": int(player_guess),
        "AI estimate": round(ai_prediction, 1),
        "Difference": round(player_error, 1),
        "Points": points,
    })


# ------------------------------------------------------------
# Show latest result
# ------------------------------------------------------------
result = st.session_state.last_result

if result:
    st.divider()
    st.header("📊 Live prediction result")

    r1, r2, r3 = st.columns(3)
    r1.metric("Your prediction", f"{result['player_guess']}/100")
    r2.metric("AI prediction", f"{result['ai_prediction']}/100")
    r3.metric("Your points", f"{result['points']}/100")

    st.metric("Difference between your guess and AI", f"{result['difference']} marks")

    if result["difference"] <= 5:
        st.success("🎉 Excellent! Your prediction is very close to the AI.")
    elif result["difference"] <= 15:
        st.info("👍 Good attempt! You are reasonably close to the AI.")
    else:
        st.warning("🤖 The AI estimate is quite different from your guess. Try another profile!")

    chart_df = pd.DataFrame({
        "Prediction": ["Your guess", "AI estimate"],
        "Marks": [result["player_guess"], result["ai_prediction"]],
    })
    fig, ax = plt.subplots()
    ax.bar(chart_df["Prediction"], chart_df["Marks"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Marks out of 100")
    ax.set_title("Your prediction vs AI estimate")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "For a newly entered student, the actual exam result is unknown. "
        "The comparison above is between your guess and the model's estimate."
    )

    st.subheader("Student details used")
    st.write({
        "Study hours per day": result["study_hours"],
        "Attendance (%)": result["attendance"],
        "Previous exam marks": result["previous_score"],
        "Sleep hours per night": result["sleep_hours"],
    })


# ------------------------------------------------------------
# Model performance and history
# ------------------------------------------------------------
st.divider()
st.header("📈 Model performance")

m1, m2, m3 = st.columns(3)
m1.metric("Mean Absolute Error", f"{mae:.2f} marks")
try:
    r2_value = float(r2)
    r2_display = (
        f"{r2_value:.3f}"
        if np.isfinite(r2_value)
        else "N/A"
    )
except (TypeError, ValueError):
    r2_display = "N/A"

m2.metric("R² score", r2_display)
m3.metric("Training records", f"{train_count:,}")

st.caption(
    f"The model was evaluated on {test_count:,} held-out records. "
    "MAE is the average absolute error; lower is better. "
    "R² describes how much variation the model explains."
)

with st.expander("Explore training dataset"):
    st.write(f"Dataset records: {len(data):,}")
    st.dataframe(data.head(20), use_container_width=True)

    fig, ax = plt.subplots()
    ax.scatter(data["study_hours"], data["actual_marks"], alpha=0.5)
    ax.set_xlabel("Study hours per day")
    ax.set_ylabel("Actual marks")
    ax.set_title("Study hours vs marks in the training dataset")
    st.pyplot(fig)
    plt.close(fig)

if st.session_state.history:
    st.header("🏅 Game history")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
    st.metric("Average points per round", f"{history_df['Points'].mean():.1f}")
    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download game history as CSV",
        data=csv,
        file_name="prediction_game_history.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Project note: the supplied CSV is synthetic and is used to train the model. "
    "Live predictions are generated from the values entered in the form."
)
