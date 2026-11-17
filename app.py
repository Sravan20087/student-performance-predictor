
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Beat the AI | Student Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. CUSTOM DARK GAMING THEME
# ==========================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background:
        radial-gradient(circle at top right, #29205b 0%, transparent 35%),
        linear-gradient(135deg, #080d1c 0%, #111a35 55%, #1d1235 100%);
    color: #f1f5f9;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Headings */
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 750 !important;
}

/* Normal text */
p, label, .stMarkdown {
    color: #dbeafe;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #172554, #312e81);
    border: 1px solid #6366f1;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 5px 20px #00000050;
}

/* Metric values */
div[data-testid="stMetricValue"] {
    color: #67e8f9 !important;
    font-weight: bold;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    background: linear-gradient(90deg, #6366f1, #06b6d4);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: bold;
    transition: 0.3s;
}

/* Button hover */
.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    box-shadow: 0 0 20px #06b6d4;
    transform: translateY(-2px);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101a35, #241442);
}

/* Expanders */
div[data-testid="stExpander"] {
    background: #172554;
    border: 1px solid #6366f1;
    border-radius: 12px;
}

/* Input boxes */
div[data-baseweb="input"],
div[data-baseweb="select"] {
    border-radius: 10px;
}

/* Forms */
div[data-testid="stForm"] {
    background: rgba(30, 41, 90, 0.35);
    border: 1px solid #414b86;
    border-radius: 16px;
    padding: 20px;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border: 1px solid #6366f1;
    border-radius: 12px;
}

/* Slider */
div[data-testid="stSlider"] {
    padding-top: 5px;
}

/* Divider */
hr {
    border-color: #414b86;
}

/* Info messages */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. DATASET CONFIGURATION
# ==========================================================

DATA_PATH = Path(__file__).parent / "student_performance.csv"

FEATURES = [
    "study_hours",
    "attendance",
    "previous_score",
    "sleep_hours"
]


# ==========================================================
# 4. LOAD DATASET
# ==========================================================

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "student_performance.csv not found. "
            "Upload it into the same GitHub folder as app.py."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + ["actual_marks"]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    df = df[required_columns].copy()

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()

    if len(df) < 20:
        raise ValueError(
            "At least 20 valid dataset records are required."
        )

    return df


# ==========================================================
# 5. TRAIN MACHINE LEARNING MODEL
# ==========================================================

@st.cache_resource
def train_model():

    df = load_dataset()

    X = df[FEATURES]
    y = df["actual_marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = np.clip(
        model.predict(X_test),
        0,
        100
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return (
        model,
        mae,
        r2,
        len(X_train),
        len(X_test)
    )


# Load and train
df = load_dataset()

model, mae, r2, train_count, test_count = train_model()


# ==========================================================
# 6. SESSION STATE
# ==========================================================

if "total_points" not in st.session_state:
    st.session_state.total_points = 0

if "rounds" not in st.session_state:
    st.session_state.rounds = 0

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ==========================================================
# 7. SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("🎮 GAME MENU")

    st.markdown("---")

    st.subheader("🏆 Player Score")

    st.metric(
        "Total Points",
        st.session_state.total_points
    )

    st.metric(
        "Rounds Played",
        st.session_state.rounds
    )

    st.markdown("---")

    st.subheader("📖 How to Play")

    st.write("""
    1. Enter student details.
    2. Predict the exam marks.
    3. Let the AI predict marks.
    4. Compare both predictions.
    5. Earn points!
    """)

    st.markdown("---")

    st.caption(
        "Beat the AI - Student Performance Prediction Game"
    )

    if st.button("🔄 Reset Game", use_container_width=True):

        st.session_state.total_points = 0
        st.session_state.rounds = 0
        st.session_state.history = []
        st.session_state.last_result = None

        st.rerun()


# ==========================================================
# 8. MAIN HEADER
# ==========================================================

st.markdown(
    "<h1 style='text-align:center;'>🎯 BEAT THE AI</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style='text-align:center;color:#67e8f9 !important;'>
    LIVE STUDENT PERFORMANCE PREDICTOR
    </h3>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align:center;font-size:17px;'>
    Can you predict student marks better than Artificial Intelligence?
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.info(
    "Educational project: This application uses a synthetic training "
    "dataset. Predictions are estimates, not guaranteed exam results."
)


# ==========================================================
# 9. STUDENT INPUT SECTION
# ==========================================================

st.header("🧑‍🎓 Enter Student Information")

st.write(
    "Adjust the student details below and make your prediction."
)

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        study_hours = st.slider(
            "📚 Study Hours Per Day",
            0.0,
            12.0,
            6.0,
            0.5
        )

        attendance = st.slider(
            "📅 Attendance Percentage",
            0,
            100,
            85
        )

    with col2:

        previous_score = st.slider(
            "📝 Previous Exam Marks",
            0,
            100,
            70
        )

        sleep_hours = st.slider(
            "😴 Sleep Hours Per Day",
            0.0,
            12.0,
            7.0,
            0.5
        )

    st.markdown("---")

    st.subheader("🎯 Your Prediction")

    player_guess = st.slider(
        "What marks do you think this student will score?",
        0,
        100,
        75
    )

    submitted = st.form_submit_button(
        "🚀 PREDICT MARKS & BEAT THE AI",
        use_container_width=True
    )


# ==========================================================
# 10. LIVE PREDICTION
# ==========================================================

if submitted:

    input_data = pd.DataFrame(
        [[
            study_hours,
            attendance,
            previous_score,
            sleep_hours
        ]],
        columns=FEATURES
    )

    ai_prediction = float(
        np.clip(
            model.predict(input_data)[0],
            0,
            100
        )
    )

    player_error = abs(
        player_guess - ai_prediction
    )

    points = max(
        0,
        int(round(100 - player_error))
    )

    if player_error < 5:
        performance = "Excellent Prediction! 🎉"

    elif player_error < 15:
        performance = "Good Prediction! 👍"

    else:
        performance = "Keep Practicing! 💪"

    result = {
        "Study Hours": study_hours,
        "Attendance": attendance,
        "Previous Marks": previous_score,
        "Sleep Hours": sleep_hours,
        "Your Prediction": player_guess,
        "AI Prediction": round(ai_prediction, 1),
        "Difference": round(player_error, 1),
        "Points": points,
        "Performance": performance
    }

    st.session_state.last_result = result

    st.session_state.total_points += points

    st.session_state.rounds += 1

    st.session_state.history.append(result)


# ==========================================================
# 11. DISPLAY PREDICTION RESULTS
# ==========================================================

result = st.session_state.last_result

if result is not None:

    st.markdown("---")

    st.header("🏆 PREDICTION RESULTS")

    st.success(result["Performance"])

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "👤 Your Prediction",
            f"{result['Your Prediction']}/100"
        )

    with col2:

        st.metric(
            "🤖 AI Prediction",
            f"{result['AI Prediction']}/100"
        )

    with col3:

        st.metric(
            "🏆 Points Earned",
            f"{result['Points']}/100"
        )

    st.markdown("---")

    st.subheader("📊 Prediction Comparison")

    chart_data = pd.DataFrame({
        "Prediction": [
            "Your Prediction",
            "AI Prediction"
        ],

        "Marks": [
            result["Your Prediction"],
            result["AI Prediction"]
        ]
    })

    fig, ax = plt.subplots(figsize=(8, 4))

    fig.patch.set_facecolor("#111a35")
    ax.set_facecolor("#111a35")

    bars = ax.bar(
        chart_data["Prediction"],
        chart_data["Marks"],
        width=0.45
    )

    bars[0].set_color("#818cf8")
    bars[1].set_color("#22d3ee")

    ax.set_ylim(0, 110)

    ax.set_ylabel(
        "Marks out of 100",
        color="white"
    )

    ax.set_title(
        "Player vs AI",
        color="white",
        fontsize=16
    )

    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("#64748b")

    ax.bar_label(
        bars,
        color="white",
        padding=4
    )

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        f"Difference between your prediction and AI: "
        f"{result['Difference']} marks."
    )

    st.caption(
        "For a new student, actual exam marks are not yet known. "
        "The game compares your guess with the AI estimate, not with "
        "a verified actual result."
    )


# ==========================================================
# 12. MODEL PERFORMANCE
# ==========================================================

st.markdown("---")

st.header("📈 MACHINE LEARNING MODEL PERFORMANCE")

m1, m2, m3 = st.columns(3)

m1.metric(
    "Mean Absolute Error",
    f"{mae:.2f} marks"
)

# Safe R2 formatting
try:

    r2_value = float(r2)

    if np.isfinite(r2_value):
        r2_display = f"{r2_value:.3f}"
    else:
        r2_display = "N/A"

except (TypeError, ValueError, OverflowError):

    r2_display = "N/A"

m2.metric(
    "R² Score",
    r2_display
)

m3.metric(
    "Training Records",
    f"{train_count:,}"
)

st.caption(
    f"Evaluation uses {test_count:,} held-out records. "
    "MAE measures average prediction error. "
    "R² measures explained variation."
)


# ==========================================================
# 13. DATASET EXPLORATION
# ==========================================================

st.markdown("---")

with st.expander("🔍 EXPLORE TRAINING DATASET"):

    st.subheader("Student Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Study Hours vs Exam Marks")

    fig, ax = plt.subplots(figsize=(8, 4))

    fig.patch.set_facecolor("#111a35")
    ax.set_facecolor("#111a35")

    ax.scatter(
        df["study_hours"],
        df["actual_marks"],
        color="#22d3ee",
        alpha=0.65
    )

    ax.set_xlabel(
        "Study Hours",
        color="white"
    )

    ax.set_ylabel(
        "Actual Marks",
        color="white"
    )

    ax.set_title(
        "Study Hours vs Student Performance",
        color="white"
    )

    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("#64748b")

    st.pyplot(fig)
    plt.close(fig)


# ==========================================================
# 14. GAME HISTORY
# ==========================================================

st.markdown("---")

st.header("🏅 PLAYER GAME HISTORY")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    average_points = history_df["Points"].mean()

    st.metric(
        "Average Points Per Round",
        f"{average_points:.1f}"
    )

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Game History",
        data=csv_data,
        file_name="student_prediction_history.csv",
        mime="text/csv"
    )

else:

    st.info(
        "Your game history will appear here after your first prediction."
    )


# ==========================================================
# 15. FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;padding:20px;'>

    <h3 style='color:#67e8f9 !important;'>
    🎯 BEAT THE AI
    </h3>

    <p>
    Student Performance Prediction Game
    </p>

    <p style='color:#94a3b8;'>
    Powered by Python | Pandas | NumPy | Scikit-learn | Streamlit
    </p>

    </div>
    """,
    unsafe_allow_html=True
)
