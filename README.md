# Beat the AI: Student Performance Prediction Game

A beginner-friendly Data Science mini-project built with Python, Streamlit and Linear Regression.

## What's included
- `app.py` — Streamlit game and dashboard
- `student_performance.csv` — illustrative synthetic dataset (1,200 rows)
- `requirements.txt` — Python dependencies

## Run it locally

1. Install Python 3.10 or newer.
2. Extract this ZIP file.
3. Open a terminal in the extracted project folder.
4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Start the game:

   ```bash
   streamlit run app.py
   ```

6. Streamlit will show a local URL, usually `http://localhost:8501`. Open it in your browser.

## How to play
1. Read the student's profile (study hours, attendance, previous score and sleep).
2. Guess the student's exam marks.
3. Lock your prediction.
4. Compare your guess and the AI's Linear Regression prediction with the actual score.
5. Earn up to 100 points per round.

## Data Science concepts demonstrated
- Data loading and exploration with Pandas
- Numerical data handling with NumPy
- Train/test split
- Linear Regression
- Model evaluation using MAE and R-squared
- Data visualization using Matplotlib
- Interactive application development using Streamlit

## Important project note
The included dataset is synthetic and generated for learning and demonstration. It does not represent real students and should not be used to make real academic decisions. The prediction target is generated from a simulated relationship between the input features and marks.

## Suggested presentation flow
1. Explain the problem: making a mark prediction from study-related features.
2. Show the dataset and its columns.
3. Explain the train/test split and Linear Regression.
4. Demonstrate one game round.
5. Explain MAE and R-squared.
6. Discuss limitations: synthetic data, simplified features and prediction uncertainty.
