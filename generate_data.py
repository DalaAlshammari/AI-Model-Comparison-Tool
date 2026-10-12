import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
n = 200

study_hours = rng.uniform(0.5, 10, n).round(1)
attendance = rng.integers(45, 101, n)
previous_score = rng.integers(30, 101, n)

score = (
    0.42 * study_hours
    + 0.035 * (attendance - 45)
    + 0.035 * (previous_score - 30)
    + rng.normal(0, 0.9, n)
)

result = np.where(score >= 3.0, "Pass", "Fail")

df = pd.DataFrame({
    "StudyHours": study_hours,
    "Attendance": attendance,
    "PreviousScore": previous_score,
    "Result": result
})

df.to_csv("student_performance_200.csv", index=False)

print("CSV file created successfully!")
print(df.head(10))
print("\nNumber of rows:", len(df))
print("\nClass counts:")
print(df["Result"].value_counts())