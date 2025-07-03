import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("StudentPerformanceFactors.csv")
print("First 5 Rows")
print(df.head())

print(df.shape)

print(df.info())

print(df.isnull().sum())

print(df.drop(columns = ["Parental_Involvement","Access_to_Resources","Extracurricular_Activities","Motivation_Level",
                 "Internet_Access","Tutoring_Sessions","Family_Income","Teacher_Quality","School_Type","Peer_Influence","Physical_Activity",
                 "Learning_Disabilities","Parental_Education_Level","Distance_from_Home","Gender"], inplace=True))

print(df.shape)

print(df.describe())

print(df.head())

df.hist(figsize=(10,6), edgecolor='black')
plt.suptitle("Distribution of Features", fontsize=16)
plt.tight_layout()
plt.show()

for col in ['Hours_Studied','Attendance','Sleep_Hours','Previous_Scores','Exam_Score']:
  plt.figure(figsize=(6,4))
  sns.boxplot(y=df[col])
  plt.title(f"Boxplot of {col}")
  plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap= 'coolwarm')
plt.title("Correlation Heatmap")
plt.show()

outliers = df[(df['Exam_Score'] < df['Exam_Score'].quantile(0.01)) |
                (df['Exam_Score'] > df['Exam_Score'].quantile(0.99))]

print("\n🔹 Potential Exam Score Outliers:")
print(outliers[['Hours_Studied', 'Previous_Scores', 'Exam_Score']])

Q1 = df['Exam_Score'].quantile(0.25)
Q3 = df['Exam_Score'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap extreme scores to acceptable range
df['Exam_Score'] = df['Exam_Score'].clip(lower=lower_bound, upper=upper_bound)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Only the 4 selected features
X = df[['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores']]
y = df['Exam_Score']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Re-train a fresh model
model = LinearRegression()
model.fit(X_train, y_train)

# Create a new student's data
import pandas as pd

new_student = pd.DataFrame([{
    'Hours_Studied': 8.0,
    'Attendance': 75.0,
    'Sleep_Hours': 5.0,
    'Previous_Scores': 55.0
}])

# Predict exam score
predicted_score = model.predict(new_student)

print(f"🎯 Predicted Exam Score: {predicted_score[0]:.1f}")