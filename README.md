# 📊 Student Exam Score Predictor

A data science project that builds a predictive model to estimate student exam scores based on factors such as study habits, sleep patterns, attendance, and previous academic performance.

## 🚀 Overview

This project uses a real-world-like dataset and applies various data science techniques:
- Data cleaning & preprocessing
- Exploratory Data Analysis (EDA)
- Outlier handling using IQR capping
- Regression modeling (Linear, Decision Tree)
- Model evaluation with metrics and cross-validation
- Predicting new student scores using the trained model

## 📁 Dataset

**File:** `StudentPerformanceFactors.csv`  
**Columns Used:**
- `Hours_Studied`
- `Attendance`
- `Sleep_Hours`
- `Previous_Scores`
- `Exam_Score` (Target)

## 🛠️ Tools & Libraries

- Python 3.x
- pandas
- numpy
- seaborn & matplotlib
- scikit-learn

## 📈 Models Used

| Model               | R² Score | Mean Squared Error |
|--------------------|----------|---------------------|
| Linear Regression  | 0.75     | 2.79                |
| Decision Tree      | 0.65     | 3.89                |
| Cross-Validated LR | 0.735    | —                   |

**✅ Final model used:** Linear Regression

## 🔮 Example Prediction

```python
new_student = pd.DataFrame([{
    'Hours_Studied': 5.0,
    'Attendance': 85.0,
    'Sleep_Hours': 7.0,
    'Previous_Scores': 75.0
}])
predicted_score = model.predict(new_student)

📊 EDA Highlights
Hours_Studied and Attendance are strong predictors

Outliers capped using IQR

Sleep_Hours had weak impact

Visualized relationships using boxplots and histograms

🧠 Future Improvements
Try advanced models (Random Forest, XGBoost)

Add categorical features like school type or gender

Deploy using Flask or Streamlit

📂 How to Run
Clone the repo

Run pip install -r requirements.txt

Execute eda.py to train the model

Modify new_student to predict scores

👤 Author
Aiyan Ahmed
Aspiring Data Scientist | GitHub: @Codewithaiyan