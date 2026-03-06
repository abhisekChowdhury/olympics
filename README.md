# 🏅 Olympics Medal Prediction Dashboard

An end-to-end **machine learning pipeline and interactive dashboard** for predicting Olympic athlete medal success based on demographic attributes, experience, and national team performance.

Live App:  
https://olympics-model.streamlit.app

---

# Project Overview

This project implements the **complete data science workflow**:

1. Exploratory Data Analysis (EDA)
2. Feature engineering and preprocessing
3. Model training and comparison
4. Explainability using SHAP
5. Deployment using Streamlit

The goal is to **predict the total medals an athlete wins across their career** using historical Olympic data.

---

# Dataset

The dataset contains historical Olympic athlete records including:

- Athlete demographics (age, height, weight, gender)
- Participation history (Olympics attended)
- National performance metrics
- Medal counts

Dataset Summary:

| Metric | Value |
|------|------|
| Total Records | ~8500 |
| Total Features | 30 |
| Numerical Features | 14 |
| Categorical Features | 16 |
| Target Variable | `total_medals_won` |

---

# Machine Learning Models

We trained and evaluated multiple models:

| Model | Description |
|------|------|
| Linear Regression | Baseline interpretable model |
| Decision Tree | Simple nonlinear model |
| Random Forest | Ensemble tree model |
| LightGBM | Gradient boosted trees |
| Neural Network | Deep learning regression model |

Best Model: **LightGBM**

Performance:

RMSE: **2.573**

Tree-based ensemble models performed best because Olympic performance involves complex nonlinear relationships between experience, national support, and demographics.

---

# Model Explainability

We used **SHAP (SHapley Additive Explanations)** to interpret model predictions.

Key insights:

- Athlete experience (`total_olympics_attended`) is the strongest predictor
- National strength (`country_total_medals`) significantly boosts success
- Demographic attributes have smaller influence

This analysis helps sports organizations identify the most impactful drivers of Olympic success.

---

# Interactive Dashboard

The Streamlit app allows users to:

- Explore descriptive analytics
- Compare model performance
- Run predictions for hypothetical athletes
- Visualize SHAP explanations

Users can adjust features such as:

- Age
- Height
- Weight
- Olympics attended
- Country performance metrics

and instantly see the predicted medal outcome.

---

# Project Structure

```text
olympics/
│
├── app.py                      # Streamlit dashboard
├── train_pipeline.py           # Model training pipeline
├── olympics_athletes_dataset.csv # Dataset
├── generate_notebook.py        # Script to generate Jupyter notebook
├── notebook.ipynb              # Analysis notebook
│
├── saved_models/               # Saved trained models and pipelines
│
├── requirements.txt
└── README.md
```

---

# Running the Project

Clone the repository:

```bash
git clone https://github.com/abhisekChowdhury/olympics.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

# Tech Stack

- Python
- Pandas
- Scikit-Learn
- LightGBM
- TensorFlow / Keras
- SHAP
- Streamlit

---

# Key Takeaways

- Olympic success depends heavily on **experience and national infrastructure**
- Tree-based models outperform linear models for this problem
- Explainable AI (SHAP) provides transparency into model predictions

---

# Author

Abhisek Chowdhury

MSIS – University of Washington
