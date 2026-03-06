# MSIS 522 HW1 - Olympics Data Science Project

## Overview
This repository contains a full end-to-end Machine Learning project aimed at predicting an athlete's total Olympic medals won, based on their physical attributes and country history.

## Deliverables Included
1. **`notebook.ipynb`**: Contains the Exploratory Data Analysis (EDA) showcasing data distributions, correlation heatmaps, and feature relationships.
2. **`train_pipeline.py`**: The main training script. It handles data preprocessing, scales numerical values, one-hot encodes categorical variables, and tunes machine learning models (Linear Regression, Decision Tree, Random Forest, LightGBM, and an MLP Neural Network) via GridSearchCV on a 70/30 split.
3. **`saved_models/`**: The directory holding the best tuned `.joblib` and `.keras` models, along with sample datasets and performance metrics (`model_comparison.csv`).
4. **`app.py`**: A fully functional 4-tab Streamlit dashboard allowing users to read an executive summary, view descriptive visualizations, inspect trained models' capabilities, and interactively predict medals while viewing SHAP explainability plots.
5. **`requirements.txt`**: The library dependencies required to seamlessly run this software.

## Running the Application Locally

1. Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the training script (Optional, models are already in `saved_models`):
```bash
python train_pipeline.py
```

3. Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
