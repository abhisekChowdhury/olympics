import nbformat as nbf

nb = nbf.v4.new_notebook()

text_1 = """\
# MSIS 522 HW1 - Data Science Workflow
This notebook contains the exploratory data analysis (EDA) and experimental pipeline for the Olympics ML dataset.

## PART 1 — Descriptive Analytics
### 1.1 Dataset Introduction
- **Dataset**: Olympics Athletes Dataset
- **Source**: Collected from historical Olympics data
- **Target Variable**: `total_medals_won`
- **Importance**: Predicting medals helps us understand the factors that contribute to an athlete's success, which can aid sports management and talent scouting.
"""

code_1 = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('olympics_athletes_dataset.csv')
print("Number of rows:", len(df))
print("Number of features:", len(df.columns))

numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(exclude=np.number).columns
print("Numeric features:", list(numeric_cols))
print("Categorical features:", list(categorical_cols))

df.head()
"""

text_2 = """\
### 1.2 Target Distribution
Plotting the distribution of `total_medals_won`.
"""

code_2 = """\
plt.figure(figsize=(8,5))
sns.histplot(df['total_medals_won'], bins=20, kde=False)
plt.title('Distribution of Total Medals Won')
plt.xlabel('Total Medals Won')
plt.ylabel('Count')
plt.show()
"""

text_3 = """\
**Interpretation**: The distribution is highly right-skewed. Most athletes win 0 or 1 medal, with very few outliers winning multiple medals.
"""

text_4 = """\
### 1.3 Feature Relationships
Visualizing interactions between features and the target variable.
"""

code_4 = """\
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Boxplot of Height vs Medals (Filtered for >0 to see spread better if needed, but we plot all)
sns.boxplot(ax=axes[0,0], x='total_medals_won', y='height_cm', data=df)
axes[0,0].set_title('Height by Total Medals Won')

# 2. Gender vs Medals
sns.barplot(ax=axes[0,1], x='gender', y='total_medals_won', data=df)
axes[0,1].set_title('Average Medals Won by Gender')

# 3. Age vs Medals
sns.scatterplot(ax=axes[1,0], x='age', y='total_medals_won', alpha=0.1, data=df)
axes[1,0].set_title('Age vs Total Medals Won')

# 4. Total Olympics Attended vs Medals
sns.barplot(ax=axes[1,1], x='total_olympics_attended', y='total_medals_won', data=df)
axes[1,1].set_title('Average Medals by Olympics Attended')

plt.tight_layout()
plt.show()
"""

text_5 = """\
**Interpretation**: 
1. Height vs Medals: Taller athletes on average seem to have slightly higher ranges in higher medal counts.
2. Gender vs Medals: Male athletes tend to have a slightly higher average of medals won in historical data.
3. Age vs Medals: Most medals are won by athletes in their 20s and 30s.
4. Total Olympics Attended: There is a strong linear relationship between attending more Olympics and winning more medals overall.
"""

text_6 = """\
### 1.4 Correlation Heatmap
"""

code_6 = """\
plt.figure(figsize=(10,8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
"""

text_7 = """\
**Interpretation**:
- `total_olympics_attended` has a moderate positive correlation with `total_medals_won`.
- `country_total_medals` is also somewhat positively correlated with individual success.
- This suggests that experience and strong national sports programs contribute significantly to individual Olympic success.
"""

text_8 = """\
## PART 2 — Predictive Analytics
We train several models (Linear Regression, Decision Tree, Random Forest, LightGBM, and an MLP Neural Network) to predict `total_medals_won`. We use a 70/30 train/test split.

*Note: For the full model training loop and GridSearchCV tuning, refer to `train_pipeline.py`. The models tuned there are saved and utilized in the Streamlit app.*
"""

code_8 = """\
# Example setup
from sklearn.model_selection import train_test_split
X = df.drop(columns=['total_medals_won'])
y = df['total_medals_won']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(text_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(text_3),
    nbf.v4.new_markdown_cell(text_4),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_markdown_cell(text_5),
    nbf.v4.new_markdown_cell(text_6),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_markdown_cell(text_7),
    nbf.v4.new_markdown_cell(text_8),
    nbf.v4.new_code_cell(code_8)
]

with open('notebook.ipynb', 'w') as f:
    nbf.write(nb, f)
print("notebook.ipynb generated successfully!")
