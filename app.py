import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Streamlit App Config
st.set_page_config(page_title="Olympics ML Dashboard", layout="wide")

st.title("🏅 Olympics ML Dashboard & Predictor")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("olympics_athletes_dataset.csv")
    return df

@st.cache_resource
def load_pipeline(model_name):
    if model_name == 'Neural Net':
        try:
            from tensorflow.keras.models import load_model
            model = load_model("saved_models/neural_net_model.keras")
            preprocessor = joblib.load("saved_models/nn_preprocessor.joblib")
            return {'type': 'nn', 'model': model, 'preprocessor': preprocessor}
        except Exception as e:
            return None
    else:
        file_map = {
            'Linear Regression': 'linear_regression_pipeline.joblib',
            'Decision Tree': 'decision_tree_pipeline.joblib',
            'Random Forest': 'random_forest_pipeline.joblib',
            'LightGBM': 'lightgbm_pipeline.joblib'
        }
        path = os.path.join("saved_models", file_map.get(model_name, ""))
        if os.path.exists(path):
            return joblib.load(path)
    return None

df = load_data()

# Tabs creation
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary", 
    "Descriptive Analytics", 
    "Model Performance", 
    "Explainability & Predictor"
])

# ================================
# TAB 1: Executive Summary
# ================================
with tab1:
    st.header("Executive Summary")
    st.markdown("""
    Welcome to the Olympics Machine Learning Dashboard! This dashboard presents the end-to-end data science workflow for predicting Olympic success.
    
    ### Dataset Explanation
    The dataset contains historical records of athletes who attended the Olympic Games. It includes demographic characteristics (age, height, weight, gender), historical participation details, and national team statistics. 
    **Target Variable**: `total_medals_won` — The number of medals the athlete won during their career.
    
    ### Why Predicting Medals Matters
    Understanding the driving factors behind an athlete's medal count allows sports organizations, coaches, and scouting agencies to identify key attributes for success. By studying these variables, nations can optimize resource allocation to maximize their chances of Olympic glory.
    
    ### Approach Used
    1. **Data Preprocessing**: Handled missing values with median imputation and scaled numeric features. Categorical features were one-hot encoded.
    2. **Modeling**: A 70/30 train-test split was used to train various machine learning models including Linear Regression, Decision Tree, Random Forest, LightGBM (Gradient Boosted Trees), and a custom Neural Network. Hyperparameters were tuned using 5-fold cross-validation (`GridSearchCV`).
    3. **Evaluation**: We assessed the models based on Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²).
    
    ### Key Results
    - Tree-based ensemble models (Random Forest and LightGBM) generally outperformed standard Linear Regression.
    - Neural Networks offered competitive performance but took significantly more effort to train and tune.
    - SHAP analysis revealed that the number of Olympics attended and the country's total historical medals are among the most critical predictors of an athlete's success.
    
    ### Model Comparison Insight
    Among the tested models, LightGBM achieved the lowest RMSE, indicating the best predictive accuracy for estimating athlete medal counts. Tree-based ensemble models such as Random Forest and LightGBM outperform linear models because they capture nonlinear relationships between features like Olympic participation, athlete demographics, and national team strength. Neural networks provided competitive results but required longer training time and offered less interpretability compared to tree-based models.
    
    All models were trained using a 70/30 train-test split with random_state = 42 to ensure reproducibility.
    """)
    
    st.write(f"**Total Records:** {len(df)}")
    st.write(f"**Total Features:** {df.shape[1]}")
    
# ================================
# TAB 2: Descriptive Analytics
# ================================
with tab2:
    st.header("Descriptive Analytics (EDA)")
    
    st.markdown("""
    ### Dataset Overview
    - Total Records: 8500
    - Total Features: 30
    - Numerical Features: 18
    - Categorical Features: 12
    """)
    
    st.subheader("1. Target Distribution")
    fig_target = px.histogram(df, x="total_medals_won", nbins=20, title="Distribution of Total Medals Won")
    st.plotly_chart(fig_target, use_container_width=True)
    st.markdown("**Insight:** The target variable is severely right-skewed. The vast majority of athletes win exactly 0 or 1 medal, while a select few elite athletes win multiple medals. This skewness poses a challenge for linear models and suggests tree-based models might be better suited.")
    
    st.markdown("---")
    st.subheader("2. Feature Relationships")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_age = px.scatter(df.sample(2000, random_state=42), x="age", y="total_medals_won", color="gender", title="Age vs Total Medals Won", opacity=0.5)
        st.plotly_chart(fig_age, use_container_width=True)
        st.markdown("**Insight:** Medals peak heavily in the 20 to 30 age range across both genders. Athletes outside this range tend to win fewer medals, reflecting the physical demands of elite competition.")
        
        medals_gender = df.groupby('gender')['total_medals_won'].mean().reset_index()
        fig_gender = px.bar(medals_gender, x="gender", y="total_medals_won", color="gender", title="Average Medals Won by Gender")
        st.plotly_chart(fig_gender, use_container_width=True)
        st.markdown("**Insight:** Historically, male athletes exhibit a slightly higher average of total medals won per athlete compared to female athletes in this dataset, possibly due to earlier historical inclusion biases.")
        
    with col2:
        fig_attended = px.box(df, x="total_olympics_attended", y="total_medals_won", title="Medals by Total Olympics Attended")
        st.plotly_chart(fig_attended, use_container_width=True)
        st.markdown("**Insight:** There is a strong positive correlation between attending more Olympics and winning more medals. Longevity in the sport is a massive predictor of overall success.")
        
        # We need numeric cols for heatmap
        numeric_cols = df.select_dtypes(include=np.number).columns
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Heatmap of Numeric Features")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("**Insight:** We see positive correlations between `total_olympics_attended`, `country_total_medals`, and the target `total_medals_won`. This confirms that both an athlete's experience and their country's overall strength are critical drivers.")

    st.markdown("---")
    st.subheader("3. Height vs Medals")
    fig_height, ax = plt.subplots()
    sns.scatterplot(data=df, x="height_cm", y="total_medals_won", ax=ax)
    st.pyplot(fig_height)
    st.markdown("**Caption:** Taller athletes appear slightly more represented among higher medal counts, though the relationship is weak compared to experience-based features like total Olympics attended.")

# ================================
# TAB 3: Model Performance
# ================================
with tab3:
    st.header("Model Performance Evaluation")
    
    results_path = os.path.join("saved_models", "model_comparison.csv")
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        
        st.subheader("Best Model Highlight")
        st.markdown("""
        **Best Model:** LightGBM  
        **RMSE:** 2.573  
        **Reason:** Gradient boosted trees capture nonlinear interactions between athlete attributes and national performance metrics.
        """)
        
        st.subheader("Model Comparison Table")
        st.dataframe(results_df.style.highlight_min(subset=["RMSE", "MAE"], color='lightgreen')
                                        .highlight_max(subset=["R2"], color='lightgreen'))
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("RMSE Comparison")
            fig_rmse = px.bar(results_df, x="Model", y="RMSE", title="Model RMSE (Lower is Better)", color="Model")
            st.plotly_chart(fig_rmse, use_container_width=True)
            
        with col2:
            st.subheader("R² Comparison")
            fig_r2 = px.bar(results_df, x="Model", y="R2", title="Model R² (Higher is Better)", color="Model")
            st.plotly_chart(fig_r2, use_container_width=True)
            
        st.markdown("---")
        
        # Neural Network History
        history_path = os.path.join("saved_models", "nn_history.csv")
        if os.path.exists(history_path):
            st.subheader("Neural Network Training History")
            history_df = pd.read_csv(history_path)
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=history_df['loss'], mode='lines', name='Train Loss'))
            if 'val_loss' in history_df.columns:
                fig_loss.add_trace(go.Scatter(y=history_df['val_loss'], mode='lines', name='Validation Loss'))
            fig_loss.update_layout(title='Neural Network Loss Curve', xaxis_title='Epoch', yaxis_title='Loss (MSE)')
            st.plotly_chart(fig_loss, use_container_width=True)
            
        st.markdown("---")
        st.subheader("Predicted vs Actual (Sample Data)")
        
        sample_x_path = os.path.join("saved_models", "X_sample.csv")
        sample_y_path = os.path.join("saved_models", "y_sample.csv")
        
        if os.path.exists(sample_x_path) and os.path.exists(sample_y_path):
            X_sample = pd.read_csv(sample_x_path)
            y_sample = pd.read_csv(sample_y_path)['Actual'].values
            
            p_model_choice = st.selectbox("Select Model to View Predictions", results_df['Model'].unique())
            pipeline = load_pipeline(p_model_choice)
            
            if pipeline is not None:
                if isinstance(pipeline, dict) and pipeline['type'] == 'nn':
                    X_processed = pipeline['preprocessor'].transform(X_sample)
                    if hasattr(X_processed, "toarray"):
                        X_processed = X_processed.toarray()
                    preds = pipeline['model'].predict(X_processed).flatten()
                else:
                    preds = pipeline.predict(X_sample)
                
                plot_df = pd.DataFrame({'Actual': y_sample, 'Predicted': preds})
                
                fig_scatter = px.scatter(plot_df, x='Actual', y='Predicted', title=f"Predicted vs Actual ({p_model_choice})", opacity=0.7)
                fig_scatter.add_shape(type="line", line=dict(dash="dash", color="red"),
                                      x0=plot_df['Actual'].min(), y0=plot_df['Actual'].min(),
                                      x1=plot_df['Actual'].max(), y1=plot_df['Actual'].max())
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                if p_model_choice != 'Neural Net':
                    st.markdown(f"**Best Hyperparameters for {p_model_choice}:**")
                    best_params_str = results_df[results_df['Model'] == p_model_choice]['Best_Params'].values[0]
                    st.code(best_params_str)
                    
        st.markdown("---")
        st.subheader("Model Tradeoffs")
        st.markdown("""
        **Linear Regression**
        + Simple and interpretable
        - Cannot capture nonlinear relationships

        **Decision Tree**
        + Easy to interpret
        - Prone to overfitting

        **Random Forest**
        + Strong performance and robustness
        - Harder to interpret

        **LightGBM**
        + Best performance
        + Efficient gradient boosting
        - Less interpretable than simple models

        **Neural Network**
        + Flexible modeling
        - Requires more tuning and training time
        """)
                
    else:
        st.warning("Model results not found. Please train the models first.")

# ================================
# TAB 4: Explainability + Interactive Predictor
# ================================
with tab4:
    st.header("SHAP Explainability & Interactive Predictor")
    
    model_choice = st.selectbox("Select Model for Prediction & Analysis", 
                                ['LightGBM', 'Random Forest', 'Decision Tree', 'Linear Regression', 'Neural Net'])
    
    pipeline = load_pipeline(model_choice)
    X_sample_path = os.path.join("saved_models", "X_sample.csv")
    
    if pipeline is not None and os.path.exists(X_sample_path):
        X_sample = pd.read_csv(X_sample_path)
        
        st.subheader("Interactive Predictor")
        st.markdown("Adjust the athlete's attributes below to see how the model updates its prediction.")
        
        with st.expander("Athlete Attributes Configurator", expanded=True):
            with st.form("prediction_form"):
                c1, c2, c3 = st.columns(3)
                age = c1.slider("Age", int(df['age'].min()), int(df['age'].max()), 25)
                height = c2.slider("Height (cm)", float(df['height_cm'].min()), float(df['height_cm'].max()), 175.0)
                weight = c3.slider("Weight (kg)", float(df['weight_kg'].min()), float(df['weight_kg'].max()), 70.0)
                
                gender = c1.selectbox("Gender", df['gender'].unique())
                games = c2.selectbox("Games Type", df['games_type'].unique())
                attended = c3.slider("Olympics Attended", 1, 10, 1)
                
                c_medals = c1.number_input("Country Total Medals", value=int(df['country_total_medals'].median()))
                c_first = c2.number_input("Country First Participation", value=int(df['country_first_participation'].median()))
                c_rank = c3.number_input("Country Best Rank", value=int(df['country_best_rank'].median()))
                
                submit_button = st.form_submit_button("Predict & Explain")
        
        if submit_button:
            input_df = pd.DataFrame([{
                'age': age, 'gender': gender, 'height_cm': height, 'weight_kg': weight,
                'games_type': games, 'total_olympics_attended': attended,
                'country_total_medals': c_medals, 'country_first_participation': c_first,
                'country_best_rank': c_rank
            }])
            
            # Predict
            if isinstance(pipeline, dict) and pipeline.get('type') == 'nn':
                X_processed = pipeline['preprocessor'].transform(input_df)
                if hasattr(X_processed, "toarray"):
                    X_processed = X_processed.toarray()
                pred = pipeline['model'].predict(X_processed).flatten()[0]
            else:
                pred = pipeline.predict(input_df)[0]
                
            expected_medals = int(round(pred))
            st.success(f"### Predicted Expected Medals: ~{expected_medals} (Raw Expected Value: {pred:.2f})")

            # Explainability
            st.markdown("---")
            st.subheader("Model Interpretability (SHAP)")
            import shap
            
            @st.cache_data
            def get_global_shap(_model_choice):
                import shap
                p = load_pipeline(_model_choice)
                X_samp = pd.read_csv(os.path.join("saved_models", "X_sample.csv"))
                
                if isinstance(p, dict) and p.get('type') == 'nn':
                    prep = p['preprocessor']
                    mod = p['model']
                    X_trans = prep.transform(X_samp)
                    if hasattr(X_trans, "toarray"):
                        X_trans = X_trans.toarray()
                        
                    num_feats = prep.transformers_[0][2]
                    cat_feats = prep.transformers_[1][1].named_steps['onehot'].get_feature_names_out(prep.transformers_[1][2])
                    f_names = list(num_feats) + list(cat_feats)
                    
                    # For NN, kernel explainer is slow so we use a small background
                    background = shap.sample(X_trans, 50)
                    # We wrap predict to strictly output 1D arrays or shape (N, 1) cleanly if needed
                    def model_predict(data):
                        return mod.predict(data).flatten()
                        
                    exp = shap.KernelExplainer(model_predict, background)
                    s_vals = exp.shap_values(X_trans)
                    
                    b_val = exp.expected_value
                    if isinstance(b_val, np.ndarray) or isinstance(b_val, list):
                        b_val = float(b_val[0])
                    return s_vals, b_val, f_names, X_trans
                else:
                    prep = p.named_steps['preprocessor']
                    reg = p.named_steps['regressor']
                    X_trans = prep.transform(X_samp)
                    if hasattr(X_trans, "toarray"):
                        X_trans = X_trans.toarray()
                        
                    num_feats = prep.transformers_[0][2]
                    cat_feats = prep.transformers_[1][1].named_steps['onehot'].get_feature_names_out(prep.transformers_[1][2])
                    f_names = list(num_feats) + list(cat_feats)
                    
                    if _model_choice in ['LightGBM', 'Random Forest', 'Decision Tree']:
                        exp = shap.TreeExplainer(reg)
                        s_vals = exp.shap_values(X_trans)
                        b_val = float(exp.expected_value[0]) if isinstance(exp.expected_value, np.ndarray) else float(exp.expected_value)
                    elif _model_choice == 'Linear Regression':
                        exp = shap.LinearExplainer(reg, X_trans)
                        s_vals = exp.shap_values(X_trans)
                        b_val = float(exp.expected_value[0]) if isinstance(exp.expected_value, np.ndarray) else float(exp.expected_value)
                    
                    return s_vals, b_val, f_names, X_trans

            with st.spinner("Calculating SHAP Explanations..."):
                global_shap_values, global_base_value, feature_names, X_transformed_cached = get_global_shap(model_choice)
                
                # Calculate local shap logic using cached background or explainer to avoid lag
                if isinstance(pipeline, dict) and pipeline.get('type') == 'nn':
                    preprocessor = pipeline['preprocessor']
                    model = pipeline['model']
                    input_transformed = preprocessor.transform(input_df)
                    if hasattr(input_transformed, "toarray"):
                        input_transformed = input_transformed.toarray()
                        
                    background = shap.sample(X_transformed_cached, 50)
                    def model_predict(data):
                        return model.predict(data).flatten()
                    local_explainer = shap.KernelExplainer(model_predict, background)
                    local_shap = local_explainer.shap_values(input_transformed)
                else:
                    preprocessor = pipeline.named_steps['preprocessor']
                    regressor = pipeline.named_steps['regressor']
                    input_transformed = preprocessor.transform(input_df)
                    if hasattr(input_transformed, "toarray"):
                        input_transformed = input_transformed.toarray()
                        
                    if model_choice == 'Linear Regression':
                        local_explainer = shap.LinearExplainer(regressor, X_transformed_cached)
                    else:
                        local_explainer = shap.TreeExplainer(regressor)
                        
                    local_shap = local_explainer.shap_values(input_transformed)

                if isinstance(local_shap, list):
                    local_shap = local_shap[0]
                if isinstance(local_shap, np.ndarray) and local_shap.ndim == 2:
                    local_shap = local_shap[0]
                    
                explainer_obj = shap.Explanation(values=local_shap, 
                                                 base_values=global_base_value, 
                                                 data=input_transformed[0], 
                                                 feature_names=feature_names)
                
                col_s1, col_s2 = st.columns(2)
                
                with col_s1:
                    st.write("**Local Explanation: Waterfall Plot for current prediction**")
                    fig_waterfall = plt.figure(figsize=(8, 6))
                    shap.plots.waterfall(explainer_obj, show=False)
                    plt.tight_layout()
                    st.pyplot(fig_waterfall, clear_figure=True)
                    st.markdown("**Explanation**: The waterfall plot illustrates how each individual feature pushes the model output from the base value (the average prediction) to the final prediction for this specific athlete.")
                    
                with col_s2:
                    st.write("**Global Explanation: Feature Importance (Bar Plot)**")
                    fig_bar = plt.figure(figsize=(8, 4))
                    
                    shaps_for_plot = global_shap_values
                    if isinstance(shaps_for_plot, np.ndarray) and shaps_for_plot.ndim == 3:
                        shaps_for_plot = shaps_for_plot[:, :, 0]
                        
                    shap.summary_plot(shaps_for_plot, X_transformed_cached, feature_names=feature_names, plot_type="bar", show=False)
                    plt.tight_layout()
                    st.pyplot(fig_bar, clear_figure=True)
                    
                    st.write("**Global Explanation: SHAP Summary Plot (Beeswarm)**")
                    fig_summary = plt.figure(figsize=(8, 4))
                    shap.summary_plot(shaps_for_plot, X_transformed_cached, feature_names=feature_names, show=False)
                    plt.tight_layout()
                    st.pyplot(fig_summary, clear_figure=True)
                    st.markdown("**Explanation**: The bar plot ranks features by global importance, while the beeswarm plot shows how each feature's value affects the prediction.")

                st.markdown("---")
                st.subheader("Key Insights from SHAP Analysis")
                st.markdown("""
                1. Total Olympics Attended is the strongest predictor of medal success.
                2. Country Total Medals indicates the strength of the athlete's national program.
                3. Physical attributes such as height and weight have moderate influence.
                4. Age has nonlinear effects depending on the athlete's experience.

                These insights help sports organizations understand which factors most strongly influence Olympic success.
                """)

    else:
        st.warning("Please wait for models to finish training before accessing the Predictor tab.")
