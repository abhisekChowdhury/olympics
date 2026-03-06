import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from scikeras.wrappers import KerasRegressor

def load_and_preprocess_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    target = 'total_medals_won'
    features = [
        'age', 'gender', 'height_cm', 'weight_kg', 'games_type',
        'total_olympics_attended', 'country_total_medals', 
        'country_first_participation', 'country_best_rank'
    ]
    
    df = df.dropna(subset=[target])
    
    X = df[features].copy()
    y = df[target].values
    
    print(f"Data shape: {X.shape}")
    return X, y

def build_preprocessor():
    numeric_features = ['age', 'height_cm', 'weight_kg', 'total_olympics_attended', 
                        'country_total_medals', 'country_first_participation', 'country_best_rank']
    categorical_features = ['gender', 'games_type']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def build_nn_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    filepath = 'olympics_athletes_dataset.csv'
    X, y = load_and_preprocess_data(filepath)
    
    preprocessor = build_preprocessor()
    
    # 2.1 Data Preparation - Train/Test Split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    os.makedirs('saved_models', exist_ok=True)
    results = []
    
    # Preprocess data for neural network
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert sparse to dense if needed for NN
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()
        
    input_dim = X_train_processed.shape[1]

    models = {
        'Linear Regression': {
            'model': LinearRegression(),
            'params': {}
        },
        'Decision Tree': {
            'model': DecisionTreeRegressor(random_state=42),
            'params': {
                'regressor__max_depth': [5, 10, None],
                'regressor__min_samples_leaf': [1, 5, 10]
            }
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'regressor__n_estimators': [50, 100],
                'regressor__max_depth': [10, 20, None]
            }
        },
        'LightGBM': {
            'model': LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            'params': {
                'regressor__n_estimators': [50, 100],
                'regressor__max_depth': [5, 10],
                'regressor__learning_rate': [0.01, 0.1]
            }
        }
    }
    
    print("Starting Model Training and Evaluation...")
    best_pipelines = {}
    
    for name, config in models.items():
        print(f"\nTraining {name}...")
        
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', config['model'])])
        
        if config['params']:
            # GridSearchCV for hyperparameter tuning
            search = GridSearchCV(pipeline, config['params'], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params_str = str(search.best_params_)
            print(f"Best parameters for {name}: {search.best_params_}")
        else:
            best_model = pipeline.fit(X_train, y_train)
            best_params_str = "{}"
            
        best_pipelines[name] = best_model
            
        y_pred = best_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name} Test Results:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}")
        
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Best_Params': best_params_str
        })
        
        joblib.dump(best_model, f'saved_models/{name.replace(" ", "_").lower()}_pipeline.joblib')

    # Neural Network Training
    print("\nTraining Neural Network...")
    nn_model = build_nn_model(input_dim)
    
    # Train the NN
    history = nn_model.fit(
        X_train_processed, y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=32,
        verbose=1
    )
    
    # Save history
    pd.DataFrame(history.history).to_csv('saved_models/nn_history.csv', index=False)
    
    # Evaluate NN
    y_pred_nn = nn_model.predict(X_test_processed).flatten()
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
    r2_nn = r2_score(y_test, y_pred_nn)
    
    print(f"Neural Net Test Results:")
    print(f"  MAE:  {mae_nn:.4f}")
    print(f"  RMSE: {rmse_nn:.4f}")
    print(f"  R2:   {r2_nn:.4f}")
    
    results.append({
        'Model': 'Neural Net',
        'MAE': mae_nn,
        'RMSE': rmse_nn,
        'R2': r2_nn,
        'Best_Params': "{'epochs': 50, 'batch_size': 32}"
    })
    
    # Save the Neural Net model and the preprocessor used for it
    nn_model.save('saved_models/neural_net_model.keras')
    joblib.dump(preprocessor, 'saved_models/nn_preprocessor.joblib')
    
    print("\nSaving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv('saved_models/model_comparison.csv', index=False)
    
    # Save a small sample of X and y for SHAP baseline and test plots
    np.random.seed(42)
    indices = np.random.choice(len(X_test), min(200, len(X_test)), replace=False)
    X_sample = X_test.iloc[indices]
    y_sample = y_test[indices]
    
    X_sample.to_csv('saved_models/X_sample.csv', index=False)
    pd.DataFrame({'Actual': y_sample}).to_csv('saved_models/y_sample.csv', index=False)
    
    print("Done! All pipelines and results saved to saved_models/ directory.")

if __name__ == '__main__':
    main()
