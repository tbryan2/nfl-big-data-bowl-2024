import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss, make_scorer
import numpy as np

# Assuming feature_engineering.py is in the same directory
from feature_engineering import create_dataset

def train_xgboost_model(df):
    """
    Train an XGBoost model using Brier Score Loss as the evaluation metric.

    Args:
        df: DataFrame with all necessary features and target variable.

    Returns:
        A trained XGBoost model.
    """
    final_df, features_list = create_dataset(df, game_id)  # Specify the game_id as needed
    X = final_df[features_list]
    y = final_df['contact_or_tackle_happened']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the XGBoost model and hyperparameter search space
    model = XGBClassifier(random_state=42)

    search_space = {
        'clf__max_depth': [2, 3, 4, 5, 6, 7, 8],
        'clf__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1.0],
        'clf__subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'clf__colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'clf__colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'clf__colsample_bynode': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'clf__reg_alpha': [0, 1, 5, 10],
        'clf__reg_lambda': [0, 1, 5, 10],
        'clf__gamma': [0, 0.1, 0.5, 1, 5, 10]

    }

    # Define a custom scorer for Brier Score Loss
    brier_scorer = make_scorer(brier_score_loss, needs_proba=True, greater_is_better=False)

    # Train the model using RandomizedSearchCV with Brier Score Loss
    opt = RandomizedSearchCV(model, search_space, n_iter=10, scoring=brier_scorer, cv=3, random_state=42)
    opt.fit(X_train, y_train)

    best_model = opt.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Calculate Brier Score Loss
    test_brier_score = brier_score_loss(y_test, y_pred_proba)
    print(f"Brier Score on Test Set: {test_brier_score}")

    return best_model

def evaluate_hit_probability(actual_df, optimal_agents_df, model):
    """
    Evaluate hit probability (using Brier Score Loss) for actual and optimal agent paths.

    Args:
        actual_df: DataFrame with actual player data.
        optimal_agents_df: DataFrame with optimal agent data.
        model: Trained XGBoost model.

    Returns:
        Tuple of DataFrames with probabilities for actual and optimal paths.
    """
    # Evaluate probabilities using the model
    actual_probabilities = model.predict_proba(actual_df)[:, 1]
    optimal_probabilities = model.predict_proba(optimal_agents_df)[:, 1]

    # Add probabilities to the dataframes
    actual_df['hit_probability'] = actual_probabilities
    optimal_agents_df['hit_probability'] = optimal_probabilities

    return actual_df, optimal_agents_df
