import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

from simulator.kubelka_munk import recipe_to_lab
from color.delta_e import ciede2000

def evaluate_predictions(y_true, y_pred):
    """
    Evaluate predictions using CIEDE2000.
    y_true: np.ndarray of shape (N, 3) [L, a, b]
    y_pred: np.ndarray of shape (N, 3) [L, a, b]
    Returns a dictionary of metrics.
    """
    de2000_errors = []
    for (L1, a1, b1), (L2, a2, b2) in zip(y_true, y_pred):
        de2000_errors.append(ciede2000(L1, a1, b1, L2, a2, b2))
    
    de2000_errors = np.array(de2000_errors)
    return {
        "mean_de00": np.mean(de2000_errors),
        "median_de00": np.median(de2000_errors),
        "max_de00": np.max(de2000_errors),
        "95th_p_de00": np.percentile(de2000_errors, 95)
    }

def eval_km_baseline(X: pd.DataFrame, y: pd.DataFrame):
    """
    Evaluate the analytical Kubelka-Munk model as a baseline.
    X: DataFrame with ['red_pct', 'blue_pct', 'yellow_pct', 'white_pct']
    y: DataFrame with ['L', 'a', 'b']
    """
    y_pred = []
    for _, row in X.iterrows():
        L, a, b = recipe_to_lab(
            row['red_pct'], row['blue_pct'], row['yellow_pct'], row['white_pct']
        )
        y_pred.append([L, a, b])
    
    return evaluate_predictions(y.values, np.array(y_pred))

def eval_xgboost_baseline(X: pd.DataFrame, y: pd.DataFrame, random_state=42):
    """
    Evaluate an XGBoost regressor baseline using 80/20 train/test split.
    Uses MultiOutputRegressor approach via predicting each Lab channel.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # XGBoost handles multiple outputs by default if we use sklearn API wrapper?
    # Actually, XGBRegressor doesn't natively support multi-output in all versions, 
    # but sklearn MultiOutputRegressor can wrap it. Let's use it.
    from sklearn.multioutput import MultiOutputRegressor
    
    model = MultiOutputRegressor(xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state
    ))
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return evaluate_predictions(y_test.values, y_pred)
