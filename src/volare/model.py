from lightgbm import LGBMRegressor
import numpy as np

def split_data(df, feature_cols, target_col='rolling_future_vol', train_frac=0.8):
    """
    Split the data into training and testing sets.
    Splits chronologically because past values influence future values (dependent data).
    train_frac: Fraction of data to be used for training (high).
    Returns X_train, X_test, y_train, y_test.
    """
    df_model = df[feature_cols + [target_col]].dropna()
    
    X = df_model[feature_cols].values
    y = df_model[target_col].values
    
    # Chronological split
    n = len(X)
    split_idx = int(n * train_frac)
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def train_model(X_train,y_train,X_val=None,y_val=None,**kwargs):
    """
    Train a Light Gradient Boosting Machine to predict volatility
    Returns trained model.
    kwargs: Additional parameters for the model.
    """
    model = LGBMRegressor(**kwargs)
    model.fit(X_train, y_train)#eval_set=[(X_val, y_val)],eval_metric='rmse',early_stopping_rounds=50,verbose=50)
    return model

def evaluate_model(model,X_test,y_test_log,eps=1e-8):
    """
    Evaluate the trained model on test data.
    Returns Mean Squared Error (MSE) of the predictions.
    May return plots.
    """
    y_pred_log = model.predict(X_test)

    # #transform back from log to linear volatility
    y_pred_vol = np.exp(y_pred_log) - eps

    rmse_log = np.sqrt(np.mean((y_test_log - y_pred_log)**2))
    mae_log  = np.mean(np.abs(y_test_log - y_pred_log))

    return y_pred_log, y_pred_vol, rmse_log, mae_log