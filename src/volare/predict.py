def forecast_volatility(model,df):
    """
    Given a trained volatility model and new input data, return predicted volatility.
    model: Trained volatility prediction model.
    df: Dataframe with features for prediction.
    Returns predicted volatility for the next horizon.
    """
    last_features = df['rolling_vol'].iloc[-1:]
    y_next = model.predict(last_features)
    return y_next[0]