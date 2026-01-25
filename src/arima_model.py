import pmdarima as pm


def train_arima_model(
    train_series,
    seasonal: bool = False,
    m: int = 1
):
    """
    Train ARIMA model using auto_arima.
    """
    model = pm.auto_arima(
        train_series,
        seasonal=seasonal,
        m=m,
        stepwise=True,
        trace=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    return model


def arima_forecast(model, n_periods: int):
    """
    Generate ARIMA forecast.
    """
    forecast = model.predict(n_periods=n_periods)
    return forecast
