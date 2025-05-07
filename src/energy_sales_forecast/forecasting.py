from pathlib import Path
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def load_series(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df.set_index("date")["sales"]

def fit_model(series: pd.Series, order=(1,1,1)):
    model = ARIMA(series, order=order).fit()
    return model

def forecast(model, periods: int = 12) -> pd.Series:
    preds = model.forecast(periods)
    return preds
