from energy_sales_forecast.forecasting import load_series, fit_model, forecast
import pandas as pd

def test_pipeline_runs(tmp_path):
    dates = pd.date_range("2023-01-31", periods=24, freq="M")
    s = pd.Series(range(24), index=dates)
    csv = tmp_path / "sales.csv"
    s.to_frame("sales").reset_index(names="date").to_csv(csv, index=False)
    series = load_series(csv)
    model = fit_model(series, order=(1,0,0))
    preds = forecast(model, periods=3)
    assert len(preds) == 3
