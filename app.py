from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from catboost import Pool
from datetime import datetime

app = FastAPI(title="Flood Risk Prediction API")

# --- 1. Загрузка модели и данных ---
model = joblib.load('catboost_flood_cv_final.joblib')
df_history = pd.read_csv('flood-kz.csv', low_memory=False)
df_history['date'] = pd.to_datetime(df_history['date'])

# Статические признаки (для заполнения)
MEDIAN_VALUES = df_history.select_dtypes(include=np.number).median().to_dict()
STATIC_COLS = ['latitude','longitude','basin_mean_elevation','basin_mean_slope',
               'basin_twi','urban_area_pct','height_cm','region','major_city']
static_map = df_history.groupby('basin')[STATIC_COLS].first().reset_index().set_index('basin').T.to_dict()

CATEGORICAL_COLS = ['region','major_city','basin','season']

# --- 2. Pydantic модель для входных данных ---
class InputData(BaseModel):
    date: str
    basin: str
    water_level: float
    soil_moisture_avg: float
    temp_avg: float
    precip_sum: float

# --- 3. Вспомогательная функция: создание признаков ---
def create_input_data(data: InputData):
    input_data = pd.DataFrame(index=[0])
    
    # Текущие значения
    input_data['water_level'] = data.water_level
    input_data['soil_moisture_avg'] = data.soil_moisture_avg
    input_data['temp_avg'] = data.temp_avg
    input_data['precip_sum'] = data.precip_sum
    
    # Статические признаки
    static_data = static_map.get(data.basin, {})
    for feature in STATIC_COLS:
        input_data[feature] = static_data.get(feature, MEDIAN_VALUES.get(feature, 0))
    
    # Категориальные признаки
    input_data['basin'] = data.basin
    input_data['region'] = static_data.get('region', 'Unknown')
    input_data['major_city'] = static_data.get('major_city', 'Unknown')
    
    # Временные признаки (season)
    date_pred = pd.to_datetime(data.date)
    month = date_pred.month
    input_data['season'] = (month % 12 // 3 + 1)  # 1:Winter, 2:Spring, 3:Summer, 4:Autumn

    # Лаги и скользящие признаки
    df_basin = df_history[df_history['basin'] == data.basin].sort_values('date')
    prior_data = df_basin[df_basin['date'] < date_pred].tail(7)
    
    for col in ['water_level','water_discharge','precip_sum']:
        for lag in [1,3,7]:
            lag_val = prior_data[col].iloc[-lag] if len(prior_data) >= lag else np.nan
            input_data[f'{col}_lag_{lag}'] = lag_val

    # Скользящие средние и суммы
    for window in [3,7]:
        input_data[f'water_level_roll_mean_{window}'] = prior_data['water_level'].tail(window).mean() if len(prior_data) >= window else np.nan
        input_data[f'precip_roll_sum_{window}'] = prior_data['precip_sum'].tail(window).sum() if len(prior_data) >= window else np.nan

    # Производные признаки
    wl = input_data['water_level'].iloc[0]
    wl_lag_1 = input_data.get('water_level_lag_1', pd.Series([np.nan]))[0]
    wl_lag_3 = input_data.get('water_level_lag_3', pd.Series([np.nan]))[0]
    input_data['level_change_1d'] = wl - wl_lag_1
    input_data['level_change_3d'] = wl - wl_lag_3
    wd = input_data.get('water_discharge', pd.Series([0]))[0]
    input_data['discharge_per_level'] = wd / (wl + 10)

    # Заполнение NaN медианами
    input_data = input_data.fillna(MEDIAN_VALUES)
    return input_data

# --- 4. Эндпоинт для прогноза ---
@app.post("/predict")
def predict(input_data: InputData):
    df_input = create_input_data(input_data)
    pool = Pool(df_input, cat_features=CATEGORICAL_COLS)
    prob_flood = model.predict_proba(pool)[0][1]
    prediction = 1 if prob_flood > 0.5 else 0
    
    # Топ фичи (можно просто показать все входные признаки)
    top_features = df_input.to_dict(orient='records')[0]
    
    # Рекомендации
    rules = "Принять меры: контроль уровня воды, оповещение служб" if prediction==1 else "Мониторинг: продолжать наблюдение"
    
    return {
        "prob_flood": prob_flood,
        "prediction": prediction,
        "top_features": top_features,
        "rules": rules
    }
