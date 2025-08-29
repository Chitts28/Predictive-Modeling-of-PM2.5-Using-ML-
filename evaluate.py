import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load processed data
df = pd.read_csv("data/processed_pm25.csv")

X = df[["temperature", "humidity", "wind_speed", "month", "dayofweek"]]
y = df["PM2.5"]

# Load model
model = tf.keras.models.load_model("results/pm25_ann_model.h5")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Predictions
y_pred = model.predict(X_scaled).flatten()

# Metrics
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)

print(f"R² Score: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")
