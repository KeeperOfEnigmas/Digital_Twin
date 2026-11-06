import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



ANALOG_SENSORS = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAL_SENSORS = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]



df = pd.read_csv('metropt+3+dataset/MetroPT3(AirCompressor).csv')
df = df.drop(columns=df.columns[0])
print(df)
print("-------------------------------")
print(f"Shape: {df.shape}") # (1516948, 16)
print("-------------------------------")


"Cleaning"
# No missing values.
print("Check for value completeness:")
print(df.info())
print("-------------------------------")

# Potential ouliers
# TP2: Range [-0.032, 10.676], negative pressure values -> potential sensor malfunction or anomalies
# TP3: Range [0.73, 10.302], reasonable pressure range
# H1: Range [-0.036, 10.288], negative pressure values -> potential sensor malfunction or anomalies
# DV_pressure: Range [-0.032, 9.844], negative pressure values -> potential sensor malfunction or anomalies
# Reservoirs: Range[0.712, 10.3], should be close to TP3
# Oil_temperature: Range [15.4, 89.05], reasonable temperature range
# Motor_current: Range [0.02, 9.295], reasonable current range
print("Statistics of Analog Sensors:")
print(df[ANALOG_SENSORS].describe())
print("-------------------------------")

# No outliers
# COMP: Range [0, 1], binary sensor
# DV_eletric: Range [0, 1], binary sensor
# Towers: Range [0, 1], binary sensor
# MPG: Range [0, 1], binary sensor
# LPS: Range [0, 1], binary sensor
# Pressure_switch: Range [0, 1], binary sensor
# Oil_level: Range [0, 1], binary sensor
# Caudal_impulses: Range [0, 1], binary sensor
print("Statistics of Digital Sensors:")
print(df[DIGITAL_SENSORS].describe())
print("-------------------------------")
for sensor in DIGITAL_SENSORS:
    print(f"Unique values for {sensor}: {df[sensor].unique()}")
print("-------------------------------")



"Preparation"
# Normalize the analog sensor values to [0, 1]
df_norm = df.copy()
scaler = MinMaxScaler()
df_norm[ANALOG_SENSORS] = scaler.fit_transform(df_norm[ANALOG_SENSORS])
print(df_norm.describe())
print("-------------------------------")

# No encoding needed, all the values are numeric