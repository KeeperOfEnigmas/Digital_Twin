import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler


class Log:
    """Write everything sent to stdout to both the console and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()



ANALOG_SENSORS = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAL_SENSORS = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "metroPT3_output.txt")
log_file = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = Log(sys.__stdout__, log_file)


def save_csv(table, name, index=True):
    """Save a DataFrame to OUTPUT_DIR/<name>.csv."""
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    table.to_csv(path, index=index)
    print(f"Saved {path}")



df = pd.read_csv('metropt+3+dataset/MetroPT3(AirCompressor).csv')
df = df.drop(columns=df.columns[0])
print(df)
save_csv(pd.concat([df.head(), df.tail()]), "data_preview")
print("-------------------------------")
print(f"Shape: {df.shape}") # (1516948, 16)
print("-------------------------------")


"Cleaning"
# No missing values.
print("Check for value completeness:")
print(df.info())
completeness = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "non_null": df.notna().sum(),
    "missing": df.isna().sum(),
})
save_csv(completeness, "completeness")
print("-------------------------------")
print("No missing values.")
print("-------------------------------")

print("Statistics of Analog Sensors:")
analog_stats = df[ANALOG_SENSORS].describe()
print(analog_stats)
save_csv(analog_stats, "analog_sensor_stats")
print("-------------------------------")
print("Potential outliers \n" \
      "TP2: Range [-0.032, 10.676], negative pressure values -> potential sensor malfunction or anomalies \n" \
      "TP3: Range [0.73, 10.302], reasonable pressure range \n" \
      "H1: Range [-0.036, 10.288], negative pressure values -> potential sensor malfunction or anomalies \n" \
      "DV_pressure: Range [-0.032, 9.844], negative pressure values -> potential sensor malfunction or anomalies \n" \
      "Reservoirs: Range[0.712, 10.3], should be close to TP3 \n" \
      "Oil_temperature: Range [15.4, 89.05], reasonable temperature range \n" \
      "Motor_current: Range [0.02, 9.295], reasonable current range")
print("-------------------------------")

print("Statistics of Digital Sensors:")
digital_stats = df[DIGITAL_SENSORS].describe()
print(digital_stats)
save_csv(digital_stats, "digital_sensor_stats")
print("-------------------------------")
unique_rows = []
for sensor in DIGITAL_SENSORS:
    unique_values = df[sensor].unique()
    print(f"Unique values for {sensor}: {unique_values}")
    unique_rows.append({"sensor": sensor, "unique_values": " ".join(map(str, unique_values))})
save_csv(pd.DataFrame(unique_rows), "digital_sensor_unique_values", index=False)
print("-------------------------------")
print("No outliers \n" \
      "COMP: Range [0, 1], binary sensor \n" \
      "DV_eletric: Range [0, 1], binary sensor \n" \
      "Towers: Range [0, 1], binary sensor \n" \
      "MPG: Range [0, 1], binary sensor \n" \
      "LPS: Range [0, 1], binary sensor \n" \
      "Pressure_switch: Range [0, 1], binary sensor \n" \
      "Oil_level: Range [0, 1], binary sensor \n" \
      "Caudal_impulses: Range [0, 1], binary sensor")
print("-------------------------------")



"Preparation"
# Normalize the analog sensor values to [0, 1]
print("Normalization:")
df_norm = df.copy()
scaler = MinMaxScaler()
df_norm[ANALOG_SENSORS] = scaler.fit_transform(df_norm[ANALOG_SENSORS])
normalized_stats = df_norm.describe()
print(normalized_stats)
save_csv(normalized_stats, "normalized_stats")
print("-------------------------------")
print("No alignment needed, all entries have the same length \n" \
      "No encoding needed, all the values are numeric \n" \
      "No aggregation needed, it leads to loss of information, expecially with the binary sensors")
print("-------------------------------")



"Exploratory Data Analysis"
ANALOG_UNITS = {"TP2": "bar", "TP3": "bar", "H1": "bar", "DV_pressure": "bar", "Reservoirs": "bar", "Oil_temperature": "°C", "Motor_current": "A"}

START = "2020-03-01 00:00"
DAYS = 0
HOURS = 6
MINUTES = 0

start = pd.Timestamp(START)
duration = pd.Timedelta(days=DAYS, hours=HOURS, minutes=MINUTES)
end = start + duration
timestamps = pd.to_datetime(df["timestamp"])
mask = (timestamps >= start) & (timestamps < end)
window_data = df.loc[mask].copy()
window_data["timestamp"] = timestamps[mask]

window_label = f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}"
file_label = f"{start:%Y-%m-%d_%H-%M}_{DAYS}d{HOURS}h{MINUTES}m"  # no ":" so it is a valid Windows filename

# Show the date on the x-axis only when the window spans more than one day
if duration > pd.Timedelta(days=1):
    time_format = "%m-%d %H:%M"
elif duration <= pd.Timedelta(minutes=10):
    time_format = "%H:%M:%S"
else:
    time_format = "%H:%M"

"Time-series Plots"
if duration <= pd.Timedelta(0):
    print("The window length must be greater than zero")
elif window_data.empty:
    print(f"No data from {window_label}")
else:
    for sensor in ANALOG_SENSORS:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(window_data["timestamp"], window_data[sensor], linewidth=0.6)
        ax.set_title(f"{sensor} from {window_label}")
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{sensor} [{ANALOG_UNITS[sensor]}]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # path = os.path.join(OUTPUT_DIR, f"{sensor}_{file_label}.png")
        path = os.path.join(OUTPUT_DIR, f"{sensor}.png")
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")

    # plt.show()

print("-------------------------------")
print("Analog singlas are periodic with very obvious repeated patterns \n" \
      "Failures are clearly visible in the plots where the patterns completely disapper")
print("-------------------------------")



"Correlation Matrix"
# Pearson correlation matrices over the whole dataset
correlation_sets = {"analog": ANALOG_SENSORS, "all_sensors": ANALOG_SENSORS + DIGITAL_SENSORS}

for name, sensors in correlation_sets.items():
    corr = df[sensors].corr()
    print(f"Correlation matrix ({name}):")
    print(corr.round(2))
    save_csv(corr, f"correlation_{name}")

    size = 1.2 + 0.75 * len(sensors)
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax, label="Correlation")

    ax.set_xticks(range(len(sensors)))
    ax.set_yticks(range(len(sensors)))
    ax.set_xticklabels(sensors, rotation=45, ha="right")
    ax.set_yticklabels(sensors)

    # Write the value in each cell; white text on the dark (strongly correlated) cells
    for i in range(len(sensors)):
        for j in range(len(sensors)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > 0.6 else "black")

    ax.set_title(f"Correlation matrix ({name.replace('_', ' ')})")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"correlation_{name}.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    print("-------------------------------")

# plt.show()

print("Very strong correlation bettwen TP3 and reservoirs, COMP and H1 and MPG, TP2 and DV_eletric \n" \
      "These correlations are suggested in the data description")
print("-------------------------------")




"Saving"
sys.stdout = sys.__stdout__
log_file.close()
print(f"Output saved to {LOG_FILE}")