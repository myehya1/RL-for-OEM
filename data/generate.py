import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Parameters
system_size_kw = 5  # 5kW solar system
battery_capacity_kwh = 10  # 10kWh battery
battery_efficiency = 0.95  # 95% round-trip efficiency
battery_soc = 0.5 * battery_capacity_kwh  # Start at 50%
battery_soc_history = []

# Time range: 1 year hourly
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31, 23)
dates = pd.date_range(start=start_date, end=end_date, freq='h')

# Helper Functions
def generate_solar_irradiance(hour, day_of_year):
    # Simple sunrise-sunset model
    max_irradiance = 1000  # W/m^2
    sunrise = 6
    sunset = 18
    daylight_hours = sunset - sunrise
    if sunrise <= hour <= sunset:
        solar_angle = np.pi * (hour - sunrise) / daylight_hours
        irradiance = max_irradiance * np.sin(solar_angle)
        # Add seasonal variation
        season_factor = 0.9 + 0.1 * np.cos(2 * np.pi * (day_of_year - 173) / 365)
        irradiance *= season_factor
        # Random cloud cover
        irradiance *= np.random.normal(0.9, 0.1)
        return max(irradiance, 0)
    else:
        return 0

def generate_temperature(hour, day_of_year):
    avg_temp = 25 + 10 * np.sin(2 * np.pi * (day_of_year - 173) / 365)
    daily_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
    noise = np.random.normal(0, 1)
    return avg_temp + daily_variation + noise

def generate_load(hour, weekday):
    # Higher load in morning and evening
    base_load = 0.5  # kW
    if 6 <= hour <= 9 or 18 <= hour <= 23:
        load = base_load + np.random.uniform(1.5, 2.5)
    elif 12 <= hour <= 14:
        load = base_load + np.random.uniform(0.5, 1.0)
    else:
        load = base_load + np.random.uniform(0.2, 0.5)
    # Weekend randomness
    if weekday >= 5:
        load *= np.random.uniform(0.8, 1.2)
    return round(load, 2)

# Storage arrays
data = []

# Simulation
for timestamp in dates:
    hour = timestamp.hour
    day_of_year = timestamp.timetuple().tm_yday
    weekday = timestamp.weekday()

    irradiance = generate_solar_irradiance(hour, day_of_year)
    temperature = generate_temperature(hour, day_of_year)
    load = generate_load(hour, weekday)

    # Solar power generation
    panel_efficiency = 0.18
    area = (system_size_kw * 1000) / (panel_efficiency * 1000)  # rough estimate
    solar_power_kw = irradiance * panel_efficiency * area / 1000
    solar_power_kw = min(solar_power_kw, system_size_kw)

    # Battery and grid calculations
    solar_contribution = min(load, solar_power_kw)
    excess_solar = max(0, solar_power_kw - load)
    remaining_load = load - solar_contribution

    # Battery charging with excess solar
    if excess_solar > 0:
        charge_possible = min(excess_solar, (battery_capacity_kwh - battery_soc))
        battery_soc += charge_possible * battery_efficiency
        battery_charge_discharge = charge_possible
    # Battery discharging for remaining load
    elif remaining_load > 0:
        discharge_possible = min(remaining_load, battery_soc)
        battery_soc -= discharge_possible / battery_efficiency
        battery_charge_discharge = -discharge_possible
        remaining_load -= discharge_possible
    else:
        battery_charge_discharge = 0

    # Power from grid if battery is empty
    grid_usage = max(0, remaining_load)

    # Save battery SOC history
    battery_soc = max(0, min(battery_soc, battery_capacity_kwh))
    battery_soc_history.append(100 * battery_soc / battery_capacity_kwh)

    # Append data
    data.append([timestamp, irradiance, temperature, round(solar_power_kw, 2),
                 load, battery_soc_history[-1], grid_usage, battery_charge_discharge])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    'Timestamp', 'Solar_Irradiance (W/m²)', 'Temperature (°C)',
    'Solar_Power_Generated (kW)', 'Load_Demand (kW)',
    'Battery_SOC (%)', 'Grid_Usage (kW)', 'Battery_Charge/Discharge (kW)'
])

# Save to CSV (optional)
df.to_csv('C:/Users/mohamad/Desktop/thesis papers energy/code/data/synthetic_microgrid_data.csv', index=False)

# Show sample data
print(df.head())

# Plot sample day
sample_day = df[df['Timestamp'].dt.date == datetime(2024, 6, 21).date()]
sample_day.plot(x='Timestamp', y=['Solar_Power_Generated (kW)', 'Load_Demand (kW)', 'Grid_Usage (kW)'], title='Sample Day - Solar vs Load')
plt.show()
