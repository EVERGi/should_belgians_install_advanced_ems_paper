from simugrid.simulation.config_parser import parse_config_file

from belgian_dwellings.simulation.custom_classes import DayAheadEngie, TreeManager, HouseManager
import datetime
import math
import pytz


def first_input_function(param_evaluation):

    microgrid = param_evaluation["microgrid"]

    num_setpoints = microgrid.management_system.num_setpoints

    input_values = dict()
    input_bounds = dict()

    add_time_input(microgrid, input_values, input_bounds)
    add_asset_powers_inputs(microgrid, input_values, input_bounds)

    add_asset_specific_inputs(microgrid, input_values, input_bounds)

    add_weather_inputs(microgrid, input_values, input_bounds)
    add_price_inputs(microgrid, input_values, input_bounds)

    return [input_values for _ in range(num_setpoints)], [
        input_bounds for _ in range(num_setpoints)
    ]


def add_time_input(microgrid, input_values, input_bounds):

    utc_time = microgrid.utc_datetime
    hour = utc_time.hour
    minute = utc_time.minute
    time_of_day = hour + minute / 60

    name = "time of day (h)"
    bounds = [0, 24]
    input_values[name] = time_of_day
    input_bounds[name] = bounds

    name = "day of week"
    bounds = [0, 7]
    input_values[name] = utc_time.weekday()
    input_bounds[name] = bounds

    min_tm_yday = utc_time.replace(month=12, day=21).timetuple().tm_yday
    nr_of_days_per_year = utc_time.replace(month=12, day=31).timetuple().tm_yday

    cur_tm_yday = utc_time.timetuple().tm_yday
    norm_tm_yday = (cur_tm_yday - min_tm_yday) % nr_of_days_per_year

    rad_tm_yday = norm_tm_yday / nr_of_days_per_year * 2 * math.pi + math.pi
    cos_val = math.cos(rad_tm_yday)

    name = "cos time of year"
    bounds = [-1, 1]
    input_values[name] = cos_val
    input_bounds[name] = bounds


def add_asset_powers_inputs(microgrid, input_values, input_bounds):
    if microgrid.power_hist != []:
        power_hist = microgrid.power_hist[0]
    else:
        power_hist = None
    for asset in microgrid.assets:
        if power_hist is None:
            last_power = 0
        else:
            asset_hist = power_hist[asset.name]
            last_power = asset_hist[-1].electrical

        name = f"{asset.name} power (kW)"
        bounds = [-asset.max_consumption_power, asset.max_production_power]
        input_values[name] = last_power
        input_bounds[name] = bounds


def add_asset_specific_inputs(microgrid, input_values, input_bounds):
    for asset in microgrid.assets:
        if asset.name.startswith("EnergyPlus"):
            add_eplus_inputs(asset, input_values, input_bounds)
        elif asset.name.startswith("WaterHeater"):
            add_water_heater_inputs(asset, input_values, input_bounds)
        elif asset.name.startswith("Charger"):
            add_charger_inputs(asset, input_values, input_bounds)


def add_weather_inputs(microgrid, input_values, input_bounds):
    environment = microgrid.environments[0]
    cur_time = microgrid.utc_datetime
    time_step = microgrid.time_step

    start_dt = cur_time - time_step
    end_dt = cur_time
    if "epw_temp_air" in environment.env_values:
        temp_air_value = environment.env_values["epw_temp_air"]
        temp_air = temp_air_value.get_forecast(start_dt, end_dt)["values"][0]
        name = "air temp (°C)"
        bounds = [-20, 40]
        input_values[name] = temp_air
        input_bounds[name] = bounds

    if "epw_dir_norm_rad" in environment.env_values:
        irradiation_value = microgrid.environments[0].env_values["epw_dir_norm_rad"]
        irradiation = irradiation_value.get_forecast(start_dt, end_dt)["values"][0]

        name = "irradiation (W/m²)"
        bounds = [0, 1000]
        input_values[name] = irradiation
        input_bounds[name] = bounds


def add_price_inputs(microgrid, input_values, input_bounds):
    """
    Never used in the 100 pre training
    reward = microgrid.tot_reward
    if reward is None:
        peak_power = 2.5
    else:
        peak_power = reward.cur_peak_power

    name = "peak power (kW)"
    bounds = [0, 100]
    input_values[name] = peak_power
    input_bounds[name] = bounds
    """
    environment = microgrid.environments[0]
    cur_time = microgrid.utc_datetime
    time_step = microgrid.time_step

    hours_future = [6, 12, 24]

    day_ahead_delta = get_day_ahead_end_time_delta(microgrid)
    start_dt = cur_time
    end_dt = start_dt + day_ahead_delta
    day_ahead_value = environment.env_values["day_ahead_price"]
    day_ahead_prices = day_ahead_value.get_forecast(start_dt, end_dt)["values"]

    cur_price = day_ahead_prices[0]
    name = "current price (€/kWh)"
    bounds = [-0.5, 1]
    input_values[name] = cur_price
    input_bounds[name] = bounds

    for hour_future in hours_future:
        num_time_steps = int(hour_future * 3600 / time_step.total_seconds())
        future_prices = day_ahead_prices[:num_time_steps]
        norm_price_min = cur_price - min(future_prices)
        name = f"normalised price on min on next {hour_future}h (€/kWh)"
        bounds = [0, 1]
        input_values[name] = norm_price_min
        input_bounds[name] = bounds

        norm_price_max = max(future_prices) - cur_price

        name = f"normalised price on max on next {hour_future}h (€/kWh)"
        bounds = [0, 1]
        input_values[name] = norm_price_max
        input_bounds[name] = bounds


def add_water_heater_inputs(asset, input_values, input_bounds):
    time_step = asset.microgrid.time_step
    cur_time = asset.microgrid.utc_datetime
    environment = asset.microgrid.environments[0]

    t_tank = asset.t_tank

    name = "tank temp (°C)"
    bounds = [asset.low_setpoint, asset.high_setpoint]
    input_values[name] = t_tank
    input_bounds[name] = bounds

    """
    Never used in the 100 pre training
    environment_key = asset.environment_keys["hot_water_flow"][0]
    start_dt = cur_time - time_step
    end_dt = cur_time
    flow_value = environment.env_values[environment_key]
    previous_flow = flow_value.get_forecast(start_dt, end_dt)["values"][0]

    volume_m3 = asset.volume
    time_step_s = time_step.total_seconds()
    max_flow = volume_m3 / time_step_s

    name = "flow (m³/s)"
    bounds = [0, max_flow]
    input_values[name] = previous_flow
    """


def add_eplus_inputs(asset, input_values, input_bounds):
    time_step = asset.microgrid.time_step
    cur_time = asset.microgrid.utc_datetime

    readings = asset.get_readings()
    if readings is None:
        indoor_temp = 20
    else:
        indoor_temp = readings["indoor_temp_C"]
    name = "indoor temp (°C)"
    bounds = [5, 40]
    input_values[name] = indoor_temp
    input_bounds[name] = bounds

    times_steps_comfort = [0, 2, 8]
    for time_step_comfort in times_steps_comfort:
        dt_diff = time_step * time_step_comfort
        dt_diff_h = dt_diff.total_seconds() / 3600
        dt_comfort = cur_time + dt_diff
        comfort_range = asset.get_comfort_dt(dt_comfort, "approximated")

        name = f"lower comfort +{dt_diff_h}h (°C)"
        bounds = [5, 40]
        input_values[name] = comfort_range[0]
        input_bounds[name] = bounds

        # Removed because correlated to the previous one
        # inputs.append(comfort_range[1])
        # input_info.append([f"upper comfort +{dt_diff_h}h (°C)", [5, 40]])


def add_charger_inputs(asset, input_values, input_bounds):
    """
    Never used in the 100 pre training
    soc = asset.soc
    soc_init = asset.soc_min
    eff = asset.eff
    charged_energy = (soc - soc_init) / eff
    if soc == 0:
        charged_energy = -1

    name = "charged energy (kWh)"
    bounds = [-1, 100]
    input_values[name] = charged_energy
    input_bounds[name] = bounds
    """


def add_prev_setpoints(microgrid, input_values, input_bounds):
    # Good idea ?
    pass


def get_day_ahead_end_time_delta(microgrid):
    utc_datetime = microgrid.utc_datetime
    local_tz = pytz.timezone("Europe/Brussels")
    local_time = utc_datetime.astimezone(local_tz)

    start_day = local_time.replace(hour=0, minute=0, second=0)

    # Hour of the day when the day-ahead prices are available for the next day
    DAY_AHEAD_HOUR = 14
    if local_time.hour < DAY_AHEAD_HOUR:
        end_horizon = start_day + datetime.timedelta(days=1)
    else:
        end_horizon = start_day + datetime.timedelta(days=2)

    horizon_timedelta = end_horizon - local_time

    return horizon_timedelta


if __name__ == "__main__":
    fildir = "data/houses_belgium/"

    config_file = f"{fildir}house_0.json"
    microgrid = parse_config_file(config_file)

    reward = DayAheadEngie()
    microgrid.set_reward(reward)

    ems = HouseManager(microgrid)
    param_evaluation = {"microgrid": microgrid}
    input_values, input_bounds = first_input_function(param_evaluation)

    microgrid.management_system.simulate_step()

    param_evaluation = {"microgrid": microgrid}

    input_values, input_bounds = first_input_function(param_evaluation)
    print(input_values)
    print(input_bounds)
    # for i, input in enumerate(input_values[0]):
    #    print(f"{input_bounds[0][i][0]} {input} {input_bounds[0][i][1]}")
