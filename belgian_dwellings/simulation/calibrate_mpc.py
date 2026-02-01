from simugrid.simulation.config_parser import parse_config_file
from simugrid.assets.energyplus import EnergyPlus
from simugrid.assets.charger import Charger

import pandas as pd

import json

import os, sys

import numpy as np
import pandas as pd

from belgian_dwellings.simulation.custom_classes import DayAheadEngie, HouseManager
from belgian_dwellings.simulation.mpc_ems import MPCManager
from belgian_dwellings.simulation.forecaster import get_new_ev_forecasting_values

import scipy.optimize


def execute_config(config_file):

    microgrid = parse_config_file(config_file)
    energyplus = []
    chargers = []
    for asset in microgrid.assets:
        if isinstance(asset, EnergyPlus):
            energyplus.append(asset)
        if isinstance(asset, Charger):
            chargers.append(asset)

    for asset in energyplus:
        asset.stop_energyplus_thread()
        tc_var = {
            "zn0_temp": [
                "Zone Mean Air Temperature",
                "conditioned space",
            ],  # deg C
            "fan_electricity": [
                "Fan Electricity Energy",
                "air source heat pump supply fan",
            ],  # J for each timestep
            "heating_backup_coil_energy": [
                "Heating Coil Heating Energy",
                "air source heat pump backup htg coil",
            ],  # J for each timestep: Usefull but not used
            "heating_backup_coil_electricity_energy": [
                "Heating Coil Electricity Energy",
                "air source heat pump backup htg coil",
            ],  # J for each timestep
            "heating_coil_energy": [
                "Heating Coil Heating Energy",
                "air source heat pump htg coil",
            ],  # J for each timestep: Usefull but not used
            "heating_coil_electricity_energy": [
                "Heating Coil Electricity Energy",
                "AIR SOURCE HEAT PUMP HTG COIL",
            ],  # J for each timestep
            # Heating Coil Crankcase Heater Electricity Energy
            "heating_coil_crankcase_heater_electricity": [
                "Heating Coil Crankcase Heater Electricity Energy",
                "AIR SOURCE HEAT PUMP HTG COIL",
            ],  # J for each timestep
            # Heating Coil Defrost Electricity Energy
            "heating_coil_defrost_electricity": [
                "Heating Coil Defrost Electricity Energy",
                "AIR SOURCE HEAT PUMP HTG COIL",
            ],  # J for each timestep
            "cooling_coil_electricity": [
                "Cooling Coil Electricity Energy",
                "AIR SOURCE HEAT PUMP CLG COIL",
            ],  # J for each timestep
            "cooling_coil_total_energy": [
                "Cooling Coil Total Cooling Energy",
                "air source heat pump clg coil",
            ],  # J for each timestep: Usefull but not used
        }
        new_attribute = {"tc_var": tc_var}
        asset.set_attributes(new_attribute)

    reward = DayAheadEngie()
    microgrid.set_reward(reward)

    # Add EMS to microgrid
    HouseManager(microgrid)

    calibration_values = [dict() for _ in range(len(energyplus))]
    ev_forecasting_values = [dict() for _ in range(len(chargers))]

    env_values = microgrid.environments[0].env_values
    while microgrid.utc_datetime < microgrid.end_time:
        # Print every new week
        if (
            microgrid.utc_datetime.weekday() == 0
            and microgrid.utc_datetime.hour == 0
            and microgrid.utc_datetime.minute == 0
        ):
            pass
            # print(microgrid.utc_datetime)

        for i, asset in enumerate(energyplus):
            readings = asset.get_readings()

            temp_air = env_values["epw_temp_air"].value
            dir_norm_rad = env_values["epw_dir_norm_rad"].value

            if readings is not None:
                readings["temp_air"] = temp_air
                readings["dir_norm_rad"] = dir_norm_rad
                for key, value in readings.items():
                    if key not in calibration_values[i].keys():
                        calibration_values[i][key] = [value]
                    else:
                        calibration_values[i][key].append(value)
                # log_readings(readings, log_file, first_step)
        for i, charger in enumerate(chargers):
            if env_values[f"det_{charger.ID}"].value != 0:
                new_forecasting_values = get_new_ev_forecasting_values(
                    microgrid, charger
                )
                # If the charger is just connected, log the soc
                for key, value in new_forecasting_values.items():
                    if key not in ev_forecasting_values[i].keys():
                        ev_forecasting_values[i][key] = [value]
                    else:
                        ev_forecasting_values[i][key].append(value)

            prev_det = charger.det

        microgrid.management_system.simulate_step()
        first_step = False
    for asset in microgrid.assets:
        if isinstance(asset, EnergyPlus):
            asset.stop_energyplus_thread()

    return calibration_values, ev_forecasting_values


def log_readings(readings, log_file, first_step):
    log_file = "calibrate_mpc.csv"

    if first_step:
        with open(log_file, "w+") as f:
            readings_header = readings.keys()
            readings_header = ",".join(readings_header)
            f.write(f"{readings_header},datetime\n")

    with open(log_file, "a+") as f:
        readings_values = readings.values()
        readings_values = ",".join(map(str, readings_values))
        f.write(f"{readings_values}\n")


def read_calibration_values():
    df = pd.read_csv("calibrate_mpc.csv")

    calibration_values = dict()
    for col in df.columns:
        calibration_values[col] = df[col].tolist()

    return calibration_values


def calc_efficiency(calibration_values):
    efficiencies = ["main", "cool", "backup"]

    dict_eff = dict()
    for eff in efficiencies:
        electricity_output = [
            k for k in calibration_values.keys() if eff in k and "electricity" in k
        ][0]
        energy_output = [
            k for k in calibration_values.keys() if eff in k and "energy" in k
        ][0]

        x = np.array(calibration_values[electricity_output])
        y = calibration_values[energy_output]

        # Get slope and intercept
        # m, b = np.polyfit(x, y, 1)

        x = x[:, np.newaxis]
        a, _, _, _ = np.linalg.lstsq(x, y)
        dict_eff[eff + "_eff"] = a[0]

    return dict_eff


def calibration_func(x, gA, therm_cap, therm_res):
    previous_temp = x[0]
    thermal_power = x[1]
    solar_irradiation = x[2]
    outdoor_temp = x[3]
    dt_s = x[4]
    next_temp = MPCManager.temp_function(
        previous_temp,
        thermal_power,
        solar_irradiation,
        outdoor_temp,
        gA,
        therm_cap,
        therm_res,
        dt_s,
    )

    return next_temp


def calc_therma_properties(calibration_values, plot=False):
    # Get the values for the thermal properties
    previous_temp = np.array(calibration_values["indoor_temp_C"])[:-1]
    thermal_power = (
        np.array(calibration_values["heat_energy_main_kWh"])
        + np.array(calibration_values["heat_energy_backup_kWh"])
        - np.array(calibration_values["cool_energy_kWh"])
    )[1:]
    dict_prop = dict()
    solar_irradiation = np.array(calibration_values["dir_norm_rad"])[:-1]
    outdoor_temp = np.array(calibration_values["temp_air"])[:-1]
    dt = calibration_values["datetime"][:-1]
    dt = pd.to_datetime(dt)

    # Get the time step
    dt_s_diff = (dt[1] - dt[0]).total_seconds()
    dt_s = np.array([dt_s_diff] * len(dt))

    # To Watts
    thermal_power = thermal_power * 1000 / (dt_s_diff / 3600)

    x = np.array([previous_temp, thermal_power, solar_irradiation, outdoor_temp, dt_s])

    y = np.array([previous_temp[1:]])

    # Add 1 temperature to y
    y = np.append(y, previous_temp[-1])

    # Get the bounds
    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

    popt, pcov = scipy.optimize.curve_fit(calibration_func, x, y, bounds=bounds)
    dict_prop["ga"] = popt[0]
    dict_prop["therm_cap"] = popt[1]
    dict_prop["therm_res"] = popt[2]

    if plot:
        # Plot fitted values
        plot_fitted_values(x, y, popt)
        diff = y - calibration_func(x, *popt)
        errors = np.abs(diff)
        error = np.mean(errors)
        std_error = np.std(errors)
        print(f"MAE thermal model: {error}")
        print(f"STD thermal model: {std_error}")

    return dict_prop


def plot_fitted_values(x, y, popt):
    import matplotlib.pyplot as plt

    # plt.plot(x[0], y, "b-", label="data")

    calibrated_y = calibration_func(np.array(x), *popt)

    plt.scatter(y, calibrated_y)

    print(f"R2: {np.corrcoef(y, calibrated_y)[0, 1] ** 2}")
    print(f"MAE: {np.mean(np.abs(y - calibrated_y))}")
    plt.show()


def get_energyplus_calibration(config_file, plot=False):
    energyplus_calibrations = []

    calibration_list, ev_forecasting_values = execute_config(config_file)

    for calibration_values in calibration_list:
        dict_eff = calc_efficiency(calibration_values)

        dict_prop = calc_therma_properties(calibration_values, plot=plot)

        ditc_all = dict()
        ditc_all.update(dict_eff)
        ditc_all.update(dict_prop)
        energyplus_calibrations.append(ditc_all)

    return energyplus_calibrations, ev_forecasting_values


if __name__ == "__main__":
    for i in range(10):
        config_file = f"data/houses_belgium/house_{i}.json"
        json_dict = json.loads(open(config_file).read())
        for key, asset in json_dict["Assets"].items():
            if asset["name"] == "EnergyPlus_0":
                house_model = asset["idf_model"].replace(
                    "./../common/energyplus_models/", ""
                )
                print(house_model)
        calibration_list, ev_forecasting_values = execute_config(config_file)
        if len(calibration_list) == 0:
            continue
        for calibration_values in calibration_list:
            dict_eff = calc_efficiency(calibration_values)

            dict_prop = calc_therma_properties(calibration_values, plot=True)

            # Print the results in one line
            print("Efficiencies")
            res_eff = [f"{k}:{v:.4f}" for k, v in dict_eff.items()]
            res_eff = " ".join(res_eff)
            print(res_eff)

            print("Properties")
            res_prop = [f"{k}:{v:.4f}" for k, v in dict_prop.items()]
            res_prop = " ".join(res_prop)
            print(res_prop)
            print()
