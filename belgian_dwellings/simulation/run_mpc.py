from simugrid.simulation.config_parser import parse_config_file
from simugrid.misc.log_plot_micro import plot_simulation
from sklearn.metrics import mean_absolute_error

import pandas as pd

import json

import time
import os, sys

import numpy as np
import pandas as pd
import datetime


from belgian_dwellings.simulation.custom_classes import DayAheadEngie
from belgian_dwellings.simulation.mpc_ems import PerfectMPC, MPCRealistic
from belgian_dwellings.simulation.calibrate_mpc import get_energyplus_calibration
from simugrid.assets.energyplus import EnergyPlus
from belgian_dwellings.simulation.tmp_2023_config import tmp_2023_config

import matplotlib.pyplot as plt


def execute_mpc(
    config_file,
    cached_calibration=None,
    cached_ev_forecasting_values=None,
    mode="perfect",
    disable_enforced=False,
):
    if cached_calibration is None or cached_ev_forecasting_values is None:
        energyplus_calibration, ev_forecasting_values = get_energyplus_calibration(
            config_file
        )
        print("Calibration: ", energyplus_calibration)
        # print("EV Forecasting Values: ", ev_forecasting_values)
    else:
        energyplus_calibration = cached_calibration
        ev_forecasting_values = cached_ev_forecasting_values

    microgrid = parse_config_file(config_file)
    # debug = False
    # if debug:
    #    time_diff = datetime.timedelta(days=40)
    #    simulation_length = datetime.timedelta(days=10)
    #    microgrid.start_time += time_diff
    #    microgrid.datetime += time_diff
    #    microgrid.utc_datetime += time_diff
    #    microgrid.end_time = microgrid.start_time + simulation_length
    microgrid.config_file = config_file.split("/")[-1]

    reward = DayAheadEngie()
    microgrid.set_reward(reward)
    if mode == "perfect":
        ems = PerfectMPC(microgrid, energyplus_calibration)
    elif mode == "realistic":
        ems = MPCRealistic(microgrid, energyplus_calibration, ev_forecasting_values)

    if disable_enforced:
        ems.do_forced_charging = False

    end_time = microgrid.end_time
    # end_time = microgrid.start_time + datetime.timedelta(days=4)

    microgrid.attribute_to_log("EnergyPlus_0", "zn0_temp")
    microgrid.attribute_to_log("EnergyPlus_0", "cur_t_low")
    microgrid.attribute_to_log("EnergyPlus_0", "cur_t_up")
    microgrid.attribute_to_log("Charger_0", "soc")
    microgrid.attribute_to_log("WaterHeater_0", "t_tank")
    while microgrid.utc_datetime < end_time:
        # Print every new week
        if (
            microgrid.utc_datetime.weekday() == 0
            and microgrid.utc_datetime.hour == 0
            and microgrid.utc_datetime.minute == 0
        ) and True:
            # pass
            print(microgrid.utc_datetime)
        microgrid.management_system.simulate_step()

    microgrid.management_system.model.close()

    for asset in microgrid.assets:
        if isinstance(asset, EnergyPlus):
            asset.stop_energyplus_thread()

    # print(f"Final opex: {microgrid.tot_reward.KPIs['opex']}")
    # print(f"Final discomfort: {microgrid.tot_reward.KPIs['discomfort']}")
    return microgrid


if __name__ == "__main__":

    cached_calibration = [
        {
            "main_eff": np.float64(2.8725944279565594),
            "cool_eff": np.float64(4.878566662856399),
            "backup_eff": np.float64(1.0),
            "ga": np.float64(3.6555405676790595),
            "therm_cap": np.float64(4210095.559369944),
            "therm_res": np.float64(0.005035836234808133),
        }
    ]
    # cached_calibration = []
    # cached_calibration = None
    ev_forecasting_values = [[]]

    mode = "perfect"
    house_num = 2
    tot_houses = 500
    config_file = f"data/houses_belgium_{tot_houses}/house_{house_num}.json"
    config_path_2023 = tmp_2023_config(config_file)

    start_t = time.time()
    microgrid = execute_mpc(
        config_path_2023,
        cached_calibration,
        ev_forecasting_values,
        mode,
        disable_enforced=False,
    )

    forecaster = microgrid.management_system.forecaster
    mae_eval_log = forecaster.mae_evaluation_log
    logged_metrics = ["predicted", "previous"]
    predicted_values = ["detention", "soc_difference"]
    print(mae_eval_log)
    # Calculate MAE for predicted and previous values
    for metric in logged_metrics:
        for pred in predicted_values:
            mae = mean_absolute_error(
                mae_eval_log["real"][pred],
                mae_eval_log[metric][pred],
            )
            print(f"MAE for {metric} {pred}: {mae:.4f}")

    print(f"Execution time: {time.time() - start_t} seconds")
    KPIs = microgrid.tot_reward.KPIs
    print(f"Final opex: {KPIs['opex']}")
    print(f"Final discomfort: {KPIs['discomfort']}")
    print(f"Final quantile dissatisfaction: {KPIs['soc_quantile']}")
    plot_simulation(microgrid=microgrid)
    plt.show()
