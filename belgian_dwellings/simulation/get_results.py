import os

from belgian_dwellings.simulation.rbc_ems import execute_rule_base
from belgian_dwellings.simulation.run_mpc import execute_mpc, get_energyplus_calibration
from belgian_dwellings.simulation.run_treec import execute_treec, find_best_tree_for_each_house
from belgian_dwellings.simulation.tmp_2023_config import tmp_2023_config
import datetime

from filelock import FileLock
import multiprocessing
import json
from simugrid.misc.log_plot_micro import (
    log_power_hist,
    log_all_attributes,
    log_reward_hist,
    plot_simulation,
)
import shutil

import matplotlib.pyplot as plt


def get_results(
    folder,
    result_file,
    ems_name,
    refresh=False,
    single_threaded=False,
    num_process=None,
    config_list=None,
):

    if not os.path.exists(result_file):
        with open(result_file, "w") as f:
            f.write(
                "config_file,ems_name,opex (€),discomfort (Kh),consumption (kWh),grid_offtake (kWh), grid_injection (kWh)\n"
            )

    config_files = sorted(os.listdir(folder))

    if num_process is None:
        num_process = multiprocessing.cpu_count() - 1 or 1
    if not single_threaded:
        pool = multiprocessing.Pool(processes=num_process)
    tmp_config_dir = None
    for config_file in config_files:
        if config_list is not None and config_file not in config_list:
            continue
        print(f"Processing {config_file}")
        config_path = os.path.join(folder, config_file)
        tmp_config_path = tmp_2023_config(config_path)
        tmp_config_dir = os.path.dirname(tmp_config_path)
        if single_threaded:
            execute_single_result(
                tmp_config_path, result_file, ems_name, refresh, config_path
            )
        else:
            pool.apply_async(
                execute_single_result,
                args=(tmp_config_path, result_file, ems_name, refresh, config_path),
            )
    if not single_threaded:
        pool.close()
        pool.join()

    # Sort the result file by config_file name
    with open(result_file, "r") as f:
        lines = f.readlines()
    lines = [lines[0]] + sorted(
        lines[1:], key=lambda x: int(x.split(",")[0].split("_")[-1].split(".")[0])
    )
    with open(result_file, "w") as f:
        f.writelines(lines)

    return tmp_config_dir


def execute_single_result(config_path, result_file, ems_name, refresh, old_config_path):
    print(f"Executing {config_path} with {ems_name}")
    config_file = os.path.basename(config_path)
    with open(result_file, "r") as f:
        lines = f.readlines()
    if refresh:
        lines = [
            line for line in lines if not line.startswith(f"{config_file},{ems_name},")
        ]

    if not any(line.startswith(f"{config_file},{ems_name},") for line in lines):
        if ems_name.startswith("RBC"):
            microgrid = rule_base_execution(config_path, ems_name)
        if ems_name.startswith("MPC"):
            microgrid = mpc_execution(config_path, ems_name, old_config_path)
        if ems_name.startswith("TreeC"):
            microgrid = treec_execution(config_path, ems_name)
        log_results(microgrid, result_file, config_file, ems_name)
    else:
        print(f"{config_file} already exists in the result file")

    # microgrid = None
    # Garbage collection
    # gc.collect()


def log_results(microgrid, result_file, config_file, ems_name):
    log_kpis(microgrid, result_file, config_file, ems_name)

    log_microgrid(microgrid, result_file, config_file, ems_name)


def log_kpis(microgrid, result_file, config_file, ems_name):
    opex = microgrid.tot_reward.KPIs["opex"]
    discomfort = microgrid.tot_reward.KPIs["discomfort"]
    print(config_file)
    print(f"opex (€): {opex}, discomfort (Kh): {discomfort}")
    consumption = get_consumption(microgrid)
    consumption_only = consumption["consumption_only"]
    grid_offtake = consumption["grid_offtake"]
    grid_injection = consumption["grid_injection"]

    lock = FileLock(result_file + ".lock")
    with lock:
        with open(result_file, "r") as f:
            lines = f.readlines()

        lines = [
            line for line in lines if not line.startswith(f"{config_file},{ems_name},")
        ]

        lines.append(
            f"{config_file},{ems_name},{opex},{discomfort},{consumption_only},{grid_offtake},{grid_injection}\n"
        )

        with open(result_file, "w") as f:
            f.writelines(lines)

    if "no_enforcement" in ems_name:
        soc_file = result_file.replace(".csv", "_soc.json")
        lock = FileLock(soc_file + ".lock")
        with lock:
            log_soc_data(microgrid, soc_file, config_file, ems_name)
    if "realistic" in ems_name:
        mae_log_file = result_file.replace(".csv", "_mae.json")
        lock = FileLock(mae_log_file + ".lock")
        with lock:
            log_mae_score(microgrid, mae_log_file, config_file, ems_name)


def log_soc_data(microgrid, soc_file, config_file, ems_name):
    soc_data = microgrid.tot_reward.soc_diff_list
    # Create a JSON file if doesn't exist otherwise append to it
    # The json file is structured with as first key the config_file, the second key the ems_name
    if os.path.exists(soc_file):
        with open(soc_file, "r") as f:
            soc_json = json.load(f)
    else:
        soc_json = {}
    if config_file not in soc_json.keys():
        soc_json[config_file] = {}
    soc_json[config_file][ems_name] = soc_data

    with open(soc_file, "w") as f:
        json.dump(soc_json, f, indent=4)


def log_mae_score(microgrid, mae_log_file, config_file, ems_name):
    forecaster = microgrid.management_system.forecaster
    mae_eval_log = forecaster.mae_evaluation_log
    # Create a JSON file if doesn't exist otherwise append to it
    if os.path.exists(mae_log_file):
        with open(mae_log_file, "r") as f:
            mae_json = json.load(f)
    else:
        mae_json = {}
    if config_file not in mae_json.keys():
        mae_json[config_file] = {}
    mae_json[config_file][ems_name] = mae_eval_log
    with open(mae_log_file, "w") as f:
        json.dump(mae_json, f, indent=4)


def log_microgrid(microgrid, result_file, config_file, ems_name):
    profiles_folder = result_file.replace(".csv", "_profiles")
    if not os.path.exists(profiles_folder):
        os.makedirs(profiles_folder)
    power_file = os.path.join(profiles_folder, f"{config_file}_{ems_name}_power.csv")
    log_power_hist(microgrid.power_hist[0], power_file)

    reward_file = os.path.join(profiles_folder, f"{config_file}_{ems_name}_reward.csv")
    log_reward_hist(microgrid.reward_hist, reward_file)

    attributes_file = os.path.join(
        profiles_folder, f"{config_file}_{ems_name}_attributes.csv"
    )
    log_all_attributes(microgrid.attributes_hist, attributes_file)


def get_consumption(microgrid):
    public_grid = "PublicGrid_0"
    consumers = ["Charger_0", "EnergyPlus_0", "WaterHeater_0", "Consumer_0"]

    time_step = microgrid.time_step
    time_step_h = time_step.total_seconds() / 3600

    microgrid.power_hist[0]["PublicGrid_0"]
    tot_consumption = 0
    tot_grid_consumption = 0
    power_hist = microgrid.power_hist[0]
    for asset_name in power_hist.keys():
        if asset_name in consumers:
            consumption_series = [
                -power_obj.electrical * time_step_h
                for power_obj in power_hist[asset_name]
            ]
            tot_consumption += sum(consumption_series)
        if asset_name == public_grid:
            offtake_series = [
                max(power_obj.electrical, 0) * time_step_h
                for power_obj in power_hist[asset_name]
            ]
            injection_series = [
                -min(power_obj.electrical, 0) * time_step_h
                for power_obj in power_hist[asset_name]
            ]
            tot_grid_injection = sum(injection_series)
            tot_grid_offtake = sum(offtake_series)
    consumption = {
        "consumption_only": tot_consumption,
        "grid_offtake": tot_grid_offtake,
        "grid_injection": tot_grid_injection,
    }
    return consumption


def rule_base_execution(config_path, ems_name):
    delta_t_h = float(ems_name.split("_")[-1][:-1])
    delta_t_comfort = datetime.timedelta(minutes=int(delta_t_h * 60))
    microgrid = execute_rule_base(config_path, delta_t_comfort)
    return microgrid


def mpc_execution(config_path, ems_name, old_config_path):
    if "perfect" in ems_name:
        energyplus_calibration, ev_forecasting_values = get_energyplus_calibration(
            config_path
        )
        mode = "perfect"
    else:
        energyplus_calibration, ev_forecasting_values = get_energyplus_calibration(
            old_config_path
        )
        mode = "realistic"

    if "no_enforcement" in ems_name:
        disable_enforced = True
    else:
        disable_enforced = False
    microgrid = execute_mpc(
        config_path,
        cached_calibration=energyplus_calibration,
        cached_ev_forecasting_values=ev_forecasting_values,
        mode=mode,
        disable_enforced=disable_enforced,
    )
    return microgrid


def treec_execution(config_path, ems_name):
    config_file = os.path.basename(config_path)
    split_path = config_path.split("/")
    house_str = config_file.replace(".json", "")
    training_folders = split_path[-2]
    tot_houses = training_folders.split("_")[2]
    training_folders = f"treec_train_{tot_houses}/"
    best_trees = find_best_tree_for_each_house(training_folders)
    if house_str not in best_trees.keys():
        print(f"Executing rule base for {config_file}")
        microgrid = rule_base_execution(config_path, "RBC_3h")
    else:
        model_filepath = best_trees[house_str]["model_path"]
        model_folder = os.path.dirname(model_filepath) + "/"
        if "no_enforcement" in ems_name:
            disable_enforced = True
        else:
            disable_enforced = False
        microgrid = execute_treec(
            config_path, model_folder, disable_enforced=disable_enforced
        )
    return microgrid


def extract_info_from_config(config_path):
    info = dict()

    json_config = json.load(open(config_path))
    assets = json_config["Assets"]
    info["pv ?"] = "no pv"
    info["home charger ?"] = "no home charger"
    info["insulation"] = "no central heating"
    info["surface area"] = "no central heating"
    info["construction type"] = "no central heating"
    info["water heater volume"] = "no water heater"
    info["good assets"] = 0
    for asset_id, asset in assets.items():
        if "SolarPv_0" == asset["name"]:
            info["good assets"] += 1
            info["pv ?"] = "pv"
        elif "Charger_0" == asset["name"]:
            info["good assets"] += 1
            info["home charger ?"] = "home charger"
        elif "EnergyPlus_0" == asset["name"]:
            info["good assets"] += 1
            idf_model = asset["idf_model"]
            idf_file = idf_model.split("/")[-1]
            info["construction type"] = idf_file.split("_")[0]
            info["surface area"] = idf_file.split("_")[1]
            info["insulation"] = idf_file.split("_")[2].replace(".idf", "")
        elif "WaterHeater_0" == asset["name"]:
            info["good assets"] += 1
            volume = asset["volume"]
            info["water heater volume"] = str(volume)

    return info


def generate_results(house_num=100, ems_names=None, refresh=False, num_process=None):

    folder = f"data/houses_belgium_{house_num}/"
    result_file = f"results/belgium_usefull_{house_num}.csv"

    single_threaded = num_process == 1

    for ems_name in ems_names:
        tmp_config_dir = get_results(
            folder,
            result_file,
            ems_name,
            refresh=refresh,
            single_threaded=single_threaded,
            num_process=num_process,
        )
        # Remove FileLock
        os.remove(result_file + ".lock")
        if tmp_config_dir is not None:
            shutil.rmtree(tmp_config_dir)


def generate_charge_completion_results_treec():
    tot_houses = 500
    folder = f"data/houses_belgium_{tot_houses}/"
    result_file = f"results/belgium_usefull_{tot_houses}_no_enforcement.csv"
    ems_name = "TreeC_no_enforcement"
    get_results(
        folder,
        result_file,
        ems_name,
        refresh=True,
        single_threaded=False,
        num_process=10,
    )


def generate_charge_completion_results_mpc_hpc(house_num):
    tot_houses = 500
    folder = f"data/houses_belgium_{tot_houses}/"
    result_file = f"results/belgium_usefull_{tot_houses}_no_enforcement.csv"
    ems_name = "MPC_realistic_forecast_no_enforcement"
    get_results(
        folder,
        result_file,
        ems_name,
        refresh=True,
        single_threaded=True,
        config_list=[f"house_{house_num}.json"],
    )

if __name__ == "__main__":

    # generate_results(house_num=500, ems_names=["MPC_realistic"], num_process=6)

    # generate_results(house_num=500, ems_names=["MPC_perfect"], num_process=10)
    # config_path = "data/houses_belgium_500/house_0.json"
    # ems_name = "RBC_charge7.4_1.5h"

    # microgrid = rule_base_execution(config_path, ems_name)

    # plot_simulation(microgrid=microgrid)
    # plt.show()

    # generate_results(
    #    house_num=500, ems_names=["RBC_charge7.4_1.5h"], refresh=False, num_process=6
    # )
    # generate_charge_completion_results_treec()
    # generate_charge_completion_results_mpc_hpc(25)
    # generate_results(house_num=500, ems_names=["RBC_1.5h"], num_process=10)

    # info = extract_info_from_config("data/houses_belgium_100/house_27.json")
    # print(info)

    house_num = 25
    tmp_config_path = f"data/houses_belgium_500_tmp_2023/house_{house_num}.json"
    result_file = "test_"
    microgrid = treec_execution(tmp_config_path, "TreeC")
    print(microgrid.tot_reward.KPIs)
