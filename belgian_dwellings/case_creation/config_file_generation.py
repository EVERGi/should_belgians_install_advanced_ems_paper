from scenario_creation import sample_houses
import copy
import os
import json
import pandas as pd
import random
import scipy.stats
import statistics

from belgian_dwellings.case_creation.occupency_profiles import create_week_occupancy_profiles

from belgian_dwellings.utils.paths import OPENSTUDIOAPPLICATION_PATH

config_file_template = {
    "Microgrid": {
        "number_of_nodes": 1,
        "timezone": "UTC",
        "start_time": "2022/01/01 00:00:00",
        "end_time": "2023/01/01 00:00:00",
        "time_step": "00:15:00",
    },
    "Assets": {
        "Asset_0": {
            "node_number": 0,
            "name": "Consumer_0",
            "max_consumption_power": 20,
            "size": 20,
        },
        "Asset_1": {
            "node_number": 0,
            "name": "PublicGrid_0",
            "max_consumption_power": 100,
            "max_production_power": 100,
            "size": 1,
        },
        "Asset_2": {
            "node_number": 0,
            "name": "EnergyPlus_0",
            "timezone": "Europe/Brussels",
            "max_consumption_power": 30,
        },
        "Asset_3": {
            "node_number": 0,
            "name": "WaterHeater_0",
            "high_setpoint": 65,
            "low_setpoint": 55,
        },
        "Asset_4": {
            "node_number": 0,
            "name": "Charger_0",
            "max_charge_cp": 3.7,
            "max_discharge_cp": 0,
            "ID": 0,
            "eff": 0.88,
            "charge_curve": False,
        },
        "Asset_5": {
            "node_number": 0,
            "name": "SolarPv_0",
            "max_production_power": 10,
            "size": 10,
        },
        "Asset_6": {
            "node_number": 0,
            "name": "Battery_0",
            "soc": 1,
            "soc_min": 0,
            "soc_max": 1,
            "charge_eff": 0.95,
            "disch_eff": 0.95,
            "max_charge": 0.625,
            "max_discharge": 0.625,
            "size": 5.12,
        },
    },
    "Environments": {
        "Environment_0": {
            "nodes_number": "0",
        }
    },
}


def generate_config_dicts(houses):
    week_occupancy = create_week_occupancy_profiles(len(houses))

    config_files = []
    for i, house in enumerate(houses):
        config_file = copy.deepcopy(config_file_template)
        print(house)
        if house["pv ?"] == "no pv":
            config_file["Assets"].pop("Asset_5")
        BATTERY_IN_HOUSE = False
        if not BATTERY_IN_HOUSE:
            config_file["Assets"].pop("Asset_6")
        if house["home charger ?"] == "no home charger":
            config_file["Assets"].pop("Asset_4")
        if house["heating ?"] == "no central heating":
            config_file["Assets"].pop("Asset_2")
            config_file["Assets"].pop("Asset_3")

        config_files.append(config_file)
        environment = config_file["Environments"]["Environment_0"]
        environment["region"] = house["region"]
        # Select random consumption profile
        environment["Consumer_0_electric"] = select_consumption_profile()

        if house["pv ?"] == "pv":
            environment["SolarPv_0"] = select_pv_profile(house)

        # print(config_file)
        if house["heating ?"] == "central heating":
            eplus_asset = config_file["Assets"]["Asset_2"]
            eplus_asset["idf_model"] = select_energyplus_model(house)
            eplus_asset["epw_weather"] = select_weather_file()
            # eplus_asset["eplus_dir"] = f"{OPENSTUDIOAPPLICATION_PATH}EnergyPlus/"
            eplus_asset["eplus_dir"] = (
                "OpenStudioApplication/usr/local/openstudioapplication/EnergyPlus/"
            )
            eplus_asset["occupancy"] = week_occupancy[i]

            water_heater = config_file["Assets"]["Asset_3"]
            water_heater_params, water_flow_profile = select_water_heater_params(house)
            water_heater.update(water_heater_params)
            environment["WaterHeater_0_flow"] = water_flow_profile

        if house["home charger ?"] == "home charger":
            ev_profile = select_ev_profile()
            environment[ev_profile] = None
        for j, asset_name in enumerate(list(config_file["Assets"].keys())):
            config_file["Assets"][f"Asset_{j}"] = config_file["Assets"].pop(asset_name)

        dynamic_prices = calculate_environment_price()
        environment.update(dynamic_prices)
    return config_files


def select_ev_profile():
    ev_dir = "data/common/charging_sessions/"
    ev_files = os.listdir(ev_dir)
    ev_file = random.choice(ev_files)
    ev_profile_path = f"./../common/charging_sessions/{ev_file}"

    return ev_profile_path


def select_pv_profile(house):
    region = house["region"]
    pv_dir = "data/common/pv_profiles/"
    pv_files = os.listdir(pv_dir)

    pv_info_path = "data/pv_data/pv_systems_belgium.csv"
    pv_info_df = pd.read_csv(pv_info_path)

    dict_pvs = {"Flanders": [], "Wallonia": [], "Brussels": []}
    dict_regions_names = {
        "Vlaanderen": "Flanders",
        "Wallonie": "Wallonia",
        "Bruxelles-Capitale": "Brussels",
    }
    for pv_file in pv_files:
        pv_id = int(pv_file.split("_")[0])
        pv_info = pv_info_df[pv_info_df["system_id"] == pv_id]

        dates = pv_file.replace(".csv", "").replace(f"{pv_id}_", "")
        region_pv_id = dict_regions_names[pv_info["region"].values[0]]
        health = pv_info[f"health_{dates}"].values[0]
        if health >= 0.95:
            relative_dir = pv_dir.replace("data/", "./../")
            dict_pvs[region_pv_id].append(f"{relative_dir}{pv_file}")

    pv_profile_path = random.choice(dict_pvs[region])

    return pv_profile_path


def select_consumption_profile():
    consumption_dir = "data/common/consumption/"
    consumption_files = os.listdir(consumption_dir)

    consumption_file = random.choice(consumption_files)
    comsumption_path = f"./../common/consumption/{consumption_file}"

    return comsumption_path


def select_energyplus_model(house):
    eplus_dir = "./../common/energyplus_models/"
    contruction_type = house["construction type"]
    surface_area = house["surface area"]
    insulation = house["insulation"]

    eplus_file = f"{contruction_type}_{surface_area}_{insulation}.idf"

    eplus_path = f"{eplus_dir}{eplus_file}"

    return eplus_path


def select_weather_file():
    eplus_dir = "./../common/energyplus_models/"
    weather_file = "brussels_2022.epw"
    weather_path = f"{eplus_dir}{weather_file}"
    return weather_path


def select_water_heater_params(house):
    water_heater_params = {
        "t_in": "Belgium",
    }
    bedrooms = house["bedrooms"]
    bedrooms_int = int(bedrooms.replace(" bedroom", ""))

    # Parameters obtained from heuristic opt in data/water_data/calc_distribution.py
    # Mean: 57.89810091786419
    # Var: 4284.162553277767
    # Skew: 5.446389586122736
    # Kurt: 66.63829380874256
    # Median: 38.65879590618502
    scale = 67.68252466
    loc = 0.0

    # Mean hot water consumption in L/day
    # Source: https://doi.org/10.1016/j.rser.2021.112035

    mean_hot_water_cons = {0: 42, 1: 42, 2: 57, 3: 81, 4: 110, 5: 121, 6: 121}

    mean_house_water_cons = mean_hot_water_cons[bedrooms_int]

    volume_tank = 3 * mean_house_water_cons / 1000  # in m3
    # Example 200L tank has a power of 2kW
    power_tank = volume_tank * 10  # in kW

    water_heater_params["volume"] = volume_tank
    water_heater_params["max_consumption_power"] = power_tank

    water_consumption = scipy.stats.invgauss(
        mu=(mean_house_water_cons - loc) / scale, loc=loc, scale=scale
    ).rvs(size=1)[0]

    df_water_profiles = pd.read_csv("data/water_data/analysis.csv")

    # Keep rows with a "2022 (L/day)" or "2023 (L/day)" value higher than 0.01
    df_water_profiles = df_water_profiles[
        (df_water_profiles["2022 (L/day)"] > 0.01)
        & (df_water_profiles["2023 (L/day)"] > 0.01)
    ]

    # Keep rows with a "day max (L)" value lower than 10000
    df_water_profiles = df_water_profiles[df_water_profiles["day max (L)"] < 10_000]

    # Find row with the average over 2022 and 2023 consumption closest to the generated consumption
    df_water_profiles["average"] = (
        df_water_profiles["2022 (L/day)"] + df_water_profiles["2023 (L/day)"]
    ) / 2
    df_water_profiles["diff"] = abs(df_water_profiles["average"] - water_consumption)
    user_key = df_water_profiles.loc[df_water_profiles["diff"].idxmin()]["user.key"]

    water_flow_profile = f"./../common/hot_water/{user_key}.csv"

    return water_heater_params, water_flow_profile


def calculate_environment_price():
    # Source: data/day_ahead_price/E_DYNAMIC_R_GREY_C_I_12_V_F_202301.pdf
    distribution_kwh_costs = [
        0.0374193,
        0.0394475,
        0.0381831,
        0.0498204,
        0.0401029,
        0.0365582,
        0.0421339,
        0.0388562,
        0.036458,
        0.0457544,
    ]
    dis_kwh_cost = statistics.mean(distribution_kwh_costs)

    distribution_capacity_costs = [
        40.0309,
        37.6491,
        41.0824,
        48.7615,
        43.5071,
        39.0604,
        45.0292,
        43.6742,
        53.0580,
        48.1157,
    ]
    month_capacity_costs = statistics.mean(distribution_capacity_costs) / 12
    kwh_offtake_cost = 0.02184 + dis_kwh_cost + 0.0020417 + 0.0144160
    year_cost = 100.7 + 14.53

    dynamic_prices = {
        "day_ahead_price": "./../common/day_ahead_price/day_ahead_22_23.csv",
        "kwh_offtake_cost": kwh_offtake_cost,
        "offtake_extra": 0.00204,
        "injection_extra": 0,
        "capacity_tariff": month_capacity_costs,
        "year_cost": year_cost,
    }
    return dynamic_prices


def log_config_files(config_files, log_folder):
    # Remove the old files
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    else:
        for file in os.listdir(log_folder):
            os.remove(f"{log_folder}/{file}")

    for i, config_file in enumerate(config_files):
        with open(f"{log_folder}/house_{i}.json", "w") as f:
            json.dump(config_file, f, indent=4)


def generate_config_files(num_houses, log_folder):
    houses = sample_houses(num_houses)
    config_files = generate_config_dicts(houses)
    log_config_files(config_files, log_folder)


if __name__ == "__main__":
    for num_houses in [500]:

        log_folder = f"data/houses_belgium_{num_houses}/"

        generate_config_files(num_houses, log_folder)
