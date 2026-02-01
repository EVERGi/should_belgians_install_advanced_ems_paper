import os, shutil

from belgian_dwellings.simulation.custom_classes import DayAheadEngie, TreeManager, HouseManager

from simugrid.misc.log_plot_micro import plot_hist, plot_attributes

from copy import deepcopy
import matplotlib.pyplot as plt

from belgian_dwellings.training.input_functions import first_input_function

from simugrid.simulation.config_parser import parse_config_file
from simugrid.assets.energyplus import EnergyPlus

from treec.logger import TreeLogger
from treec.train import tree_train, tree_validate

import gc


def set_action_infos(ems):
    action_infos = [dict() for _ in range(ems.num_setpoints)]
    count = 0
    for asset, setpoints in ems.control_points.items():
        for setpoint_name in setpoints.keys():
            cur_action = action_infos[count]
            if setpoint_name == "power_sp":
                min_power = -asset.max_consumption_power
                max_power = asset.max_production_power

                name = "Power setpoint (kW): "
                cur_action[name] = dict()

                bounds = [min_power, max_power]
                cur_action[name]["bounds"] = bounds

            elif setpoint_name.startswith("zn0"):
                if "heating" in setpoint_name:

                    names = [
                        "Heating setpoint (°C): ",
                        "Temperature above current (°C): ",
                        "Match highest low comfort temperature for next (h): ",
                    ]
                elif "cooling" in setpoint_name:
                    names = [
                        "Cooling setpoint (°C): ",
                        "Temperature below current (°C): ",
                        "Match lowest high comfort temperature for next (h): ",
                    ]
                bounds = [[ems.t_low, ems.t_high], [-2, 5], [0, 12]]
                for i, name in enumerate(names):
                    cur_action[name] = dict()
                    cur_action[name]["bounds"] = bounds[i]

            count += 1
    return action_infos


def evaluate_microgrid_trees(trees, params_evaluation):
    microgrid = params_evaluation["microgrid"]
    input_func = params_evaluation["input_func"]
    ManagerClass = params_evaluation["ManagerClass"]
    render = params_evaluation["render"]

    new_microgrid = deepcopy(microgrid)

    for asset in new_microgrid.assets:
        if isinstance(asset, EnergyPlus):
            asset.start_energyplus_thread()

    reward = DayAheadEngie()
    new_microgrid.set_reward(reward)

    ems = ManagerClass(new_microgrid, trees, input_func)
    ems.do_forced_charging = False

    try:
        while new_microgrid.datetime < new_microgrid.end_time:
            new_microgrid.management_system.simulate_step()
    except RuntimeError as e:
        HIGH_PRICE = 50_000
        all_nodes_visited = ems.all_nodes_visited
        return -HIGH_PRICE, all_nodes_visited

    opex = new_microgrid.tot_reward.KPIs["opex"]
    discomfort = new_microgrid.tot_reward.KPIs["discomfort"]

    # 5 €/Kh until 30 Kh discomfort to get similar discomfort as MPC with perfect forecast
    if discomfort > 30:
        weighted_comfort = (discomfort - 30) * 5
    else:
        weighted_comfort = 0

    # forced_charged_energy = new_microgrid.tot_reward.KPIs["forced_charged_energy"]

    # 0.2 €/kWh extra for forced simulation charging
    # weighted_forced_charging = forced_charged_energy * 0.2
    weighted_forced_charging = 0
    PRICE_BELOW_5_PERCENT = 0.5
    PRICE_ABOVE_5_PERCENT = 2.0
    FIVE_PERCENT = 0.05
    for i, soc_diff in enumerate(reward.soc_diff_list):
        energy_diff = reward.energy_diff_list[i]
        if soc_diff <= FIVE_PERCENT:
            weighted_forced_charging += energy_diff * PRICE_BELOW_5_PERCENT
        else:
            energy_five_percent = energy_diff * FIVE_PERCENT / soc_diff
            energy_above = energy_diff - energy_five_percent
            weighted_forced_charging += (
                energy_five_percent * PRICE_BELOW_5_PERCENT
                + energy_above * PRICE_ABOVE_5_PERCENT
            )

    result = opex - weighted_comfort - weighted_forced_charging

    # result = new_microgrid.tot_reward.KPIs["opex"]

    if render:
        power_hist = new_microgrid.power_hist
        kpi_hist = new_microgrid.reward_hist
        attributes_hist = new_microgrid.attributes_hist
        plot_attributes(attributes_hist)
        plot_hist(power_hist, kpi_hist)
        plt.show()

    all_nodes_visited = ems.all_nodes_visited

    for asset in new_microgrid.assets:
        if isinstance(asset, EnergyPlus):
            asset.stop_energyplus_thread()

    # Ensure garbage collection
    new_microgrid = None
    ems = None
    reward = None
    asset = None
    gc.collect()

    return result, all_nodes_visited


def params_house(
    log_folder, house_num, tot_houses, create_logger=True, params_change={}
):

    common_params = {
        "input_func": first_input_function,
        "eval_func": evaluate_microgrid_trees,
        "ManagerClass": TreeManager,
        "house_num": house_num,
        "render": False,
        "tot_steps": 365 * 24 * 4,
        "pruning_tol": 5,
    }

    num_gen = 1
    algo_type = "tree"
    algo_params = {
        "gen": num_gen,
        "fixed": True,
        "pop_size": 6,
        "single_threaded": False,
        "pygmo_algo": "pso_gen",
        "pool_size": None,
    }

    for key, value in params_change.items():
        if key in common_params.keys():
            common_params[key] = value
        elif key in algo_params.keys():
            algo_params[key] = value

    microgrid = setup_microgrid(house_num, tot_houses, common_params, algo_params)
    logger = setup_logger(
        algo_type, common_params, algo_params, log_folder, create_logger
    )

    common_params["microgrid"] = microgrid
    common_params["logger"] = logger

    return common_params, algo_params


def setup_microgrid(house_num, tot_houses, common_params, algo_params):
    fildir = f"data/houses_belgium_{tot_houses}/"

    config_file = f"{fildir}house_{house_num}.json"

    common_params["config_file"] = config_file
    microgrid = parse_config_file(config_file)

    ems = HouseManager(microgrid)

    action_infos = set_action_infos(ems)
    common_params["action_infos"] = action_infos

    tree_nodes = 20

    dimension = ems.num_setpoints * (3 * tree_nodes + 1)
    algo_params["dimension"] = dimension

    for asset in microgrid.assets:
        if isinstance(asset, EnergyPlus):
            asset.stop_energyplus_thread()
            asset.pickle_deepcopy = True

    return microgrid


def setup_logger(algo_type, common_params, algo_params, log_folder, create_logger):
    if create_logger:
        logger = TreeLogger(log_folder, algo_type, common_params, algo_params)
    else:
        logger = None

    return logger


def train_house(house_num, tot_houses, params_change={}):
    log_folder = f"treec_train_{tot_houses}/house_{house_num}"

    common_params, algo_params = params_house(
        log_folder, house_num, tot_houses, params_change=params_change
    )
    logger = common_params["logger"]

    tree_train(common_params, algo_params)

    return logger.folder_name


def valid_house(
    training_folder,
    house_num,
    tot_houses,
    render=False,
    create_logger=True,
    params_change={},
):
    treec_train_100 = training_folder + "/"
    validate_folder = training_folder + "/validation/"
    valid_params, _ = params_house(
        validate_folder,
        house_num,
        tot_houses,
        create_logger=create_logger,
        params_change=params_change,
    )
    valid_params["render"] = render

    prune_params = valid_params
    tree_validate(valid_params, training_folder, prune_params)


def train_and_valid_house(
    house_num,
    tot_houses,
    params_change={},
):
    log_folder = train_house(house_num, tot_houses, params_change)
    print("Training completed successfully")

    valid_house(log_folder, house_num, tot_houses, params_change=params_change)

    TreeLogger.clean_logs(log_folder)
    return log_folder


def print_action_infos():
    config_file = "data/houses_belgium_100/house_0.json"
    microgrid = parse_config_file(config_file)

    ems = HouseManager(microgrid)

    action_infos = set_action_infos(ems)
    print(action_infos)

    params = {"microgrid": microgrid}
    output = first_input_function(params)
    print(output)


def valid_all_trainings(training_folder):
    for house_num in range(100):
        house_folder = f"{training_folder}/house_{house_num}/"
        if os.path.exists(house_folder):
            for folder in os.listdir(house_folder):
                indiv_folder = house_folder + folder + "/"
                valid_folder = indiv_folder + "validation/"
                if not os.path.exists(valid_folder):
                    print(f"Validating {indiv_folder}")
                    try:
                        valid_house(indiv_folder, house_num, 100)
                    except Exception as e:
                        print(f"Error validating {indiv_folder}: {e}")


def redo_failed_pruning(tot_houses):
    treec_train_folder = f"treec_train_{tot_houses}/"
    failed_folders = find_failed(treec_train_folder)

    for folder in failed_folders:
        folder = folder + "/"
        validation_folder = os.path.join(folder, "validation")
        shutil.rmtree(validation_folder)
        house_num = int(folder.split("/")[1].split("_")[1])
        print(f"Revalidating house {house_num} in folder {folder}")

        valid_house(folder, house_num, tot_houses)


def find_failed(treec_train_folder):

    failed_folders = []
    for folder in sorted(os.listdir(treec_train_folder)):
        for subfolder in os.listdir(os.path.join(treec_train_folder, folder)):
            sub_folder_dir = os.path.join(treec_train_folder, folder, subfolder)
            episod_score_file = os.path.join(sub_folder_dir, "episode_score_file.csv")

            with open(episod_score_file, "r") as file:
                lines = file.readlines()
            try:
                score = float(lines[-1].split(",")[-1])
            except ValueError:
                continue
            except Exception as e:
                print(e)
                print(folder, subfolder)
            validation_folder = os.path.join(sub_folder_dir, "validation")

            if not os.path.exists(validation_folder):
                continue
            for subvalidation_folder in os.listdir(validation_folder):

                validation_episode_file = os.path.join(
                    validation_folder,
                    subvalidation_folder,
                    "episode_score_file.csv",
                )

                if os.path.exists(validation_episode_file):
                    with open(validation_episode_file, "r") as file:
                        lines = file.readlines()

                    validation_score = float(lines[-1].split(",")[-1])
                    if validation_score < score - 5:
                        print("Strange result in folder:", folder, subfolder)
                        print(f"Score: {score}, Validation score: {validation_score}")
                        failed_folders.append(sub_folder_dir)
    return failed_folders


if __name__ == "__main__":
    # valid_all_trainings("treec_train_100")

    # params_change = {"single_threaded": False, "gen": 2, "pop_size": 20, "pool_size": 3}

    tot_houses = 500

    # house_num = 452
    # train_house(house_num, tot_houses, params_change=params_change)
    # train_and_valid_house(house_num, tot_houses, params_change)
    # log_folder = f"treec_train_{tot_houses}/house_{house_num}/house_{house_num}_tree_0/"
    # valid_house(log_folder, house_num, tot_houses)

    # print_action_names()
    redo_failed_pruning(tot_houses)
