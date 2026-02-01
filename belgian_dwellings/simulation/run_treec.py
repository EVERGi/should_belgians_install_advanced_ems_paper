from simugrid.simulation.config_parser import parse_config_file
from simugrid.misc.log_plot_micro import plot_simulation

import os
from belgian_dwellings.simulation.custom_classes import DayAheadEngie, TreeManager
from belgian_dwellings.simulation.tmp_2023_config import tmp_2023_config
from belgian_dwellings.training.input_functions import first_input_function

from treec.logger import TreeLogger

from simugrid.assets.energyplus import EnergyPlus
import os
import datetime
import time

# import fabric
import json
import numpy as np
import matplotlib.pyplot as plt


def find_best_tree_for_each_house(log_folder):
    best_trees = {}
    for house_folder in sorted(os.listdir(log_folder)):
        house_folder_path = os.path.join(log_folder, house_folder)
        for training_folder in os.listdir(house_folder_path):
            validation_folder = os.path.join(
                house_folder_path, training_folder, "validation/"
            )
            if os.path.isdir(validation_folder):
                eval_score_file = os.path.join(
                    validation_folder, f"{house_folder}_tree_0/episode_score_file.csv"
                )
                # Format of the file is:
                # episode,1
                # score,-2232.617251823832
                if os.path.isfile(eval_score_file):
                    with open(eval_score_file, "r") as file:
                        lines = file.readlines()
                        score = float(lines[1].split(",")[1])
                        model_folder = (
                            validation_folder + f"{house_folder}_tree_0/models/"
                        )

                        # Round score the 1 decimal
                        score_str = str(round(score, 1))
                        model_filename = f"model_0_{score_str}.json"
                        model_path = model_folder + model_filename
                        if house_folder not in best_trees:

                            best_trees[house_folder] = {
                                "model_path": model_path,
                                "score": score,
                            }
                        elif score > best_trees[house_folder]["score"]:
                            best_trees[house_folder] = {
                                "model_path": model_path,
                                "score": score,
                            }
    return best_trees


def execute_treec(config_file, model_folder, disable_enforced=False):

    microgrid = parse_config_file(config_file)

    reward = DayAheadEngie()
    microgrid.set_reward(reward)

    trees = TreeLogger.get_best_trees(model_folder)
    ems = TreeManager(microgrid, trees, first_input_function)

    if disable_enforced:
        ems.do_forced_charging = False

    end_time = microgrid.end_time
    # end_time = microgrid.start_time + datetime.timedelta(days=1)

    microgrid.attribute_to_log("EnergyPlus_0", "zn0_temp")
    microgrid.attribute_to_log("EnergyPlus_0", "cur_t_low")
    microgrid.attribute_to_log("EnergyPlus_0", "cur_t_up")
    microgrid.attribute_to_log("Charger_0", "soc")
    microgrid.attribute_to_log("WaterHeater_0", "t_tank")
    while microgrid.utc_datetime < end_time:
        # Print every new week
        if (
            microgrid.utc_datetime.weekday() == 1
            and microgrid.utc_datetime.hour == 0
            and microgrid.utc_datetime.minute == 0
        ):
            pass
            # print(microgrid.utc_datetime)
        microgrid.management_system.simulate_step()

    for asset in microgrid.assets:
        if isinstance(asset, EnergyPlus):
            asset.stop_energyplus_thread()
    # print(f"Final opex: {microgrid.tot_reward.KPIs['opex']}")
    # print(f"Final discomfort: {microgrid.tot_reward.KPIs['discomfort']}")
    return microgrid


def execute_run():

    house_num = 0
    tot_houses = 100
    train_num = 0

    config_file = f"data/houses_belgium_{tot_houses}/house_{house_num}.json"
    model_folder = f"treec_train_{tot_houses}/house_{house_num}/house_{house_num}_tree_{train_num}/validation/house_{house_num}_tree_0/models/"

    start_time = time.time()
    delta_t_comfort = datetime.timedelta(hours=1)
    microgrid = execute_treec(config_file, model_folder, disable_enforced=True)
    print(f"Time taken: {time.time() - start_time}")
    # plot_simulation(microgrid, show=False)
    opex = microgrid.tot_reward.KPIs["opex"]
    discomfort = microgrid.tot_reward.KPIs["discomfort"]
    print(f"Final opex: {opex}")
    print(f"Final discomfort: {discomfort}")


if __name__ == "__main__":
    tot_houses = 500
    log_folder = f"treec_train_{tot_houses}"
    best_trees = find_best_tree_for_each_house(log_folder)

    house_num = 25
    run_house = f"house_{house_num}"
    print(best_trees[run_house])
    model_path = best_trees[run_house]["model_path"]
    model_folder = os.path.dirname(model_path) + "/"
    config_file = f"data/houses_belgium_{tot_houses}/{run_house}.json"
    # config_path = os.path.join(folder, config_file)
    tmp_config_path = tmp_2023_config(config_file)
    microgrid = execute_treec(config_file, model_folder, disable_enforced=True)
    KPIs = microgrid.tot_reward.KPIs
    opex = KPIs["opex"]
    discomfort = KPIs["discomfort"]
    forced_charged_energy = KPIs["forced_charged_energy"]
    print(f"Final opex: {opex}")
    print(f"Final discomfort: {discomfort}")
    print(f"Final forced charged energy: {forced_charged_energy}")
    print(f"Final quantile dissatisfaction: {KPIs['soc_quantile']}")

    ems = microgrid.management_system
    soc_diff_list = microgrid.tot_reward.soc_diff_list

    quant_values = [i for i in np.arange(0, 1.05, 0.05)]

    y = []
    # for q in quant_values:
    #    count = np.quantile(soc_diff_list, q, method="hazen")
    #    y.append(count)

    print(soc_diff_list)

    plot_simulation(microgrid)
    plt.figure()
    plt.plot(quant_values, y)
    plt.show()
