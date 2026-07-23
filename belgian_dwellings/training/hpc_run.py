import os, sys
from belgian_dwellings.training.train_treec import train_house, valid_house
from belgian_dwellings.utils.progress import log
import random
import time
from treec.logger import TreeLogger

def get_max_training(tot_houses):
    train_folder = f"treec_train_{tot_houses}/"
    max_train = 0
    for house_folder in os.listdir(train_folder):
        all_tree_train = [f for f in os.listdir(train_folder+house_folder)]
        num_train = len(all_tree_train)
        if num_train > max_train:
            max_train = num_train

    return max_train

def hpc_run(
    house_num,
    tot_houses,
    gen=300,
    pop_size=200,
    max_train=5,
):
    # Check if the house is already trained
    house_folder = f"treec_train_{tot_houses}/house_{house_num}/"
    if os.path.exists(house_folder):
        all_tree_train = [f for f in os.listdir(house_folder)]
        if len(all_tree_train) >= max_train:
            log(f"house_{house_num}: already has {len(all_tree_train)} trees, skipped")
            return

    params_change = {
        "single_threaded": False,
        "gen": gen,
        "pop_size": pop_size,
    }

    log_folder = train_house(house_num, tot_houses, params_change)
    log(f"house_{house_num}: trained -> {log_folder}")

    valid_house(log_folder, house_num, tot_houses)
    
    TreeLogger.clean_logs(log_folder)
    open("out/eplusout_perflog.csv", 'w').close()

    return log_folder


if __name__ == "__main__":
    tot_houses = 500
    house_num = int(sys.argv[1])
    rand_start = 20 * random.random()
    time.sleep(rand_start)
    hpc_run(house_num, tot_houses)
