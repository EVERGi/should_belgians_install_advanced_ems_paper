import sys


from belgian_dwellings.simulation.get_results import get_results, generate_charge_completion_results_mpc_hpc

def get_results_hpc(house_num, ems_name= "MPC_realistic_forecast"):
    tot_houses = 500
    config_list = [f"house_{house_num}.json"]
    single_threaded = True

    folder = f"data/houses_belgium_{tot_houses}/"
    result_file = f"results/belgium_usefull_{tot_houses}.csv"
    get_results(
        folder,
        result_file,
        ems_name,
        refresh=True,
        single_threaded=single_threaded,
        config_list=config_list,
    )

if __name__ == "__main__":
    house_num = int(sys.argv[1])
    if len(sys.argv) == 2:
        get_results_hpc(house_num)
        generate_charge_completion_results_mpc_hpc(house_num)
    else:
        mode = sys.argv[2]
        if mode == "perfect":
            ems_name = "MPC_perfect"
            get_results_hpc(house_num, ems_name)
        elif mode == "realistic":
            ems_name = "MPC_realistic_forecast"
            get_results_hpc(house_num, ems_name)
        elif mode == "no_enforcing":
            generate_charge_completion_results_mpc_hpc(house_num)

    print(f"Results for house {house_num} have been generated.")
    