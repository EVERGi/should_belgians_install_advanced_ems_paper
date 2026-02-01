from belgian_dwellings.simulation.run_treec import execute_tree
from belgian_dwellings.simulation.rbc_ems import execute_rule_base
from belgian_dwellings.simulation.run_mpc import execute_mpc
import datetime

if __name__ == "__main__":
    house_num = 2
    tot_houses = 100
    train_num = 6

    config_file = f"data/houses_belgium_{tot_houses}/house_{house_num}.json"
    model_folder = f"treec_train_{tot_houses}/house_{house_num}/house_{house_num}_tree_{train_num}/validation/house_{house_num}_tree_0/models/"

    # Execute the three EMS
    microgrid_tree = execute_tree(config_file, model_folder)
    print("Tree KPIs:")
    print(
        f"opex: {microgrid_tree.tot_reward.KPIs['opex']}, discomfort: {microgrid_tree.tot_reward.KPIs['discomfort']}"
    )
    microgrid_rbc = execute_rule_base(
        config_file, delta_t_comfort=datetime.timedelta(hours=3)
    )
    print("RBC KPIs:")
    print(
        f"opex: {microgrid_rbc.tot_reward.KPIs['opex']}, discomfort: {microgrid_rbc.tot_reward.KPIs['discomfort']}"
    )
    microgrid_mpc = execute_mpc(config_file)

    print("MPC KPIs:")
    print(
        f"opex: {microgrid_mpc.tot_reward.KPIs['opex']}, discomfort: {microgrid_mpc.tot_reward.KPIs['discomfort']}"
    )
