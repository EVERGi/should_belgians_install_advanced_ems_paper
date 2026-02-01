import argparse


from belgian_dwellings.plots.result_plots import plot_all_paper_plots

from belgian_dwellings.simulation.hpc_get_results_mpc import get_results_hpc, generate_charge_completion_results_mpc_hpc
from belgian_dwellings.training.hpc_run import hpc_run

from belgian_dwellings.simulation.get_results import generate_results, generate_charge_completion_results_treec

def plot_results():
    """Plot the results of the simulations."""
    print("Plotting results...")
    plot_all_paper_plots()
    # Add your plotting logic here

def reproduce_mpc_results():
    modes = ["perfect", "realistic", "no_enforcing"]

    for mode in modes:
        for house_num in range(500):    
            if mode == "perfect":
                ems_name = "MPC_perfect"
                get_results_hpc(house_num, ems_name)
            elif mode == "realistic":
                ems_name = "MPC_realistic_forecast"
                get_results_hpc(house_num, ems_name)
            elif mode == "no_enforcing":
                generate_charge_completion_results_mpc_hpc(house_num)


def train_treec_models():
    for i in range(500):
        hpc_run(i, 500)

def run_all_results():
    generate_results(house_num=500, ems_names=["TreeC", "RBC_1.5h"], num_process=10)
    
    generate_charge_completion_results_treec()
    
    reproduce_mpc_results()


    def main():
        parser = argparse.ArgumentParser(
            description="Belgian Dwellings EMS Command Line Tool"
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        subparsers.add_parser("plot", help="Plot the results of the simulations")
        subparsers.add_parser("train", help="Train TreeC models")
        subparsers.add_parser("run", help="Run all simulations and generate results")
        
        args = parser.parse_args()
        
        if args.command == "plot":
            plot_results()
        elif args.command == "train":
            train_treec_models()
        elif args.command == "run":
            run_all_results()
        else:
            parser.print_help()
    generate_results(house_num=500, ems_names=["TreeC", "RBC_1.5h"], num_process=10)
    
    generate_charge_completion_results_treec()
    
    reproduce_mpc_results()


def main():
    parser = argparse.ArgumentParser(
        description="Belgian Dwellings EMS Command Line Tool"
    )
    

if __name__ == "__main__":
    main()