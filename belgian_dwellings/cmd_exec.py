"""Command line entry point to reproduce the paper's results.

Three independent stages, matching how the paper was actually produced:

1. ``plots``        - regenerate every figure in the paper from an existing results folder
                       (either the one downloaded from Zenodo, or one produced by ``run`` below).
2. ``train``         - reproduce the trained models:
                         - ``treec``        trains (and validates) the TreeC decision trees per house,
                                             using the same evolutionary-optimization parameters as the
                                             paper (300 generations, population of 200, 5 trees/house).
                         - ``mpc-forecast``  the MPC's thermal calibration and EV arrival/departure
                                             forecaster are not saved models: they are (re-)fit from an
                                             EnergyPlus year-long run for each house, immediately before
                                             that house's MPC simulation (see
                                             ``belgian_dwellings.simulation.calibrate_mpc``). This
                                             subcommand exercises that exact fitting step and writes the
                                             resulting per-house calibration parameters to
                                             ``<results-dir>/mpc_calibration_<tot-houses>.csv`` so the fit
                                             can be inspected/reproduced independently of a full MPC run.
                                             ``run`` below performs this same fit again on demand (as the
                                             paper's pipeline does) - nothing is cached/re-used from here.
3. ``run``           - execute the EMS simulations (RBC, TreeC, MPC perfect/realistic, and the
                        "no enforcement" variants used for the SOC/charge-completion plots) across a set
                        of houses, writing the resulting KPI/profile data into a results folder of your
                        choice so you can compare against the original results without overwriting them.

All three stages are thin wrappers around the existing simulation/training/plotting code in
``belgian_dwellings`` - see that code for what each EMS actually does. For the full 500-house study,
training and MPC result generation are compute-heavy (EnergyPlus + Gurobi + evolutionary search per
house); the paper itself ran those at that scale via the SLURM array scripts under
``belgian_dwellings/training/hpc_train.sh`` and ``belgian_dwellings/simulation/hpc_get_results_mpc*.sh``.
This CLI is meant for reproducing on a subset of houses, or the full set if you have the time/hardware.

Examples
--------
    python -m belgian_dwellings.cmd_exec plots

    python -m belgian_dwellings.cmd_exec train treec --houses 0-9 --workers 4
    python -m belgian_dwellings.cmd_exec train mpc-forecast --houses 0-9

    python -m belgian_dwellings.cmd_exec run --houses 0-9 --results-dir results_rerun
    python -m belgian_dwellings.cmd_exec run --houses 0-9 --target no-enforcement --workers 4
"""

import argparse
import functools
import multiprocessing
import os

from belgian_dwellings.utils.progress import (
    log,
    progress_bar,
    set_progress_enabled,
)


def parse_house_spec(spec, tot_houses):
    """Parse a house selection string like "0-9,25,40-45" into a sorted list of indices.

    ``None`` (the default) means every house in the ``tot_houses``-sized dataset.
    """
    if spec is None:
        return list(range(tot_houses))

    houses = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-")
            houses.update(range(int(start), int(end) + 1))
        else:
            houses.add(int(part))
    return sorted(h for h in houses if 0 <= h < tot_houses)


def warn_if_missing(paths):
    missing = [path for path in paths if not os.path.exists(path)]
    for path in missing:
        log(f"Warning: '{path}' not found. See README.md for setup/download instructions.")


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


def cmd_plots(args):
    from belgian_dwellings.plots.result_plots import plot_all_paper_plots

    warn_if_missing(["results/belgium_usefull_500.csv", "data/houses_belgium_500"])

    log("Generating paper figures from results/belgium_usefull_500.csv")
    plot_all_paper_plots()
    log("Figures written to results/figures/")


# ---------------------------------------------------------------------------
# train treec
# ---------------------------------------------------------------------------


def cmd_train_treec(args):
    from belgian_dwellings.training.hpc_run import hpc_run

    warn_if_missing([f"data/houses_belgium_{args.tot_houses}"])

    houses = parse_house_spec(args.houses, args.tot_houses)
    log(
        f"Training TreeC models for {len(houses)} house(s) out of {args.tot_houses} "
        f"(gen={args.gen}, pop_size={args.pop_size}, max_trees={args.max_trees})"
    )

    # TreeLogger creates treec_train_<N>/house_<i> with os.mkdir (not os.makedirs), so the
    # root folder must exist up front - true by default for tot_houses=500 (from Zenodo),
    # but not for a fresh/custom tot_houses with nothing trained yet.
    os.makedirs(f"treec_train_{args.tot_houses}", exist_ok=True)

    worker = functools.partial(
        hpc_run,
        tot_houses=args.tot_houses,
        gen=args.gen,
        pop_size=args.pop_size,
        max_train=args.max_trees,
    )

    # The optimizer prints its own per-generation table, so track whole houses here.
    bar = progress_bar(len(houses), "Houses trained", leave=True)
    if args.workers <= 1:
        for house_num in houses:
            worker(house_num)
            bar.update(1)
    else:
        # Each house's training already parallelizes its population evaluation internally,
        # so only use --workers > 1 if you have enough cores to run several houses at once.
        with multiprocessing.Pool(processes=args.workers) as pool:
            for _ in pool.imap_unordered(worker, houses):
                bar.update(1)
    bar.close()


# ---------------------------------------------------------------------------
# train mpc-forecast
# ---------------------------------------------------------------------------


def cmd_train_mpc_forecast(args):
    from belgian_dwellings.simulation.calibrate_mpc import get_energyplus_calibration
    from belgian_dwellings.simulation.tmp_2023_config import tmp_2023_config

    warn_if_missing([f"data/houses_belgium_{args.tot_houses}"])

    houses = parse_house_spec(args.houses, args.tot_houses)
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(
        args.results_dir, f"mpc_calibration_{args.tot_houses}.csv"
    )

    already_done = set()
    if os.path.exists(out_path) and not args.refresh:
        with open(out_path, "r") as f:
            for line in f.readlines()[1:]:
                already_done.add(int(line.split(",")[0]))
    else:
        with open(out_path, "w") as f:
            f.write(
                "house_num,config_file,main_eff,cool_eff,backup_eff,ga,therm_cap,therm_res\n"
            )

    config_dir = f"data/houses_belgium_{args.tot_houses}"
    log(f"Fitting MPC calibration + EV forecaster for {len(houses)} house(s)")
    bar = progress_bar(len(houses), "Houses calibrated", leave=True)
    for house_num in houses:
        if house_num in already_done:
            log(f"house_{house_num}: already calibrated, skipped (--refresh to redo)")
            bar.update(1)
            continue

        config_file = f"house_{house_num}.json"
        config_path = os.path.join(config_dir, config_file)
        tmp_config_path = tmp_2023_config(config_path)

        calibration_list, _ = get_energyplus_calibration(
            tmp_config_path, progress_desc=f"house_{house_num}"
        )
        bar.update(1)

        with open(out_path, "a") as f:
            if not calibration_list:
                # Dwelling has no EnergyPlus/heat-pump asset (e.g. no central heating).
                f.write(f"{house_num},{config_file},,,,,,\n")
            else:
                cal = calibration_list[0]
                f.write(
                    f"{house_num},{config_file},{cal['main_eff']},{cal['cool_eff']},"
                    f"{cal['backup_eff']},{cal['ga']},{cal['therm_cap']},{cal['therm_res']}\n"
                )

    bar.close()

    tmp_config_dir = f"{config_dir}_tmp_2023"
    if os.path.exists(tmp_config_dir):
        import shutil

        shutil.rmtree(tmp_config_dir)

    log(f"Calibration summary written to {out_path}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

DEFAULT_MAIN_EMS = ["RBC_1.5h", "TreeC", "MPC_perfect", "MPC_realistic_forecast"]


def cmd_run(args):
    from belgian_dwellings.simulation.get_results import (
        generate_charge_completion_results_mpc_hpc,
        generate_charge_completion_results_treec,
        generate_results,
    )

    warn_if_missing(
        [
            f"data/houses_belgium_{args.tot_houses}",
            f"treec_train_{args.tot_houses}",
            "EnergyPlus-24.1.0",
        ]
    )

    houses = parse_house_spec(args.houses, args.tot_houses)

    main_file = f"{args.results_dir}/belgium_usefull_{args.tot_houses}.csv"
    soc_file = f"{args.results_dir}/belgium_usefull_{args.tot_houses}_no_enforcement.csv"

    if args.target in ("main", "all"):
        log(f"Running {', '.join(args.ems)} on {len(houses)} house(s) -> {main_file}")
        generate_results(
            house_num=args.tot_houses,
            ems_names=args.ems,
            refresh=args.refresh,
            num_process=args.num_process,
            houses=houses,
            results_dir=args.results_dir,
        )

    if args.target in ("no-enforcement", "all"):
        log(f"Running the no-enforcement variants on {len(houses)} house(s) -> {soc_file}")
        generate_charge_completion_results_treec(
            houses=houses, results_dir=args.results_dir
        )

        worker = functools.partial(
            generate_charge_completion_results_mpc_hpc, results_dir=args.results_dir
        )
        bar = progress_bar(
            len(houses), f"MPC_realistic_forecast_no_enforcement ({len(houses)} houses)"
        )
        if args.workers <= 1:
            for house_num in houses:
                worker(house_num)
                bar.update(1)
        else:
            with multiprocessing.Pool(processes=args.workers) as pool:
                for _ in pool.imap_unordered(worker, houses):
                    bar.update(1)
        bar.close()


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def add_house_selection_args(parser, default_tot_houses=500):
    parser.add_argument(
        "--tot-houses",
        type=int,
        default=default_tot_houses,
        help="Size of the house dataset to use, i.e. data/houses_belgium_<N>/ (default: 500, as in the paper)",
    )
    parser.add_argument(
        "--houses",
        default=None,
        help="Subset of house indices to process, e.g. '0-9,25,40-45' (default: all houses)",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce the results and plots of the 'Should Belgians install advanced EMS' paper."
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars (they also switch off automatically when the output "
        "is redirected to a file)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plots_parser = subparsers.add_parser(
        "plots", help="Regenerate all paper figures from results/belgium_usefull_500.csv"
    )
    plots_parser.set_defaults(func=cmd_plots)

    train_parser = subparsers.add_parser("train", help="Reproduce model training")
    train_subparsers = train_parser.add_subparsers(dest="train_target", required=True)

    treec_parser = train_subparsers.add_parser(
        "treec", help="Train (and validate) TreeC decision trees per house"
    )
    add_house_selection_args(treec_parser)
    treec_parser.add_argument("--gen", type=int, default=300, help="Generations (default: 300, as in the paper)")
    treec_parser.add_argument(
        "--pop-size", type=int, default=200, help="Population size (default: 200, as in the paper)"
    )
    treec_parser.add_argument(
        "--max-trees",
        type=int,
        default=5,
        help="Number of independent trainings per house before skipping it (default: 5, as in the paper)",
    )
    treec_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of houses to train concurrently (default: 1; each house's training already "
        "parallelizes internally, so only raise this if you have cores to spare)",
    )
    treec_parser.set_defaults(func=cmd_train_treec)

    mpc_forecast_parser = train_subparsers.add_parser(
        "mpc-forecast",
        help="Fit the MPC thermal calibration + EV forecaster per house and save a summary CSV",
    )
    add_house_selection_args(mpc_forecast_parser)
    mpc_forecast_parser.add_argument(
        "--results-dir", default="results", help="Where to write the calibration summary CSV (default: results)"
    )
    mpc_forecast_parser.add_argument(
        "--refresh", action="store_true", help="Recompute houses already present in the summary CSV"
    )
    mpc_forecast_parser.set_defaults(func=cmd_train_mpc_forecast)

    run_parser = subparsers.add_parser(
        "run", help="Run EMS simulations and write KPI results to a results folder"
    )
    add_house_selection_args(run_parser)
    run_parser.add_argument(
        "--ems",
        nargs="+",
        default=DEFAULT_MAIN_EMS,
        help=f"EMS variants to run for the main results file (default: {DEFAULT_MAIN_EMS})",
    )
    run_parser.add_argument(
        "--target",
        choices=["main", "no-enforcement", "all"],
        default="all",
        help="'main' runs --ems into the main results CSV, 'no-enforcement' runs the TreeC/MPC "
        "charge-completion variants needed for the SOC plots, 'all' runs both (default: all)",
    )
    run_parser.add_argument(
        "--results-dir",
        default="results",
        help="Folder to write results into, so original results aren't overwritten (default: results)",
    )
    run_parser.add_argument(
        "--num-process",
        type=int,
        default=None,
        help="Worker processes for the main EMS runs (default: cpu_count() - 1)",
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for the per-house MPC no-enforcement runs (default: 1)",
    )
    run_parser.add_argument(
        "--refresh", action="store_true", help="Recompute houses already present in the results CSV"
    )
    run_parser.set_defaults(func=cmd_run)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    set_progress_enabled(not args.no_progress)
    args.func(args)


if __name__ == "__main__":
    main()
