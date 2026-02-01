import pytz

from belgian_dwellings.simulation.rbc_ems import execute_rule_base
from simugrid.misc.log_plot_micro import plot_simulation
from simugrid.assets.energyplus import EnergyPlus

import matplotlib.pyplot as plt
import datetime

import numpy as np

plt.rcParams.update(
    {
        "font.size": 13,  # Default text size
        "axes.titlesize": 15,  # Axes title size
        "axes.labelsize": 13,  # Axes label size
        "xtick.labelsize": 11,  # X tick label size
        "ytick.labelsize": 11,  # Y tick label size
        "legend.fontsize": 11,  # Legend font size
    }
)

config_file = {
    "Microgrid": {
        "number_of_nodes": 1,
        "timezone": "UTC",
        "start_time": "2022/01/01 00:00:00",
        "end_time": "2022/01/10 00:00:00",
        "time_step": "00:15:00",
    },
    "Assets": {
        "Asset_0": {
            "node_number": 0,
            "name": "PublicGrid_0",
            "max_consumption_power": 100,
            "max_production_power": 100,
            "size": 1,
        },
        "Asset_1": {
            "node_number": 0,
            "name": "EnergyPlus_0",
            "timezone": "Europe/Brussels",
            "max_consumption_power": 30,
            "idf_model": "/home/django/Documents/Thesis_MOBI/Optimesh/ems_belgium_usefull/data/common/energyplus_models/apartment_<35m²_ground roof.idf",
            "epw_weather": "/home/django/Documents/Thesis_MOBI/Optimesh/ems_belgium_usefull/data/common/energyplus_models/brussels_2022.epw",
            "eplus_dir": "OpenStudioApplication/usr/local/openstudioapplication/EnergyPlus/",
            "occupancy": {
                "weekday": {
                    "00:00-07:00": "sleeping",
                    "07:00-08:00": "at home",
                    "08:00-18:00": "absent",
                    "18:00-23:15": "at home",
                    "23:15-24:00": "sleeping",
                },
                "saturday": {
                    "00:00-09:00": "sleeping",
                    "09:00-13:00": "at home",
                    "13:00-20:30": "absent",
                    "20:30-23:45": "at home",
                    "23:45-24:00": "sleeping",
                },
                "sunday": {
                    "00:00-08:30": "sleeping",
                    "08:30-22:15": "at home",
                    "22:15-24:00": "sleeping",
                },
            },
        },
    },
    "Environments": {
        "Environment_0": {
            "nodes_number": "0",
            "offtake_extra": 0.0,
            "day_ahead_price": "/home/django/Documents/Thesis_MOBI/Optimesh/ems_belgium_usefull/data/common/day_ahead_price/day_ahead_22_23.csv",
            "kwh_offtake_cost": 0.1257293,
            "injection_extra": -0.00905,
            "capacity_tariff": 3.4938417,
            "year_cost": 115.84,
        }
    },
}


def plot_first_week_profiles():

    microgrid = execute_rule_base(config_file)

    eplus_asset = microgrid.assets[1]

    datetimes = list()
    t_low = list()
    t_up = list()

    t_refs = list()

    for dt, bound_list in eplus_asset._cached_comfort_range["approximated"].items():

        datetimes.append(dt)
        local_dt = dt.astimezone(pytz.timezone(eplus_asset.timezone))
        t_ref = eplus_asset.get_t_ref(local_dt)
        t_refs.append(t_ref)
        low_comfort = bound_list[0]
        high_comfort = bound_list[1]

        t_low.append(low_comfort)
        t_up.append(high_comfort)

    fig, ax = plt.subplots()

    plt.ylim(3, 30)
    plt.ylabel("Temperature (°C)")
    plt.plot(datetimes, t_up, label="T$_{up}$")
    plt.plot(datetimes, t_low, label="T$_{low}$")

    plt.plot(datetimes, t_refs, label="T$_{e,ref}$")

    # Plot from the 3rd of January 2022 to the 10th of January 2022

    start_date = datetime.datetime.strptime("2022-01-03 00:00:00", "%Y-%m-%d %H:%M:%S")
    end_date = datetime.datetime.strptime("2022-01-10 00:00:00", "%Y-%m-%d %H:%M:%S")

    plt.xlim(start_date, end_date)

    fig.autofmt_xdate()
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "/home/django/Documents/Thesis_MOBI/Thesis/thesis_document/figures/relevance_belgium/comfort_range.pdf"
    )
    plt.show()


def plot_comfort_bound_equation(target="thesis"):
    if target == "thesis":
        plt.rcParams.update(
            {
                "font.size": 13,  # Default text size
                "axes.titlesize": 15,  # Axes title size
                "axes.labelsize": 13,  # Axes label size
                "xtick.labelsize": 11,  # X tick label size
                "ytick.labelsize": 11,  # Y tick label size
                "legend.fontsize": 11,  # Legend font size
            }
        )
    else:
        plt.rcParams.update(
            {
                "font.size": 13,  # Default text size
                "axes.titlesize": 15,  # Axes title size
                "axes.labelsize": 17,  # Axes label size
                "xtick.labelsize": 16,  # X tick label size
                "ytick.labelsize": 16,  # Y tick label size
                "legend.fontsize": 15,  # Legend font size
            }
        )

    t_e_refs = list(np.arange(-9, 29, 0.1))
    statuses = ["sleeping", "at home"]
    t_lows = {stat: [] for stat in statuses}
    t_ups = {stat: [] for stat in statuses}
    for t_e_ref in t_e_refs:
        for status in statuses:
            t_low, t_up = EnergyPlus.calc_comfort_range(t_e_ref, status)
            t_lows[status].append(t_low)
            t_ups[status].append(t_up)
    if target == "thesis":
        fig, ax = plt.subplots()
    elif target == "paper":
        fig, ax = plt.subplots(figsize=(10, 4))
    plt.ylabel("Indoor temperature (°C)")
    plt.xlabel("Reference outdoor temperature T$_{e,ref}$ (°C)")
    plt.plot(
        t_e_refs,
        t_ups["sleeping"],
        label="Comfort range when sleeping",
        color="tab:blue",
    )
    plt.plot(t_e_refs, t_lows["sleeping"], color="tab:blue")
    plt.plot(
        t_e_refs,
        t_ups["at home"],
        label="Comfort range when at home",
        color="tab:orange",
    )
    plt.xlim(-9, 28)
    plt.ylim(15, 32)
    plt.plot(t_e_refs, t_lows["at home"], color="tab:orange")
    plt.legend()
    plt.gca().set_axisbelow(True)
    plt.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    if target == "thesis":
        FIG_DIR = "/home/django/Documents/Thesis_MOBI/Thesis/thesis_document/figures/relevance_belgium/"
    elif target == "paper":
        FIG_DIR = "/sdd/storage/Documents/Thesis_MOBI/electrified_dwellings_paper/electrified_dwellings_paper/figures/"

    plt.savefig(f"{FIG_DIR}comfort_bound_equation.pdf")


if __name__ == "__main__":
    # plot_first_week_profiles()
    plot_comfort_bound_equation("paper")
    plt.show()
