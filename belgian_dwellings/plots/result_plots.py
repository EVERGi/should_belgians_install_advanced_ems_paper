from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import os


from belgian_dwellings.simulation.get_results import extract_info_from_config
import csv

import datetime
import numpy as np
from scipy.stats import linregress
import pytz
import json
import polars as pl

from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

target = "paper"
    
FIG_DIR = "results/figures/"

if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

figsize = (10, 4)

VIOLIN_PLOT = True
MEAN_VIOLIN = True

ems_ticks_label_dict = {
    "MPC_realistic_forecast": "MPC",
    "MPC_perfect": "MPC P",
    "RBC_1.5h": "RBC",
    "TreeC": "TreeC",
}

ylabel_dict = {
    "opex (€)": "Electricity cost (€)",
    "discomfort (Kh)": "Thermal discomfort (Kh)",
    "opex_diff (€/kWh)": "Electricity price difference with RBC (€/kWh)",
    "base_price (€/kWh)": "Electricity price (€/kWh)",
    "avg_soc_final (%)": "Mean departure\nSOC (%)",
    "base_price_no_pv (€/kWh)": "Electricity price for dwellings without PV (€/kWh)",
    "base_price_pv (€/kWh)": "Electricity price for dwellings with PV (€/kWh)",
    "simple_opex_diff (€)": "Electricity cost difference with RBC (€)",
}

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
    ylabel_dict["simple_opex_diff (€)"] = "Electricity cost difference\nwith RBC (€)"

ALPHA = 1.0


def order_info_set(info_set):
    if "ground wall roof" == info_set[0]:
        info_set = info_set[1:] + info_set[:1]
    elif "home charger" == info_set[0]:
        info_set = info_set[1:] + info_set[:1]
    elif "105-124m²" == info_set[0]:
        info_set = (
            [info_set[-1], info_set[4]] + info_set[1:4] + [info_set[0], info_set[5]]
        )
    elif "apartment unit" == info_set[0]:
        info_set = [info_set[2], info_set[1]] + info_set[3:] + [info_set[0]]
    elif "0.126" == info_set[0]:
        info_set = info_set[-1:] + info_set[:-1]
    elif "ground roof" == info_set[0]:
        info_set = info_set[2:] + info_set[:2]
    return info_set


def csv_to_dict(file_path, dt_format="%d/%m/%Y %H:%M:%S"):
    # Say the first column is str and the rest are float by reading first line

    # Read the first line of the csv as fast as possible
    with open(file_path, mode="r") as infile:
        first_line = infile.readline().strip()
    header = first_line.split(",")

    schema_overrides = {
        header_col: pl.String if header_col == "datetime" else pl.Float64
        for header_col in header
    }
    polars_df = pl.read_csv(file_path, schema_overrides=schema_overrides)
    return polars_df


def csv_to_dict_fast(file_path):
    with open(file_path, mode="r") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        csv_dict = {col: [None] * (365 * 24 * 4) for col in header}
        for row in reader:
            for i, col in enumerate(header):
                if col == "datetime":
                    csv_dict[col][i] = row[i]
                else:
                    csv_dict[col][i] = float(row[i])


def csv_to_dict_polars(file_path):
    df = pl.read_csv(file_path)
    # output = df.to_dict(as_series=False)
    return df


def get_max_power(config_path):
    json_config = json.load(open(config_path))
    assets = json_config["Assets"]
    max_powers = dict()
    for asset_id, asset in assets.items():
        # if "SolarPv_0" == asset["name"]:
        #    max_powers["pv"] = asset["max_production_power"]
        if "Charger_0" == asset["name"]:
            max_powers["home charger"] = asset["max_charge_cp"]
        elif "WaterHeater_0" == asset["name"]:
            max_powers["WaterHeater"] = asset["max_consumption_power"]
        elif "EnergyPlus_0" == asset["name"]:
            idf_model = asset["idf_model"]
            idf_path = os.path.dirname(config_path) + "/" + idf_model[2:]
            # Read the IDF file to get the max power
            with open(idf_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                line_clean = line.replace(" ", "")
                first_elem = line_clean.split(",")[0]
                if "Gross Rated Heating Capacity {W}" in line:
                    heat_capacity = float(first_elem)
                elif "Gross Rated Heating COP {W/W}" in line:
                    heat_cop = float(first_elem)
                elif "Gross Rated Total Cooling Capacity {W}" in line:
                    cooling_capacity = float(first_elem)
                elif "Gross Rated Cooling COP {W/W}" in line:
                    cooling_cop = float(first_elem)
                elif "Nominal Capacity {W}" in line:
                    max_backup_power = float(first_elem)
            # Calculate the max power
            max_heat_power = heat_capacity / heat_cop
            max_cooling_power = cooling_capacity / cooling_cop
            max_power = max_heat_power + max_backup_power
            max_powers["EnergyPlus"] = max_power / 1000

    sum_power = sum(max_powers.values())

    return sum_power


def get_proportion_flex(power_file_path):
    csv_dict = csv_to_dict(power_file_path)
    flex_assets = ["Charger_0", "WaterHeater_0", "EnergyPlus_0"]

    flex_consumption = 0
    for asset in flex_assets:
        if asset in csv_dict.columns:
            flex_consumption += -sum(csv_dict[asset])

    not_flex_consumption = -sum(csv_dict["Consumer_0"])

    return flex_consumption / (flex_consumption + not_flex_consumption) * 100


def total_plot(result_file, config_dir, ems_names=None):

    start_dt = datetime.datetime(2023, 1, 1, 0, 0, 0)
    end_dt = datetime.datetime(2024, 1, 1, 0, 0, 0)

    ems_base = "RBC_1.5h"
    data = get_data_results_file(result_file)
    data = add_soc_data(data, result_file, start_dt, end_dt)

    config_files = [config_file for config_file in data[ems_base]["config_file"]]

    config_paths = [config_dir + config_file for config_file in config_files]
    dwelling_info_data = get_dwelling_info(config_paths)
    data = add_base_price_with_without_pv(data, result_file, dwelling_info_data)

    if ems_names is None:
        ems_names = list(data.keys())

    to_plot = [
        "opex (€)",
        "discomfort (Kh)",
        "opex_diff (€/kWh)",
        "base_price (€/kWh)",
        "avg_soc_final (%)",
        "base_price_no_pv (€/kWh)",
        "base_price_pv (€/kWh)",
        "simple_opex_diff (€)",
    ]

    for y_axis in to_plot:
        plt.figure(figsize=(10, 3.5))
        tick_labels = [
            ems_ticks_label_dict.get(ems_name, ems_name) for ems_name in ems_names
        ]
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
        ]
        if y_axis == "discomfort (Kh)":
            boxplot_data = filter_discomfort(data, ems_names, dwelling_info_data)
        else:
            boxplot_data = [data[ems_name][y_axis] for ems_name in ems_names]
        print(f"Plotting {y_axis} with {len(boxplot_data[1])} points")

        if VIOLIN_PLOT:
            violin_parts = plt.violinplot(
                boxplot_data, showmedians=not MEAN_VIOLIN, showmeans=MEAN_VIOLIN
            )
            # Set tick labels
            plt.xticks(range(1, len(tick_labels) + 1), tick_labels)
            # Set colors for each violin
            set_violin_colors(violin_parts, colors)
            # pc.set_alpha(ALPHA)
            # if color == "#ff7f0e":
            #    violin_parts["medians"][i].set_color("white")
        else:
            bp = plt.boxplot(
                boxplot_data,
                tick_labels=tick_labels,
                patch_artist=True,
                boxprops=dict(facecolor="white", alpha=ALPHA),
            )
            print(f"Boxplot data for {y_axis}:")
            for i, patch in enumerate(bp["boxes"]):
                color = colors[i]
                patch.set_facecolor(color)
                if color == "#ff7f0e":
                    bp["medians"][i].set_color("white")

        print(f"Boxplot data for {y_axis}:")
        for i, name in enumerate(ems_names):
            mean_box = np.mean(boxplot_data[i])
            print(f"Mean for {name}: {mean_box:.4f}")

        if y_axis.startswith("base_price"):
            plt.ylim(0.075, 0.35)

        # plt.yticks(fontsize=16)
        y_label = ylabel_dict[y_axis]
        plt.ylabel(y_label)
        plt.gca().set_axisbelow(True)
        plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.tight_layout()
        if y_axis != "base_price (€/kWh)":
            plt.savefig(FIG_DIR + f"total_{y_axis.split(" ")[0]}.pdf")


def filter_discomfort(data, ems_names, dwelling_info_data):
    # Get infor for each house if it has central heating, if it doesn't then remove it from the plotted data
    box_plot_data = []
    for ems_name in ems_names:
        ems_data = data[ems_name]
        insulation_info = dwelling_info_data["insulation"]
        filtered_data = [
            discomfort
            for i, discomfort in enumerate(ems_data["discomfort (Kh)"])
            if insulation_info[i] != "no central heating"
        ]
        box_plot_data.append(filtered_data)

    return box_plot_data


def add_soc_data(data, result_file, start_dt, end_dt):

    all_session_soc_i = get_all_soc_init(start_dt, end_dt)
    a = 0

    json_file = result_file.replace(".csv", "_no_enforcement_soc.json")

    content = json.load(open(json_file))

    all_config_files = list(content.keys())
    charging_sessions_files = {
        config_file: get_charging_session_file(config_file, result_file)
        for config_file in all_config_files
    }
    for ems_name in data.keys():
        new_ems_name = ems_name + "_no_enforcement"
        avg_soc_final_list = []
        for config_file in data[ems_name]["config_file"]:
            if (
                config_file not in content.keys()
                or new_ems_name not in content[config_file].keys()
            ):
                continue
            if content[config_file][new_ems_name] == []:
                continue
            avg_soc = 95 - np.mean(content[config_file][new_ems_name]) * 100
            avg_soc_final_list.append(avg_soc)

        data[ems_name]["avg_soc_final (%)"] = avg_soc_final_list

    for ems_name in data.keys():
        if len(data[ems_name]["avg_soc_final (%)"]) == 0:
            data[ems_name]["avg_soc_final (%)"] = [95]

    return data


def add_base_price_with_without_pv(data, result_file, dwelling_info_data):
    for ems_name in data.keys():
        data[ems_name]["base_price_no_pv (€/kWh)"] = []
        data[ems_name]["base_price_pv (€/kWh)"] = []
        for i, _ in enumerate(data[ems_name]["config_file"]):
            pv_info = dwelling_info_data["pv ?"][i].replace(" ", "_")
            base_price = data[ems_name]["base_price (€/kWh)"][i]
            data[ems_name][f"base_price_{pv_info} (€/kWh)"].append(base_price)
    return data


def feature_plot(
    result_file,
    config_dir,
    ems_names=None,
    y_axis="opex_diff (€/kWh)",
):

    data = get_data_results_file(result_file)

    ems_base = "RBC_1.5h"
    if ems_names is None:
        ems_names = list(data.keys())
    plotted_ems = ems_names.copy()
    plotted_ems.remove(ems_base)

    # Define colors for each EMS
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    ems_colors = {ems: colors[i % len(colors)] for i, ems in enumerate(plotted_ems)}

    # Plot opex diff (€/kWh) with and without PV for as a boxplot
    # plt.figure(figsize=(10, 5))
    config_files = [config_file for config_file in data[ems_base]["config_file"]]
    config_paths = [config_dir + config_file for config_file in config_files]

    dwelling_info_data = get_dwelling_info(config_paths)

    for key in dwelling_info_data.keys():
        info_set = sorted(set(dwelling_info_data[key]))
        info_set = order_info_set(info_set)
        box_plot_data = dict()
        for ems_name in plotted_ems:

            box_plot_data[ems_name] = []
            for info in info_set:
                info_data = []
                for i, opex_diff in enumerate(data[ems_name][y_axis]):
                    # print(len(dwelling_info_data[key]))
                    # print(dwelling_info_data[key][88])
                    if dwelling_info_data[key][i] == info:
                        info_data.append(opex_diff)
                box_plot_data[ems_name].append(info_data)
        plt.figure(figsize=(10, 5))
        num_ems = len(plotted_ems)
        base_position = [i for i, _ in enumerate(info_set)]
        width = 0.5 / len(plotted_ems)

        # Store boxplot patches for legend
        legend_patches = []

        for ems_pos, ems_name in enumerate(plotted_ems):
            offset = -0.5 + (ems_pos + 1) / (num_ems + 1)
            positions = [pos + offset for pos in base_position]
            bp = plt.boxplot(
                box_plot_data[ems_name],
                positions=positions,
                widths=width,
                patch_artist=True,
                boxprops=dict(facecolor=ems_colors[ems_name], alpha=ALPHA),
            )
            # Store the first box patch for legend
            legend_patches.append(bp["boxes"][0])

        offset = -0.5 + (len(plotted_ems) + 1) / 2 / (num_ems + 1)
        tick_positions = [pos + offset for pos in base_position]

        plt.xticks(tick_positions, info_set)

        # Add legend with EMS names and colors
        legend_labels = [ems_ticks_label_dict.get(ems, ems) for ems in plotted_ems]
        plt.legend(legend_patches, legend_labels)

        # plt.xlabel(key)
        plt.ylabel(y_axis)
        plt.title(f"{y_axis} for {key}")
        plt.tick_params(axis="both", which="major", labelsize=13)
        plt.xticks(rotation=45)
        plt.tight_layout()


def all_ems_plot(
    result_file,
    config_dir,
    ems_names,
    y_axis="opex_diff (€/kWh)",
    show_num_points=False,
):
    data = get_data_results_file(result_file)
    ems_base = "RBC_1.5h"

    config_files = [config_file for config_file in data[ems_base]["config_file"]]
    config_paths = [config_dir + config_file for config_file in config_files]

    dwelling_info_data = get_dwelling_info(config_paths)

    for key in dwelling_info_data.keys():
        if key in ["construction type"]:  # , "surface area"]:
            continue
        ems_plot(
            result_file,
            config_dir,
            ems_names=ems_names,
            y_axis=y_axis,
            feature_key=key,
            show_num_points=show_num_points,
        )


def ems_plot(
    result_file,
    config_dir,
    ems_names=None,
    y_axis="opex_diff (€/kWh)",
    feature_key="pv ?",
    show_num_points=False,
):
    data = get_data_results_file(result_file)

    ems_base = "RBC_1.5h"
    if ems_names is None:
        ems_names = list(data.keys())
    plotted_ems = ems_names.copy()
    if ems_base in plotted_ems:
        plotted_ems.remove(ems_base)

    config_files = [config_file for config_file in data[ems_base]["config_file"]]
    config_paths = [config_dir + config_file for config_file in config_files]

    dwelling_info_data = get_dwelling_info(config_paths)

    if feature_key not in dwelling_info_data:
        print(f"Feature key '{feature_key}' not found in dwelling info data")
        return

    info_set = sorted(set(dwelling_info_data[feature_key]))
    info_set = order_info_set(info_set)

    colors = {
        "pv": "#ffd700",
        "home charger": "#2E8B57",
        "Little tank (126 L)": "#87CEEB",
        "Medium tank (171 L)": "#4682B4",
        "Large tank (243-363 L)": "#191970",
        "No insulation": "#FF0000",
        "Partial insulation": "#FDAA44FF",
        "Full insulation": "#32CD32",
        0: "#ffffff",
        1: "#e0b3ff",
        2: "#d480ff",
        3: "#c44dff",
        4: "#b81aff",
    }

    plt.figure(figsize=figsize)

    box_plot_data = {info: [] for info in info_set}

    for info in info_set:
        for ems_name in plotted_ems:
            info_data = []
            for i, value in enumerate(data[ems_name][y_axis]):
                if dwelling_info_data[feature_key][i] == info:
                    info_data.append(value)
            box_plot_data[info].append(info_data)

    merge_info = {
        "No insulation": ["none"],
        "Partial insulation": ["roof", "ground roof", "wall roof"],
        "Full insulation": ["ground wall roof"],
        "Little tank (126 L)": ["0.126"],
        "Medium tank (171 L)": ["0.171"],
        "Large tank (243-363 L)": ["0.243", "0.33", "0.363"],
    }
    for info, merge_list in merge_info.items():
        for merge_item in merge_list:
            if merge_item in box_plot_data:
                if info not in box_plot_data:
                    box_plot_data[info] = []
                for ems_ind, data_list in enumerate(box_plot_data[merge_item]):
                    if ems_ind >= len(box_plot_data[info]):
                        box_plot_data[info].append(data_list)
                    else:
                        box_plot_data[info][ems_ind].extend(data_list)
                del box_plot_data[merge_item]
    info_set = list(box_plot_data.keys())

    num_features = len(info_set)
    base_position = [i for i, _ in enumerate(plotted_ems)]
    width = 0.5 / len(info_set)

    # Store boxplot patches for legend
    legend_patches = []

    for feature_pos, info in enumerate(info_set):
        offset = -0.5 + (feature_pos + 1) / (num_features + 1)
        positions = [pos + offset for pos in base_position]
        color = colors.get(info, "#ffffff")  # Default color if not found
        if VIOLIN_PLOT:
            violin_parts = plt.violinplot(
                box_plot_data[info],
                positions=positions,
                widths=width,
                showmedians=not MEAN_VIOLIN,
                showmeans=MEAN_VIOLIN,
            )
            set_violin_colors(violin_parts, [color] * len(positions))
            # Store the first violin patch for legend
            legend_patches.append(violin_parts["bodies"][0])

        else:
            bp = plt.boxplot(
                box_plot_data[info],
                positions=positions,
                widths=width,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=ALPHA),
            )
            # Store the first box patch for legend
            legend_patches.append(bp["boxes"][0])

        if show_num_points:
            # Add sample count text for each boxplot
            for i, pos in enumerate(positions):
                n_samples = len(box_plot_data[info][i])
                plt.text(
                    pos,
                    plt.ylim()[0] + 0.01 * (plt.ylim()[1] - plt.ylim()[0]),
                    f"n={n_samples}",
                    ha="center",
                    va="bottom",
                )

    offset = -0.5 + (len(info_set) + 1) / 2 / (num_features + 1)
    tick_positions = [pos + offset for pos in base_position]

    tick_labels = [
        ems_ticks_label_dict.get(ems_name, ems_name) for ems_name in plotted_ems
    ]
    plt.xticks(tick_positions, tick_labels)
    # plt.yticks(fontsize=16)

    label_dict = {
        "pv": "PV",
        "no pv": "No PV",
        "home charger": "EV charger",
        "no home charger": "No EV charger",
        "no central heating": "No heat pump",
        "no water heater": "No water heater",
        0: "No assets",
        1: "1 asset",
        2: "2 assets",
        3: "3 assets",
        4: "4 assets",
    }
    # Add legend with feature values and colors
    label_name = [label_dict.get(info, info) for info in info_set]
    plt.legend(legend_patches, label_name, loc="upper left")

    if y_axis == "opex_diff (€/kWh)":
        # plt.ylim(-70, 510)
        plt.ylim(-0.01, 0.05)
    elif y_axis == "simple_opex_diff (€)":
        plt.ylim(-70, 510)

    plt.ylabel(ylabel_dict.get(y_axis, y_axis))
    # plt.title(f"{y_axis} for {feature_key}")
    plt.gca().set_axisbelow(True)

    plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)

    plt.tight_layout()

    file_name = f"{feature_key.replace(" ?", "").replace(" ","_")}_results.pdf"
    plt.savefig(FIG_DIR + file_name)


def get_data_results_file(result_file, base_ems="RBC_1.5h"):
    data_results = dict()
    with open(result_file, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        splt_lines = line.split(",")
        config_file = splt_lines[0]
        ems_name = splt_lines[1]
        opex = splt_lines[2]
        discomfort = splt_lines[3]
        if ems_name not in data_results.keys():
            data_results[ems_name] = dict()
        if "config_file" not in data_results[ems_name].keys():
            data_results[ems_name]["config_file"] = []
            data_results[ems_name]["opex (€)"] = []
            data_results[ems_name]["discomfort (Kh)"] = []

            data_results[ems_name]["consumption_only (kWh)"] = []
            data_results[ems_name]["grid_offtake (kWh)"] = []
            data_results[ems_name]["grid_injection (kWh)"] = []

        data_results[ems_name]["config_file"].append(config_file)
        data_results[ems_name]["opex (€)"].append(-float(opex))
        data_results[ems_name]["discomfort (Kh)"].append(float(discomfort))

        if len(splt_lines) > 4:
            consumption_only = splt_lines[4]
            grid_offtake = splt_lines[5]
            grid_injection = splt_lines[6]
            data_results[ems_name]["consumption_only (kWh)"].append(
                float(consumption_only)
            )
            data_results[ems_name]["grid_offtake (kWh)"].append(float(grid_offtake))
            data_results[ems_name]["grid_injection (kWh)"].append(float(grid_injection))

    ems_names = set(data_results.keys())
    for ems_name in ems_names:
        opex_diff = [
            (data_results[base_ems]["opex (€)"][i] - opex_ems)
            / data_results[base_ems]["consumption_only (kWh)"][i]
            for i, opex_ems in enumerate(data_results[ems_name]["opex (€)"])
        ]
        simple_opex_diff = [
            data_results[base_ems]["opex (€)"][i] - opex_ems
            for i, opex_ems in enumerate(data_results[ems_name]["opex (€)"])
        ]
        data_results[ems_name]["simple_opex_diff (€)"] = simple_opex_diff

        data_results[ems_name]["opex_diff (€/kWh)"] = opex_diff
        base_price_kwh = [
            opex_ems / data_results[base_ems]["consumption_only (kWh)"][i]
            for i, opex_ems in enumerate(data_results[ems_name]["opex (€)"])
        ]
        data_results[ems_name]["base_price (€/kWh)"] = base_price_kwh

    return data_results


def get_dwelling_info(config_paths):
    dwelling_info_data = {}
    for config_path in config_paths:
        dwelling_info = extract_info_from_config(config_path)
        for key, value in dwelling_info.items():
            if key not in dwelling_info_data.keys():
                dwelling_info_data[key] = [value]
            else:
                dwelling_info_data[key].append(value)

    return dwelling_info_data


def get_reward_history(result_file):
    with open(result_file, "r") as f:
        lines = f.readlines()

    reward_history = dict()
    for line in lines[1:]:
        splt_line = line.split(",")
        config_file = splt_line[0]
        ems_name = splt_line[1]
        reward_key = f"{config_file}_{ems_name}"
        reward_history[reward_key] = dict()
        profile_dir = result_file.replace(".csv", "_profiles/")
        reward_file = profile_dir + f"{reward_key}_reward.csv"
        reward_dict = csv_to_dict(reward_file)

        reward_history[reward_key] = reward_dict
    return reward_history


def get_opex_composition(result_file):
    reward_history = get_reward_history(result_file)
    opex_composition = dict()
    for reward_key, kpi_history in reward_history.items():
        opex_composition[reward_key] = dict()
        for reward_type in kpi_history.columns:
            if reward_type not in ["datetime", "discomfort", "forced_charged_energy"]:
                opex_composition[reward_key][reward_type] = -kpi_history[reward_type][
                    -1
                ]

    return opex_composition


def calc_composition_diff(opex_composition, data_results, base_ems="RBC_1.5h"):
    composition_diff = dict()
    for ems_name in data_results.keys():
        config_files = data_results[ems_name]["config_file"]
        for i, config_file in enumerate(config_files):
            reward_key = f"{config_file}_{ems_name}"
            base_key = f"{config_file}_{base_ems}"
            composition_diff[reward_key] = dict()
            for reward_type in opex_composition[reward_key].keys():
                composition_diff[reward_key][f"{reward_type} diff (€/kWh)"] = (
                    opex_composition[reward_key][reward_type]
                    - opex_composition[base_key][reward_type]
                )  # / data_results[base_ems]["consumption_only (kWh)"][i]
    return composition_diff


def difference_MPC_RBC(result_file, config_dir):
    base_ems = "RBC_1.5h"
    with open(result_file, "r") as f:
        lines = f.readlines()

    data_structured = get_data_results_file(result_file)
    # print(data_structured)
    to_plot = {
        "Yearly electricity bill (€)": data_structured[base_ems]["opex (€)"],
        "Yearly bill divided by consumption (€/kWh)": data_structured[base_ems][
            "base_price (€/kWh)"
        ],
        "Difference with rule-based control (€/kWh)": data_structured["MPC_perfect"][
            "opex_diff (€/kWh)"
        ],
        "base_price (€/kWh)": data_structured[base_ems]["base_price (€/kWh)"],
    }

    config_files = data_structured[base_ems]["config_file"]
    pv_in_config = [
        extract_info_from_config(config_dir + config_file)["pv ?"] == "pv"
        for config_file in config_files
    ]

    colors_pv = {True: "#FF6347", False: "#4682B4"}
    colors = [colors_pv[pv] for pv in pv_in_config]
    for key in to_plot.keys():

        plt.figure(figsize=(10, 5))
        x_axis = [i + 1 for i, _ in enumerate(config_files)]
        plt.bar(x_axis, to_plot[key], color=colors)

        # Plot the average line
        avg = np.mean(to_plot[key])
        plt.axhline(y=avg, color="red", linestyle="--", label="Average")
        # Indicate the average value
        # plt.text(
        #    len(config_files) + 0.5,
        #    avg,
        #    f"Avg: {avg:.2f}",
        #    color="red",
        #    verticalalignment="center",
        # )
        plt.xlim(-1, len(config_files) + 2)
        plt.ylabel(key)
        # plt.title(key)
        # plt.xticks(rotation=45)
        plt.xlabel("Dwelling number")
        # Add manual legend
        for pv in colors_pv.keys():
            if pv:
                label = "With PV"
            else:
                label = "Without PV"
            plt.bar([0], [0], color=colors_pv[pv], label=label)
        plt.legend()
        plt.tight_layout()


def plot_proportion_flex_vs_diff_opex(result_file, config_dir, ems_name="MPC_perfect"):
    data_results = get_data_results_file(result_file)

    base_ems = "RBC_1.5h"
    config_files = data_results[base_ems]["config_file"]
    flex_power_prop = []
    max_power_setup = []
    for config_file in config_files:
        power_dir = result_file.replace(".csv", "_profiles/")
        power_filepath = power_dir + config_file + f"_{base_ems}_power.csv"
        flex_power_prop.append(get_proportion_flex(power_filepath))
        config_path = config_dir + config_file
        max_power = get_max_power(config_path)
        max_power_setup.append(max_power)

    diff_opex = data_results[ems_name]["opex_diff (€/kWh)"]
    plt.figure(figsize=(10, 5))
    plt.scatter(flex_power_prop, diff_opex)

    # Add regression line

    slope, intercept, r_value, p_value, std_err = linregress(flex_power_prop, diff_opex)
    regression_line = np.array(flex_power_prop) * slope + intercept
    plt.plot(
        flex_power_prop,
        regression_line,
        color="red",
        label=f"Regression line (R²={r_value**2:.2f}, p-value={p_value:.2e})",
    )

    plt.xlabel("Proportion of consumed energy from controllable assets (%)")
    plt.ylabel("Difference with rule-based control (€/kWh)")
    # plt.title("Difference in opex (€/kWh) vs proportion of flexible power")
    plt.tight_layout()
    plt.legend()

    # Plot the max power setup
    plt.figure(figsize=(10, 5))
    plt.scatter(max_power_setup, diff_opex)
    slope, intercept, r_value, p_value, std_err = linregress(max_power_setup, diff_opex)
    regression_line = np.array(max_power_setup) * slope + intercept
    plt.plot(
        max_power_setup,
        regression_line,
        color="red",
        label=f"Regression line (R²={r_value**2:.2f}, p-value={p_value:.2e})",
    )
    plt.xlabel("Max power setup (kW)")
    plt.ylabel("Difference with rule-based control (€/kWh)")

    plt.tight_layout()
    plt.legend()

    """
    config_paths = [config_dir + config_file for config_file in config_files]
    dwelling_info_data = get_dwelling_info(config_paths)

    for key in dwelling_info_data.keys():
        info_set = sorted(set(dwelling_info_data[key]))
        info_set = order_info_set(info_set)
        plt.figure(figsize=(10, 5))
        for info in info_set:
            flex_power_prop_info = []
            diff_opex_info = []
            for i, info_i in enumerate(dwelling_info_data[key]):
                if info_i == info:
                    flex_power_prop_info.append(flex_power_prop[i])
                    diff_opex_info.append(diff_opex[i])

            plt.scatter(flex_power_prop_info, diff_opex_info, label=info)

        plt.xlabel("Proportion of flexible power (%)")
        plt.ylabel("Difference in opex (€/kWh)")
        plt.tight_layout()
        plt.legend()
    """


def plot_opex_diff_separate(
    result_file,
    ems_names=None,
):
    opex_composition = get_opex_composition(result_file)
    data_results = get_data_results_file(result_file)
    if ems_names is None:
        ems_names = list(set(data_results.keys()))

    ems_base = "RBC_1.5h"
    comp_diff = calc_composition_diff(opex_composition, data_results, ems_base)

    config_files = [
        config_file for config_file in data_results[ems_base]["config_file"]
    ]

    plt.figure(figsize=figsize)
    reward_colors = {
        "trans_dis": "tab:orange",
        "day_ahead": "tab:blue",
        "capacity": "tab:green",
    }
    label_names = {
        "day_ahead": "Day-ahead",
        "trans_dis": "Offtake extras",
        "capacity": "Peak",
    }
    all_rewards = list(reward_colors.keys())
    width = 0.8 / len(all_rewards)
    labels = []
    colors = []

    legend_patches = []
    for tick, ems_name in enumerate(ems_names):
        for rew_ind, reward_type in enumerate(all_rewards):
            offset = (rew_ind - len(all_rewards) / 2 + 0.5) * width
            x_pos = tick + offset
            if reward_type == "opex":
                continue
            diff_info = []
            for config_file in config_files:
                reward_key = f"{config_file}_{ems_name}"
                diff_info.append(comp_diff[reward_key][f"{reward_type} diff (€/kWh)"])

            std_diff = np.std(diff_info)
            color = reward_colors.get(reward_type, "black")

            if ems_name == "TreeC":
                label = label_names[reward_type]
                labels += [label]
                colors += [color]
            else:
                label = None

            mean_diff = -np.mean(diff_info)
            neg_dif_info = [-x for x in diff_info]
            if color == "tab:orange":
                medianprops = dict(color="white")
            else:
                medianprops = None
            if VIOLIN_PLOT:
                violin_parts = plt.violinplot(
                    [neg_dif_info],
                    positions=[x_pos],
                    widths=width,
                    showmedians=not MEAN_VIOLIN,
                    showmeans=MEAN_VIOLIN,
                    # showmeans=True,
                    # widths=width,
                    # patch_artist=True,
                    # boxprops=dict(facecolor=color, alpha=ALPHA),
                    # medianprops=medianprops,
                    # label=label,
                )
                set_violin_colors(violin_parts, [color])
                # Store the first violin patch for legend
                if label is not None:
                    legend_patches.append(violin_parts["bodies"][0])

            else:
                pc = plt.boxplot(
                    neg_dif_info,
                    positions=[x_pos],
                    widths=width,
                    patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=ALPHA),
                    medianprops=medianprops,
                    label=label,
                )
                if label is not None:
                    legend_patches.append(pc["boxes"][0])

    xticks = [ems_ticks_label_dict.get(ems_name, ems_name) for ems_name in ems_names]
    plt.xticks(ticks=range(len(ems_names)), labels=xticks)

    ylim = plt.gca().get_ylim()

    # Put y ticks ever 0.01
    # plt.yticks(np.arange(-0.01, 0.035, 0.005))
    # plt.ylim(ylim[0], ylim[1])

    # handles, labels = plt.gca().get_legend_handles_labels()

    # Add the whisker to the legend
    # whisker_proxy = Line2D(
    #    [0], [0], color="black", linewidth=1.5, label="Standard deviation"
    # )
    # handles.append(whisker_proxy)
    plt.ylabel(ylabel_dict["simple_opex_diff (€)"])

    plt.legend(
        legend_patches,
        labels,
        loc="upper right",
    )
    plt.gca().set_axisbelow(True)
    plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)

    plt.tight_layout()

    plt.savefig(FIG_DIR + "opex_diff_separate.pdf")


def set_violin_colors(violin_parts, colors):
    """
    Set the color of the violin plot parts.
    """

    part_colors = [col if col != "#ffffff" else "gray" for col in colors]
    for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):

        if partname not in violin_parts:
            continue
        vp = violin_parts[partname]
        vp.set_colors(part_colors)

    for i, pc in enumerate(violin_parts["bodies"]):
        pc.set_color(colors[i])
        if colors[i] == "#ffffff":
            pc.set_edgecolor("black")
        else:
            pc.set_edgecolor(colors[i])

        pc.set_facecolor(colors[i])


def plot_opex_composition(result_file, ems_names=None):
    opex_composition = get_opex_composition(result_file)
    data_results = get_data_results_file(result_file)

    if ems_names is None:
        ems_names = list(set(data_results.keys()))

    ems_base = "RBC_1.5h"

    to_plot = dict()
    for ems_name in ems_names + ["RBC_1.5h"]:
        to_plot[ems_name] = {"mean": dict(), "samples": dict()}
        for reward_type in opex_composition[list(opex_composition.keys())[0]].keys():

            all_reward = [
                opex_composition[f"{config_file}_{ems_name}"][reward_type]
                for config_file in data_results[ems_name]["config_file"]
            ]
            if reward_type == "opex":
                print(np.mean(all_reward))
            else:
                to_plot[ems_name]["mean"][reward_type] = np.mean(all_reward)

    plt.figure(figsize=(10, 5))

    # Stacked bar plot of mean seperated by ems_name
    bottom = np.zeros(len(ems_names))
    for reward_type in opex_composition[list(opex_composition.keys())[0]].keys():

        if reward_type == "opex":
            continue
        reward_values = [
            to_plot[ems_name]["mean"][reward_type] for ems_name in ems_names
        ]

        plt.bar(ems_names, reward_values, bottom=bottom, label=reward_type)
        bottom += reward_values

    plt.ylabel("Mean value (€)")
    plt.title("Mean value of opex composition")
    plt.legend()
    plt.tight_layout()

    comp_diff = calc_composition_diff(opex_composition, data_results, ems_base)

    config_files = [
        config_file for config_file in data_results[ems_base]["config_file"]
    ]

    plt.figure()
    for ems_name in ems_names:
        for reward_type in opex_composition[list(opex_composition.keys())[0]].keys():
            if reward_type == "opex":
                continue
            diff_info = []
            for config_file in config_files:
                reward_key = f"{config_file}_{ems_name}"
                diff_info.append(comp_diff[reward_key][f"{reward_type} diff (€/kWh)"])
            plt.plot(
                config_files,
                diff_info,
                label=f"{ems_name} - {reward_type}",
                marker="o",
            )

    config_paths = [config_dir + config_file for config_file in config_files]

    dwelling_info_data = get_dwelling_info(config_paths)

    diff_to_plot = ["day_ahead", "trans_dis", "capacity"]

    color_diff = {"day_ahead": "#FF6347", "trans_dis": "#4682B4", "capacity": "#32CD32"}

    plotted_ems = ems_names.copy()
    try:
        plotted_ems.remove(ems_base)
    except ValueError:
        pass
    for ems_name in plotted_ems:
        for key in dwelling_info_data.keys():
            info_set = sorted(set(dwelling_info_data[key]))
            info_set = order_info_set(info_set)
            plt.figure(figsize=(10, 5))

            for tick, info in enumerate(info_set):
                width_bar = 0.15
                for i, diff in enumerate(diff_to_plot):
                    diff_info = []
                    for j, config_file in enumerate(config_files):
                        reward_key = f"{config_file}_{ems_name}"
                        if dwelling_info_data[key][j] == info:
                            diff_info.append(
                                comp_diff[reward_key][f"{diff} diff (€/kWh)"]
                            )
                    mean_diff = np.mean(diff_info)
                    if info == info_set[0]:
                        label = diff
                    else:
                        label = None
                    plt.bar(
                        tick + (i - 1) * width_bar,
                        mean_diff,
                        width_bar,
                        label=label,
                        align="center",
                        color=color_diff[diff],
                    )

            plt.xticks(ticks=range(len(info_set)), labels=list(info_set))
            plt.ylabel("Mean value (€/kWh)")
            plt.title(f"Mean value of opex composition difference {ems_name}")
            plt.legend()
            plt.tight_layout()


def plot_average_daily_offtake(result_file, ems_names=None):
    start_year = 2023
    price_file = "data/common/day_ahead_price/day_ahead_22_23.csv"
    price_dict = csv_to_dict(price_file)

    data_results = get_data_results_file(result_file)
    if ems_names is None:
        ems_names = list(data_results.keys())
    config_files = data_results[ems_names[0]]["config_file"]
    grid_hist = {ems: [] for ems in ems_names}
    for config_file in config_files:
        for ems_name in ems_names:
            power_dir = result_file.replace(".csv", "_profiles/")
            power_filepath = power_dir + config_file + f"_{ems_name}_power.csv"
            power_dict = csv_to_dict(power_filepath)
            grid_hist[ems_name].append(np.array(power_dict["PublicGrid_0"]))

    plt.figure(figsize=(10, 5))
    for ems_name in ems_names:
        x_axis = [
            datetime.datetime(start_year, 1, 1, 0, 0)
            + datetime.timedelta(minutes=i * 15)
            for i in range(365 * 24 * 4)
        ]
        # Put the dates to the Brussels timezone
        brussels = pytz.timezone("Europe/Brussels")
        x_axis = [
            time.replace(tzinfo=datetime.timezone.utc).astimezone(brussels)
            for time in x_axis
        ]

        yearly_mean = np.mean(grid_hist[ems_name], axis=0)

        plt.step(x_axis, yearly_mean, label=ems_name, alpha=ALPHA, where="post")
        plt.legend(loc="upper right")

    plt.ylabel("Grid offtake (kW)")
    plt.ylim(bottom=0)

    price_history = price_dict["price (€/kWh)"]
    for i, price in enumerate(price_history):
        time_index = price_dict["datetime"][i]
        if time_index == f"{start_year}-01-01T00:00:00Z":
            start_index = i
        elif time_index == f"{start_year+1}-01-01T00:00:00Z":
            end_index = i
            break
    price_history = price_history[start_index:end_index]

    plt.legend(loc="upper right")
    ax2 = plt.twinx()
    ax2.step(
        x_axis,
        price_history,
        color="red",
        label="Day ahead price (€/kWh)",
        alpha=0.2,
        where="post",
    )
    ax2.set_ylabel("Dynamic price (€/kWh)")
    ax2.set_ylim(0, 0.15)
    # plt.title("Average daily grid offtake")
    plt.tight_layout()

    plt.figure(figsize=figsize)
    for ems_name in ems_names:
        yearly_mean = np.mean(grid_hist[ems_name], axis=0)

        x_axis_day = [
            datetime.datetime(start_year, 1, 1, 0, 0)
            + datetime.timedelta(minutes=i * 15)
            for i in range(24 * 4)
        ]
        daily_list = [[] for _ in x_axis_day]
        price_list = [[] for _ in x_axis_day]
        for i, time in enumerate(x_axis):
            day_axis_index = time.hour * 4 + time.minute // 15
            daily_list[day_axis_index].append(yearly_mean[i])
            price_list[day_axis_index].append(price_history[i])

        daily_mean = [np.mean(daily) for daily in daily_list]

        x_axis_str = [time.strftime("%H:%M") for time in x_axis_day]
        if ems_name == "RBC_1.5h":
            label = "RBC"
        elif ems_name == "MPC":
            label = "Model predictive control"
        elif ems_name == "TreeC":
            label = "TreeC"
        elif ems_name == "MPC_perfect":
            label = "MPC P"
        elif ems_name == "MPC_realistic":
            label = "Model predictive control naive forecast"
        elif ems_name == "MPC_realistic_forecast":
            label = "MPC"

        # plt.step(x_axis_day, daily_mean, label=label, alpha=1.0, where="post")
        x_axis_day_shifted = [
            time_day + datetime.timedelta(minutes=7, seconds=30)
            for time_day in x_axis_day
        ]
        daily_total = [mean_pow * 0.5 for mean_pow in daily_mean]
        plt.plot(
            x_axis_day_shifted,
            daily_total,
            label=label,
            alpha=ALPHA,
            marker="o",
            markersize=2,
        )

    # Show ticks every 4 hours with the format HH:MM
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=4))
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))

    # Put in slim vertical gird in the plot on the x tick marks
    # plt.gca().xaxis.grid(True)

    plt.ylabel("Total consumption of \n the 500 dwellings (MW)")
    plt.xlabel("Time of day CET")
    plt.ylim(0, 1)
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    plt.gca().set_axisbelow(True)
    plt.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)

    # plt.legend()
    ax3 = plt.twinx()
    price_mean = [np.mean(price) for price in price_list]

    ax3.step(
        x_axis_day,
        price_mean,
        color="black",
        alpha=0.5,
        where="post",
        label="Day-ahead",
    )
    ax3.set_ylabel("Mean price (€/kWh)")
    ax3.set_ylim(0, 0.15)
    # Put the yticks every 0.03 from 0 to 0.15
    ax3.yaxis.set_ticks(np.arange(0, 0.16, 0.03))
    # plt.title("Average daily grid offtake")

    lines2, labels2 = ax3.get_legend_handles_labels()

    # Combine legends from both axes
    lines = lines1 + lines2
    labels = labels1 + labels2
    if target == "thesis":
        plt.legend(lines, labels, loc="lower right")
    elif target == "paper":
        plt.legend(lines, labels, fontsize=14, ncol=5, loc="lower right")

    plt.tight_layout()
    plt.savefig(FIG_DIR + "average_daily_grid_power.pdf")


def calc_time_csv_reader():
    csv_file = "results/belgium_usefull_100_profiles/house_0.json_MPC_power.csv"

    start_time = datetime.datetime.now()
    output_dict = csv_to_dict(csv_file)
    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    print(f"Time taken to read {csv_file}: {time_taken.total_seconds()} seconds")

    start_time = datetime.datetime.now()
    output_dict = csv_to_dict_fast(csv_file)
    # print(output_dict["datetime"])
    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    print(
        f"Time taken to read {csv_file} with DictReader: {time_taken.total_seconds()} seconds"
    )

    start_time = datetime.datetime.now()
    output_dict = csv_to_dict_polars(csv_file)
    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    print(
        f"Time taken to read {csv_file} with Polars: {time_taken.total_seconds()} seconds"
    )


def get_charging_session_file(config_file, result_file):
    tot_houses = result_file.split("_")[2].replace(".csv", "")
    config_dir = f"data/houses_belgium_{tot_houses}/"
    config_path = config_dir + config_file
    config_house = json.load(open(config_path))
    environment = config_house["Environments"]["Environment_0"]

    csv_file = None
    for env_key in environment.keys():
        if "common/charging_sessions" in env_key:
            csv_file = env_key.split("/")[-1]

    return csv_file


def cummul_freq_soc(result_file, ems_names=["TreeC"]):

    start_dt = datetime.datetime(2023, 1, 1, 0, 0, 0)
    end_dt = datetime.datetime(2024, 1, 1, 0, 0, 0)
    all_session_soc_i = get_all_soc_init(start_dt, end_dt)

    json_file = result_file.replace(".csv", "_no_enforcement_soc.json")

    content = json.load(open(json_file))

    all_config_files = list(content.keys())
    charging_sessions_files = {
        config_file: get_charging_session_file(config_file, result_file)
        for config_file in all_config_files
    }

    interval = 0.1

    soc_eval = np.arange(0, 95 + interval, interval)

    all_soc = {ems_name: [] for ems_name in ems_names}
    all_soc["Arrival SOC"] = []
    for ems_name in all_soc.keys():
        ems_index = ems_name + "_no_enforcement"
        all_soc[ems_name] = {soc: [] for soc in soc_eval}
        for config_file, ems_dict in content.items():
            if ems_index not in ems_dict.keys() and ems_name != "Arrival SOC":
                continue
            if ems_name != "Arrival SOC":
                soc_diff_list = ems_dict[ems_index]
                soc_diff_list = [95 - soc_diff * 100 for soc_diff in soc_diff_list]
            else:
                if charging_sessions_files[config_file] is None:
                    continue
                soc_diff_list = all_session_soc_i[charging_sessions_files[config_file]][
                    "values"
                ]
                soc_diff_list = [soc_diff * 100 for soc_diff in soc_diff_list]
            # if len(soc_i) != len(soc_diff_list):
            #    print("Not normal ... ")
            if soc_diff_list == []:
                continue
            for soc in soc_eval:
                num_soc_under = np.sum(np.array(soc_diff_list) <= soc)

                soc_value = num_soc_under / len(soc_diff_list) * 100
                all_soc[ems_name][soc].append(soc_value)
    plt.figure()

    rbc_mpc_cumul = [0] * len(soc_eval)
    rbc_mpc_cumul[-1] = 100

    color_dict = {
        "TreeC": "tab:orange",
        "MPC_realistic_forecast": "tab:green",
        "Arrival SOC": "tab:gray",
    }
    linewidth_std = 0.8

    plt.plot([95], [100], label="Mean", color="black")
    plt.plot(
        [95],
        [100],
        label="10$^{th}$ and 90$^{th}$ percentiles",
        color="black",
        linestyle="--",
        linewidth=linewidth_std,
    )

    for ems_name in list(all_soc.keys())[::-1]:
        if ems_name in ["MPC_perfect"]:
            continue
        y = [np.mean(all_soc[ems_name][soc]) for soc in soc_eval]

        perc10 = [np.percentile(all_soc[ems_name][soc], 10) for soc in soc_eval]
        perc90 = [np.percentile(all_soc[ems_name][soc], 90) for soc in soc_eval]

        label = ems_ticks_label_dict.get(ems_name, ems_name)
        if label == "Arrival SOC":
            label = "Arrival SOC for all"
        else:
            label = f"Departure SOC for {label}"

        plt.plot(
            soc_eval,
            y,
            label=f"{label}",
            # linestyle=linestyle,
            color=color_dict.get(ems_name, "tab:blue"),
            linewidth=2,
        )
        plt.plot(
            soc_eval,
            perc10,
            linestyle="--",
            color=plt.gca().lines[-1].get_color(),
            alpha=ALPHA,
            # label=f"{label} 80% interpercentile range",
            linewidth=linewidth_std,
        )
        plt.plot(
            soc_eval,
            perc90,
            linestyle="--",
            color=plt.gca().lines[-1].get_color(),
            alpha=ALPHA,
            linewidth=linewidth_std,
        )
        # Add a flat line for "RBC and MPC perfect forecast" at y=0 for all quantiles
        # plt.xlim(-1, 101)
    plt.plot(
        soc_eval,
        rbc_mpc_cumul,
        label="Departure SOC for RBC and MPC P",
        color="tab:blue",
        linewidth=2,
    )

    # Put the x ticks every 5 in both directions
    plt.xticks(np.arange(5, 105, 10))
    plt.xlim(-1, 100)
    # plt.yticks(np.arange(0, 101, 5))
    plt.yticks(np.arange(0, 110, 10))
    plt.ylim(-5, 105)

    # plt.ylim(-5, 60)
    plt.xlabel("SOC (%)")
    plt.ylabel("Cumulative relative frequency (%)")
    plt.legend()
    plt.gca().set_axisbelow(True)
    plt.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR + "cummul_freq_soc.pdf")


def do_quantile_plot_soc(result_file, ems_names=["TreeC"], alt_metric=False):
    json_file = result_file.replace(".csv", "_no_enforcement_soc.json")

    content = json.load(open(json_file))
    interval = 0.005

    quant_evaluated = np.arange(0, 1 + interval, interval)
    all_quantiles = {ems_name: {} for ems_name in ems_names}
    for ems_name in ems_names:
        ems_index = ems_name + "_no_enforcement"
        all_quantiles[ems_name] = {quant: [] for quant in quant_evaluated}
        for config_file, ems_dict in content.items():
            if ems_index not in ems_dict.keys():
                continue
            soc_diff_list = ems_dict[ems_index]
            if not alt_metric:
                soc_diff_list = [soc_diff * 100 for soc_diff in soc_diff_list]
            else:
                soc_diff_list = [95 - soc_diff * 100 for soc_diff in soc_diff_list]
            if soc_diff_list == []:
                continue
            for quant in quant_evaluated:
                quant_value = np.quantile(soc_diff_list, quant, method="hazen")
                all_quantiles[ems_name][quant].append(quant_value)

    plt.figure()

    x = [quant * 100 for quant in quant_evaluated]
    if not alt_metric:
        plt.plot(x, [0] * len(x), label="RBC and MPC P")
    else:
        x = x[::-1]
        plt.plot(x, [95] * len(x), label="RBC and MPC P")

    for ems_name in ems_names:

        y = [np.mean(all_quantiles[ems_name][quant]) for quant in quant_evaluated]
        yerr = [np.std(all_quantiles[ems_name][quant]) for quant in quant_evaluated]

        label = ems_ticks_label_dict.get(ems_name, ems_name)

        plt.plot(x, y, label=f"{label} mean", linewidth=2)
        plt.plot(
            x,
            np.array(y) + np.array(yerr),
            linestyle="--",
            color=plt.gca().lines[-1].get_color(),
            alpha=ALPHA,
            label=f"{label} standard deviation",
        )
        plt.plot(
            x,
            np.array(y) - np.array(yerr),
            linestyle="--",
            color=plt.gca().lines[-1].get_color(),
            alpha=ALPHA,
        )
        # Add a flat line for "RBC and MPC perfect forecast" at y=0 for all quantiles
        plt.xlim(40, 105)

    plt.xlabel("Quantile (%)")
    if not alt_metric:
        plt.ylabel("SOC difference with max charging (%)")
    else:
        plt.ylabel("SOC at end of charging session (%)")
    plt.legend()
    plt.gca().set_axisbelow(True)
    plt.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    plt.savefig(FIG_DIR + "quantile_soc_diff.pdf")


class WhiskerHandler(HandlerBase):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        # Create vertical line
        line1 = Line2D(
            [width / 2, width / 2],
            [height * 0.2, height * 0.8],
            color="black",
            linewidth=1.5,
            transform=trans,
        )
        # Create top horizontal line
        line2 = Line2D(
            [width * 0.3, width * 0.7],
            [height * 0.8, height * 0.8],
            color="black",
            linewidth=1.5,
            transform=trans,
        )
        return [line1, line2]


def get_all_soc_init(start_dt, end_dt):
    session_dir = "data/common/charging_sessions/"
    all_session_soc_i = dict()

    for file in os.listdir(session_dir):
        csv_path = session_dir + file
        df = pl.read_csv(csv_path, schema_overrides={"type_0 (/)": str})

        # Convert datetime column to datetime type
        df = df.with_columns(
            pl.col("").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ").alias("datetime")
        )
        # Filter the dataframe for the given date range
        df_filtered = df.filter(
            (pl.col("datetime") >= start_dt) & (pl.col("datetime") <= end_dt)
        )

        soc_filtered = df_filtered.filter(df_filtered["soc_i_0 (/)"] != 0.0)
        # Only keep values that are not 0.0
        soc_i_col = soc_filtered["soc_i_0 (/)"]
        datetime_filt_col = soc_filtered["datetime"]
        all_session_soc_i[file] = {}
        all_session_soc_i[file]["values"] = soc_i_col.to_list()
        all_session_soc_i[file]["datetime"] = datetime_filt_col.to_list()

    return all_session_soc_i


def do_histogram_plot_soc(result_file, ems_names=["TreeC"], alt_metric=False):

    bins = np.array([0, 0.0000001, 2, 5, 10, 25, 50, 85])
    if alt_metric:
        bins = np.array([95 - b for b in bins][::-1])

    json_file = result_file.replace(".csv", "_no_enforcement_soc.json")

    content = json.load(open(json_file))
    all_histograms = {ems_name: [] for ems_name in ems_names}
    max_soc_diff = 0
    for ems_name in ems_names:
        ems_index = ems_name + "_no_enforcement"
        for config_file, ems_dict in content.items():
            if ems_index not in ems_dict.keys():
                continue
            soc_diff_list = ems_dict[ems_index]
            if not alt_metric:
                soc_diff_list = [soc_diff * 100 for soc_diff in soc_diff_list]
            else:
                soc_diff_list = [95 - soc_diff * 100 for soc_diff in soc_diff_list]

            if not soc_diff_list:
                continue
            max_soc_diff = max(max_soc_diff, max(soc_diff_list))

            hist, _ = np.histogram(soc_diff_list, bins=bins, density=False)
            hist = hist / sum(hist)  # Normalize the histogram
            all_histograms[ems_name].append(hist)

    print(max_soc_diff)

    plt.figure(figsize=(8, 4.8))
    num_bins = len(bins) - 1
    num_ems = len(ems_names)
    width = (
        0.8 / num_ems
    )  # fixed width for each bar, so bars are always next to each other

    # Create bin range labels
    bin_labels = [f"{bins[i]:.3g}%-{bins[i+1]:.3g}%" for i in range(num_bins)]
    if alt_metric:
        bin_labels[-1] = "fully charged\n(95%)"
    else:
        bin_labels[0] = "max charge"
        bin_labels[1] = f"0%-{bins[2]:.3g}%"
    x = np.arange(num_bins)  # one x position per bin
    x_pos = list(x)

    if alt_metric:
        x_pos[-1] += width

    combined_label = "RBC and MPC P"

    combined_offset = -(len(ems_names) + 1) / 2 * width  # next to last bar
    combined_heights = [0] * num_bins
    combined_heights[0] = 100  # 100% probability in first bin
    if not alt_metric:
        x_pos_rbc = x_pos[0] + combined_offset
    else:
        x_pos_rbc = x_pos[-1] + combined_offset
    plt.bar(
        x_pos_rbc,
        combined_heights[0],
        width=width,
        label=combined_label,
        # color="gray",
        alpha=ALPHA,
        align="center",
    )

    for idx, ems_name in enumerate(ems_names):
        hist_array = np.array(all_histograms[ems_name])
        mean_hist = np.mean(hist_array, axis=0) * 100
        std_hist = np.std(hist_array, axis=0) * 100
        # Offset bars for each ems_name within each bin
        offset = (idx - (num_ems - 1) / 2) * width
        new_x_pos = [x_pos[i] + offset for i in range(num_bins)]

        label = ems_ticks_label_dict.get(ems_name, ems_name)

        plt.bar(
            new_x_pos,
            mean_hist,
            width=width,
            label=label,
            yerr=std_hist,
            capsize=3,
            align="center",
            alpha=ALPHA,
            error_kw=dict(ecolor="black", lw=1, capsize=3),
        )

    x_ticks = x_pos.copy()
    if not alt_metric:
        x_ticks[0] -= width / 2
    else:
        x_ticks[-1] -= width / 2

    plt.xticks(
        x_ticks,
        bin_labels,
        rotation=0,
        ha="center",
    )
    if not alt_metric:
        plt.xlabel("SOC difference with max charging (%)")
    else:
        plt.xlabel("SOC at end of charging session")
    plt.ylabel("Frequency (%)")
    # plt.title("Average histogram of SOC difference (with error bars)")
    # Create custom legend with T-shaped whisker for error bars

    handles, labels = plt.gca().get_legend_handles_labels()

    # Add the whisker to the legend
    whisker_proxy = Line2D(
        [0], [0], color="black", linewidth=1.5, label="Standard deviation"
    )
    handles.append(whisker_proxy)

    plt.legend(
        handles=handles, handler_map={whisker_proxy: WhiskerHandler()}, loc="best"
    )
    plt.gca().set_axisbelow(True)
    plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    plt.savefig(FIG_DIR + "histogram_soc_diff.pdf")


def plot_all_paper_plots():
    tot_houses = 500
    result_file = f"results/belgium_usefull_{tot_houses}.csv"
    config_dir = f"data/houses_belgium_{tot_houses}/"

    ems_names = ["RBC_1.5h", "TreeC", "MPC_realistic_forecast", "MPC_perfect"]
    total_plot(result_file, config_dir, ems_names)

    plot_average_daily_offtake(result_file, ems_names=ems_names)
    ems_names = ["TreeC", "MPC_realistic_forecast", "MPC_perfect"]
    plot_opex_diff_separate(result_file, ems_names=ems_names)

    all_ems_plot(
        result_file,
        config_dir,
        ems_names=ems_names,
        y_axis="simple_opex_diff (€)",
        show_num_points=False,
    )

    cummul_freq_soc(result_file, ["TreeC", "MPC_realistic_forecast"])


def get_proportions_power_asset(ems_name, profile_dir):

    all_power = {
        "Consumer_0": 0,
        "PublicGrid_0": 0,
        "WaterHeater_0": 0,
        "Charger_0": 0,
        "SolarPv_0": 0,
        "EnergyPlus_0": 0,
    }
    for filename in os.listdir(profile_dir):
        if ems_name in filename and filename.endswith("_power.csv"):
            filepath = os.path.join(profile_dir, filename)
            power_dict = csv_to_dict(filepath)
            for key in all_power.keys():
                if key in power_dict:
                    all_power[key] += sum(power_dict[key])

    print(all_power)


if __name__ == "__main__":
    tot_houses = 500
    result_file = f"results/belgium_usefull_{tot_houses}.csv"
    config_dir = f"data/houses_belgium_{tot_houses}/"

    ems_names = ["RBC_1.5h", "TreeC", "MPC_realistic_forecast", "MPC_perfect"]
    total_plot(result_file, config_dir, ems_names)
    # plot_average_daily_offtake(result_file, ems_names=ems_names)

    # ems_names = ["TreeC", "MPC_realistic_forecast", "MPC_perfect"]
    # plot_opex_diff_separate(result_file, ems_names=ems_names)

    ems_names = ["TreeC", "MPC_realistic_forecast", "MPC_perfect"]

    # all_ems_plot(
    #    result_file,
    #    config_dir,
    #    ems_names=ems_names,
    #    y_axis="simple_opex_diff (€)",
    #    show_num_points=False,
    # )
    # plot_all_paper_plots()
    plt.show()
