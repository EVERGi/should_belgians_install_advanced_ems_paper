from simugrid.misc.log_plot_micro import (
    plot_power_hist,
    get_power_hist_data_dict,
    read_csv_to_dict,
    plot_attributes,
)
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import DateFormatter
import numpy as np
import polars as pl
from result_plots import (
    FIG_DIR,
    ALPHA,
    get_all_soc_init,
    get_charging_session_file,
    target,
)
import pytz


if target == "thesis":
    plt.rcParams.update(
        {
            "font.size": 11,  # Default text size
            "axes.titlesize": 13,  # Axes title size
            "axes.labelsize": 11,  # Axes label size
            "xtick.labelsize": 10,  # X tick label size
            "ytick.labelsize": 10,  # Y tick label size
            "legend.fontsize": 10,  # Legend font size
        }
    )
    figsize = (8, 10)
elif target == "paper":
    figsize = (15, 8)
    plt.rcParams.update(
        {
            "font.size": 11,  # Default text size
            "axes.titlesize": 13,  # Axes title size
            "axes.labelsize": 12,  # Axes label size
            "xtick.labelsize": 12,  # X tick label size
            "ytick.labelsize": 12,  # Y tick label size
            "legend.fontsize": 11,  # Legend font size
        }
    )


def plot_week_control():
    ems_to_plot = {
        "RBC": "RBC_1.5h",
        "TreeC": "TreeC_no_enforcement",
        "MPC": "MPC_realistic_forecast_no_enforcement",
        "MPC P": "MPC_perfect",
    }
    ems_color_dict = {
        "RBC": "tab:blue",
        "TreeC": "tab:orange",
        "MPC": "tab:green",
        "MPC P": "tab:red",
    }
    house_num = 1
    house_json = f"house_{house_num}.json"
    start_dt = datetime.datetime(2023, 3, 18)
    end_dt = datetime.datetime(2023, 3, 20)

    tz_bru = pytz.timezone("Europe/Brussels")
    start_dt_bru = tz_bru.localize(start_dt)
    end_dt_bru = tz_bru.localize(end_dt)
    start_dt_utc = start_dt_bru.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    end_dt_utc = end_dt_bru.astimezone(datetime.timezone.utc).replace(tzinfo=None)

    fig, ax = plt.subplots(
        8, 1, figsize=figsize, sharex=True, height_ratios=[2, 2, 2, 2, 2, 2, 2, 2]
    )
    # x = np.arange(0, 10, 2)
    # y = np.vstack([ay, by, cy])

    for i, ems_name in enumerate(ems_to_plot.keys()):
        result_name = ems_to_plot[ems_name]
        # Plot power
        power_filepath = get_power_filepath(house_num, result_name)

        plot_example_day(
            power_filepath, start_dt_utc, end_dt_utc, ax=ax[i], ems_name=ems_name
        )

        ems_color = ems_color_dict[ems_name]
        attributes_filepath = power_filepath.replace("_power.csv", "_attributes.csv")

        plot_soc(
            attributes_filepath, start_dt_utc, end_dt_utc, ax[-3], ems_color, ems_name
        )
        plot_tank_temp(
            attributes_filepath, start_dt_utc, end_dt_utc, ax[-2], ems_color, ems_name
        )
        plot_indoor_temp(
            attributes_filepath, start_dt_utc, end_dt_utc, ax[-1], ems_color, ems_name
        )

    plot_day_ahead_price(start_dt_utc, end_dt_utc, ax[-4])

    for i in range(4):
        pass
        ax[i].set_ylim(-5, 10)

    ax[-3].set_ylim(0, 100)
    # ax[-2].set_ylim(0, 100)
    ax[-1].set_ylim(15, 25)

    if target == "thesis":
        ax[0].legend(ncol=3, loc="upper right")

        ax[-3].legend(ncol=2, loc="upper right")
        ax[-1].legend(loc="upper left")
    elif target == "paper":
        ax[0].legend(ncol=6, loc="upper right")
        ax[-4].legend(loc="lower right")
        ax[-3].legend(ncol=4, loc="upper right")
        ax[-1].legend(loc="upper right")

    if target == "thesis":
        for i in range(len(ax)):
            ax[i].yaxis.set_label_coords(-0.045, 0.5)
        ax[-4].yaxis.set_label_coords(-0.07, 0.5)
    elif target == "paper":
        ax[-4].yaxis.set_label_coords(-0.026, 0.5)
        ax[-3].yaxis.set_label_coords(-0.026, 0.5)
    for i in range(len(ax)):
        ax[i].set_axisbelow(True)
        ax[i].grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)

    # ax[-3].yaxis.set_label_coords(-0.07, 0.5)
    # ax[-2].yaxis.set_label_coords(-0.07, 0.5)

    min_y = 0
    max_y = 0

    for i, _ in enumerate(ems_to_plot.keys()):
        # Calculate the ylim for each subplot
        ylim = ax[i].get_ylim()
        min_y = min(min_y, ylim[0])
        max_y = max(max_y, ylim[1])

    for i, _ in enumerate(ems_to_plot.keys()):
        # Set the same ylim for each subplot
        ax[i].set_ylim(min_y, max_y)

    plt.xlim(start_dt_utc, end_dt_utc)
    formatter = DateFormatter("%d/%m %H:%M", tz=pytz.timezone("Europe/Brussels"))
    ax[-1].xaxis.set_major_formatter(formatter)
    # Only show the ticks at datetime(2023, 3, 17,23,0) and datetime(2023, 3, 19, 23, 0)
    ax[-1].set_xticks(
        [start_dt_utc + datetime.timedelta(hours=12) * i for i in range(5)]
    )
    plt.xlabel("Time of day CET")

    plt.tight_layout()
    plt.savefig(
        f"{FIG_DIR}/example_two_days_house_{house_num}.pdf",
        dpi=300,
    )


def get_power_filepath(house_num, ems_name):
    house_json = f"house_{house_num}.json"
    if "no_enforcement" in ems_name:
        profile_dir = "results/belgium_usefull_500_no_enforcement_profiles/"
    else:
        profile_dir = "results/belgium_usefull_500_profiles/"

    power_filepath = f"{profile_dir}{house_json}_{ems_name}_power.csv"
    return power_filepath


def plot_example_day(power_filepath, start_dt, end_dt, ax, ems_name):

    # csv_file = f"data/example_day/RBC_house3_5s_30th_may.csv"

    # Interpolate only column grid of the dataframe
    # df["grid"] = -df["grid"].interpolate()

    # average_every = 24

    # df_grid = df["grid"]
    # df_grid = df_grid.rolling(average_every).mean()
    # df_grid = df_grid / 1000

    # Load column only negative
    # df["load"] = df["load"].clip(upper=0)

    # df.plot(x="time", y="grid")
    # Convert start_dt and end_dt to Europe/Brussels timezone

    cols_to_stack = [
        "SolarPv_0",
        "Consumer_0",
        "EnergyPlus_0",
        "WaterHeater_0",
        "Charger_0",
    ]

    df = pl.read_csv(power_filepath)

    # In polars, put the datetime column as index
    df = df.with_columns(
        pl.col("datetime").str.strptime(pl.Datetime, "%d/%m/%Y %H:%M:%S")
    )
    df = df.filter((pl.col("datetime") >= start_dt) & (pl.col("datetime") <= end_dt))

    df_plot = df

    # Filter from start_dt to end_dt

    # Average on 10 seconds instead of 5

    # w_minvar1nc_pos = df_plot.clone()
    w_minvar1nc_pos = df_plot.clone().select(cols_to_stack)
    w_minvar1nc_pos = w_minvar1nc_pos.with_columns(
        [
            pl.when(pl.col(col) >= 0).then(-pl.col(col)).otherwise(0).alias(col)
            for col in cols_to_stack
        ]
    )
    w_minvar1nc_neg = df_plot.clone().select(cols_to_stack)
    w_minvar1nc_neg = w_minvar1nc_neg.with_columns(
        [
            pl.when(pl.col(col) < 0).then(-pl.col(col)).otherwise(0).alias(col)
            for col in cols_to_stack
        ]
    )

    # initialize stackplot
    # fig, ax = plt.subplots(figsize=(12, 5))
    # Put two plots one on top of the other

    # create and format stackplot
    pos_vstack = np.vstack([w_minvar1nc_pos[col] for col in w_minvar1nc_pos.columns])

    # Propose me a list of 4 colors that represent solar, electrical load, battery and electric vehicle charger
    # Stay in the matplotlib theme
    colors_dict = {
        "SolarPv_0": "#ffd700",  # Gold for Solar PV
        "Consumer_0": "#d480ff",  # Blue for Electrical load
        "EnergyPlus_0": "#FF0000",  # Red for BESS
        "WaterHeater_0": "#87CEEB",  # Green for EV charger
        "Charger_0": "#2E8B57",  # Orange for EV charger
        "Grid": "grey",  # Grey for Grid
    }
    colors = [colors_dict[col] for col in colors_dict.keys()]

    labels_dict = {
        "SolarPv_0": "Solar PV",
        "Consumer_0": "Base load",
        "EnergyPlus_0": "Heat pump",
        "WaterHeater_0": "Water heater",
        "Charger_0": "EV charger",
        "Grid": "Grid",
    }
    labels = [labels_dict[col] for col in colors_dict.keys()]

    index = [
        start_dt + datetime.timedelta(minutes=15) * i for i in range(df_plot.shape[0])
    ]

    ax.stackplot(
        index, pos_vstack, labels=labels, colors=colors, step="post", alpha=ALPHA
    )

    neg_vstack = np.vstack([w_minvar1nc_neg[col] for col in w_minvar1nc_neg.columns])
    ax.stackplot(index, neg_vstack, colors=colors, step="post", alpha=ALPHA)

    # Plot the grid as a line on top
    # Propose me a color that represents the grid
    grid_color = "black"
    linewidth = 1

    ax.step(
        index,
        df["PublicGrid_0"],
        label="Grid",
        color=grid_color,
        linewidth=linewidth,
        where="post",
    )
    if target == "thesis":
        ax.set_ylabel(f"{ems_name}\npower (kW)")
    elif target == "paper":
        ax.set_ylabel(f"{ems_name}\npower\n(kW)")


def plot_soc(attributes_filepath, start_dt, end_dt, ax, color, ems_name):

    tot_houses = int(attributes_filepath.split("/")[1].split("_")[2])

    config_file = attributes_filepath.split("/")[-1].split(".json")[0] + ".json"
    # config_file = f"data/houses_belgium_{tot_houses}/{house_json}"
    all_soc_i = get_all_soc_init(start_dt, end_dt)
    result_file = f"belgium_usefull_{tot_houses}.csv"
    charging_session_file = get_charging_session_file(config_file, result_file)
    cur_soc_i = all_soc_i[charging_session_file]

    df = pl.read_csv(attributes_filepath)
    df = df.with_columns(
        pl.col("datetime").str.strptime(pl.Datetime, "%d/%m/%Y %H:%M:%S")
    )
    df = df.filter((pl.col("datetime") >= start_dt) & (pl.col("datetime") <= end_dt))

    # Add rows to the df
    dict_to_concat = {
        "datetime": [],
        "EV SOC": [],
    }
    for i, soc in enumerate(cur_soc_i["values"]):
        dt_soc = cur_soc_i["datetime"][i] + datetime.timedelta(seconds=1)
        dict_to_concat["datetime"].append(dt_soc)
        dict_to_concat["EV SOC"].append(soc)
    to_concat_df = pl.DataFrame(dict_to_concat)

    soc_df = df.select(
        pl.col("datetime"),
        pl.col("Charger_0_soc").alias("EV SOC"),
    )

    # Concatenate the two dataframes
    soc_df = pl.concat([soc_df, to_concat_df], how="vertical")
    soc_df = soc_df.sort("datetime")

    soc_df = soc_df.with_columns((pl.col("EV SOC") * 100.0).alias("EV SOC"))

    # Replace 0 by NaN
    soc_df = soc_df.with_columns(
        pl.when(pl.col("EV SOC") == 0.0)
        .then(None)
        .otherwise(pl.col("EV SOC"))
        .alias("EV SOC")
    )

    ax.plot(
        soc_df["datetime"],
        soc_df["EV SOC"],
        label=ems_name,
        color=color,
        linewidth=1,
        alpha=ALPHA,
    )
    if target == "thesis":
        ax.set_ylabel("EV SOC\n(%)")
    elif target == "paper":
        ax.set_ylabel("EV SOC\n(%)\n")


def plot_tank_temp(attributes_filepath, start_dt, end_dt, ax, color, ems_name):
    df = pl.read_csv(attributes_filepath)
    df = df.with_columns(
        pl.col("datetime").str.strptime(pl.Datetime, "%d/%m/%Y %H:%M:%S")
    )
    df = df.filter((pl.col("datetime") >= start_dt) & (pl.col("datetime") <= end_dt))
    df = df.select(
        pl.col("datetime"),
        pl.col("WaterHeater_0_t_tank").alias("Tank Temp"),
    )

    ax.plot(
        df["datetime"],
        df["Tank Temp"],
        label="Tank Temp",
        color=color,
        linewidth=1,
        alpha=ALPHA,
    )

    if target == "thesis":
        ax.set_ylabel("Tank temp\n(°C)")
    elif target == "paper":
        ax.set_ylabel("Tank\ntemp\n(°C)")


def plot_indoor_temp(attributes_filepath, start_dt, end_dt, ax, color, ems_name):
    df = pl.read_csv(attributes_filepath)
    df = df.with_columns(
        pl.col("datetime").str.strptime(pl.Datetime, "%d/%m/%Y %H:%M:%S")
    )
    df = df.filter((pl.col("datetime") >= start_dt) & (pl.col("datetime") <= end_dt))
    df = df.select(
        pl.col("datetime"),
        pl.col("EnergyPlus_0_zn0_temp").alias("Indoor Temp"),
        pl.col("EnergyPlus_0_cur_t_low").alias("Indoor Temp Low bound"),
        pl.col("EnergyPlus_0_cur_t_up").alias("Indoor Temp Upper bound"),
    )

    # If Indoor temp low boun equals 5, replace it with NaN
    df = df.with_columns(
        pl.when(pl.col("Indoor Temp Low bound") == 5.0)
        .then(None)
        .otherwise(pl.col("Indoor Temp Low bound"))
        .alias("Indoor Temp Low bound")
    )
    df = df.with_columns(
        pl.when(pl.col("Indoor Temp Upper bound") == 40.0)
        .then(None)
        .otherwise(pl.col("Indoor Temp Upper bound"))
        .alias("Indoor Temp Upper bound")
    )

    linewidth_bounds = 0.8
    if ems_name == "RBC":
        ax.step(
            df["datetime"],
            df["Indoor Temp Low bound"],
            color="gray",
            linewidth=linewidth_bounds,
            where="pre",
            label="Comfort range",
        )
        ax.step(
            df["datetime"],
            df["Indoor Temp Upper bound"],
            color="gray",
            linewidth=linewidth_bounds,
            where="pre",
        )

    ax.plot(df["datetime"], df["Indoor Temp"], color=color, linewidth=1, alpha=ALPHA)

    if target == "thesis":
        ax.set_ylabel("Indoor temp\n(°C)")
    elif target == "paper":
        ax.set_ylabel("Indoor\ntemp\n(°C)")


def plot_day_ahead_price(start_dt, end_dt, ax):
    day_ahead_price_filepath = "data/common/day_ahead_price/day_ahead_22_23.csv"
    df = pl.read_csv(day_ahead_price_filepath)
    df = df.with_columns(
        pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ")
    )
    df = df.filter((pl.col("datetime") >= start_dt) & (pl.col("datetime") <= end_dt))
    df = df.select(
        pl.col("datetime"),
        pl.col("price (€/kWh)").alias("Day Ahead Price"),
    )

    ax.step(
        df["datetime"],
        df["Day Ahead Price"],
        label="Day-ahead",
        color="black",
        linewidth=1,
        where="post",
        # label="Day-ahead",
    )
    if target == "thesis":
        ax.set_ylabel("Price (€/kWh)")
    elif target == "paper":
        ax.set_ylabel("Price\n(€/kWh)\n")
    ax.xaxis.set_major_formatter(DateFormatter("%d/%m"))


if __name__ == "__main__":
    plot_week_control()
    plt.show()
