import polars as pl

import matplotlib.pyplot as plt
import numpy as np

import os

from belgian_dwellings.simulation.calibrate_mpc import (
    execute_config,
    calc_efficiency,
    calc_therma_properties,
    calibration_func,
)


def read_epw_file(file_path):
    # Read the EPW file, skipping the first 8 header lines
    df = pl.read_csv(
        file_path,
        skip_rows=8,
        has_header=False,
        new_columns=[
            "Year",
            "Month",
            "Day",
            "Hour",
            "Minute",
            "Data Source and Uncertainty Flags",
            "Dry Bulb Temperature (C)",
            "Dew Point Temperature (C)",
            "Relative Humidity (%)",
            "Atmospheric Station Pressure (Pa)",
            "Extraterrestrial Horizontal Radiation (Wh/m2)",
            "Extraterrestrial Direct Normal Radiation (Wh/m2)",
            "Horizontal Infrared Radiation Intensity (Wh/m2)",
            "Global Horizontal Radiation (Wh/m2)",
            "Direct Normal Radiation (Wh/m2)",
            "Diffuse Horizontal Radiation (Wh/m2)",
            "Global Horizontal Illuminance (lux)",
            "Direct Normal Illuminance (lux)",
            "Diffuse Horizontal Illuminance (lux)",
            "Zenith Luminance (cd/m2)",
            "Wind Direction (degrees)",
            "Wind Speed (m/s)",
            "Total Sky Cover (tenths)",
            "Opaque Sky Cover (tenths)",
            "Visibility (km)",
            "Ceiling Height (m)",
            "Present Weather Observation",
            "Present Weather Codes",
            "Precipitable Water (cm)",
            "Aerosol Optical Depth (thousandths)",
            "Snow Depth (cm)",
            "Days Since Last Snowfall",
            "Albedo",
            "Liquid Precipitation Depth (mm)",
            "Liquid Precipitation Quantity (hr)",
        ],
    )

    return df


def epw_df_to_datetime_and_temp(df, local_timezone="Europe/Brussels"):
    # Combine Year, Month, Day, Hour, and Minute into a single datetime column
    df = df.with_columns(
        pl.datetime(
            year=pl.col("Year"),
            month=pl.col("Month"),
            day=pl.col("Day"),
            hour=pl.col("Hour") - 1,  # Adjust hour to be zero-based
            minute=pl.col("Minute"),
        ).alias("datetime")
    )
    # Convert to local timezone
    df = df.with_columns(
        pl.col("datetime")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(local_timezone)
        .alias("datetime")
    )

    # Select only the datetime and Dry Bulb Temperature columns
    df = df.select(["datetime", "Dry Bulb Temperature (C)"])

    return df


def get_max_min_daily_temps(df):
    # Resample to daily frequency and get max and min temperatures
    daily_max = df.group_by(pl.col("datetime").dt.date()).agg(
        pl.col("Dry Bulb Temperature (C)").max().alias("Max Temp (C)")
    )
    daily_min = df.group_by(pl.col("datetime").dt.date()).agg(
        pl.col("Dry Bulb Temperature (C)").min().alias("Min Temp (C)")
    )

    # Join max and min dataframes on the date
    daily_temps = daily_max.join(daily_min, on="datetime")

    # Sort by date
    daily_temps = daily_temps.sort("datetime")

    return daily_temps


def add_average_daily_min_max(daily_temps):
    # Add an extra column with the average of daily min and max temperatures
    daily_temps = daily_temps.with_columns(
        ((pl.col("Max Temp (C)") + pl.col("Min Temp (C)")) / 2).alias("Avg Temp (C)")
    )
    return daily_temps


def calc_approx_and_real_t_ref(daily_temps):
    # Approximate t_ref is (t_days[-1] + 0.8 * t_days[-1] + 0.4 * t_days[-2] + 0.2 * t_days[-3]) / 2.4
    # Real t_ref is (t_days[0] + 0.8 * t_days[-1] + 0.4 * t_days[-2] + 0.2 * t_days[-3]) / 2.4
    # Where t_days is the average daily temperature of the previous 4 days
    # Add an extra column with the approximate and real t_ref values\
    # This calculation should be done for each day, starting from the 4th day
    daily_temps = daily_temps.with_columns(
        (
            (
                pl.col("Avg Temp (C)").shift(0)
                + 0.8 * pl.col("Avg Temp (C)").shift(1)
                + 0.4 * pl.col("Avg Temp (C)").shift(2)
                + 0.2 * pl.col("Avg Temp (C)").shift(3)
            )
            / 2.4
        ).alias("Real t_ref (C)")
    )
    daily_temps = daily_temps.with_columns(
        (
            (
                pl.col("Avg Temp (C)").shift(1)
                + 0.8 * pl.col("Avg Temp (C)").shift(1)
                + 0.4 * pl.col("Avg Temp (C)").shift(2)
                + 0.2 * pl.col("Avg Temp (C)").shift(3)
            )
            / 2.4
        ).alias("Approx t_ref (C)")
    )

    return daily_temps


def get_comfort_ranges(daily_temps, status):
    # Add columns with comfort ranges based on t_ref and status
    daily_temps = daily_temps.with_columns(
        pl.struct(["Real t_ref (C)"])
        .map_elements(lambda x: calc_t_low(x["Real t_ref (C)"], status))
        .alias(f"Real t_low {status} (C)")
    )
    daily_temps = daily_temps.with_columns(
        pl.struct(["Approx t_ref (C)"])
        .map_elements(lambda x: calc_t_low(x["Approx t_ref (C)"], status))
        .alias(f"Approx t_low {status} (C)")
    )

    return daily_temps


def calc_t_low(t_ref, status):
    comfort_range = calc_comfort_range(t_ref, status)
    return comfort_range[0]


def calc_t_up(t_ref, status):
    comfort_range = calc_comfort_range(t_ref, status)
    return comfort_range[1]


def calc_comfort_range(t_ref, status):
    # W and Alpha values for 10% predicted percentage of dissatisfied
    # Source: https://doi.org/10.1016/j.apenergy.2008.07.011
    W = 5
    ALPHA = 0.7

    if t_ref is None:
        return [None, None]

    if status == "at home":
        if t_ref < 12.5:
            t_n = 20.4 + 0.06 * t_ref
        else:
            t_n = 16.63 + 0.36 * t_ref

        t_up = t_n + ALPHA * W
        t_low = max(t_n - (1 - ALPHA) * W, 18)
    elif status == "sleeping":
        if t_ref < 0:
            t_n = 16
        elif 0 <= t_ref < 12.6:
            t_n = 16 + 0.23 * t_ref
        elif 12.6 <= t_ref < 21.8:
            t_n = 9.18 + 0.77 * t_ref
        else:
            t_n = 26

        t_up = min(t_n + ALPHA * W, 26)
        t_low = max(t_n - (1 - ALPHA) * W, 16)
    elif status == "absent":
        t_low = 5
        t_up = 40

    return [t_low, t_up]


def calc_error_stats_in_comfort_range(daily_temps, statuses, plot_histogram=False):
    errors = {}
    for status in statuses:
        real_low_col = f"Real t_low {status} (C)"
        approx_low_col = f"Approx t_low {status} (C)"

        # Calculate error statistics between real and approximate t_low
        errors[status] = abs(daily_temps[real_low_col] - daily_temps[approx_low_col])
        if plot_histogram:
            # Plot histogram of errors
            plt.figure(figsize=(10, 6))
            plt.hist(
                errors[status].to_numpy(), bins=20, color="skyblue", edgecolor="black"
            )
            plt.title(f"Histogram of Absolute Errors in t_low for {status}")
            plt.xlabel("Absolute Error (C)")
            plt.ylabel("Frequency")
            plt.grid(axis="y", alpha=0.75)
            plt.show()
    all_errors = []
    for status in statuses:
        all_errors.extend(errors[status].to_list())
    all_errors = [e for e in all_errors if e is not None]
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)

    return {"mean_error": mean_error, "std_error": std_error}


def calc_comfort_stats():
    epw_path = "data/common/energyplus_models/brussels_2023.epw"
    df = read_epw_file(epw_path)
    df = epw_df_to_datetime_and_temp(df)
    daily_temps = get_max_min_daily_temps(df)
    daily_temps = add_average_daily_min_max(daily_temps)
    print(daily_temps)

    daily_temps = calc_approx_and_real_t_ref(daily_temps)
    print(daily_temps)
    statuses = ["at home", "sleeping"]
    for status in statuses:
        daily_temps = get_comfort_ranges(daily_temps, status)
        # print(daily_temps)
    stats = calc_error_stats_in_comfort_range(daily_temps, statuses)
    print(f"Error statistics for {status}: {stats}")


def get_custom_energyplus_calibration(config_file, plot=False):
    energyplus_calibrations = []

    calibration_list, ev_forecasting_values = execute_config(config_file)

    errors, naive_errors = None, None
    for calibration_values in calibration_list:
        previous_temp = np.array(calibration_values["indoor_temp_C"])[:-1]
        thermal_power = (
            np.array(calibration_values["heat_energy_main_kWh"])
            + np.array(calibration_values["heat_energy_backup_kWh"])
            - np.array(calibration_values["cool_energy_kWh"])
        )[1:]
        dict_prop = dict()
        solar_irradiation = np.array(calibration_values["dir_norm_rad"])[:-1]
        outdoor_temp = np.array(calibration_values["temp_air"])[:-1]
        dt = calibration_values["datetime"][:-1]
        dt = pl.Series(dt).to_list()

        # Get the time step
        dt_s_diff = (dt[1] - dt[0]).total_seconds()
        dt_s = np.array([dt_s_diff] * len(dt))
        # To Watts
        thermal_power = thermal_power * 1000 / (dt_s_diff / 3600)

        x = np.array(
            [previous_temp, thermal_power, solar_irradiation, outdoor_temp, dt_s]
        )

        y = np.array([previous_temp[1:]])

        # Add 1 temperature to y
        y = np.append(y, previous_temp[-1])

        # dict_eff = calc_efficiency(calibration_values)

        dict_prop = calc_therma_properties(calibration_values, plot=plot)

        diff = y - calibration_func(
            x,
            dict_prop["ga"],
            dict_prop["therm_cap"],
            dict_prop["therm_res"],
        )
        errors = np.abs(diff)
        error = np.mean(errors)
        std_error = np.std(errors)

        naive_errors = np.abs(previous_temp[1:] - previous_temp[:-1])
        # print(f"MAE thermal model: {error}")
        # print(f"STD thermal model: {std_error}")

    return errors, naive_errors


def execute_calibration():
    all_errors = []
    all_naive_errors = []
    for i in range(500):
        print(f"Calibrating house {i}")
        config_path = f"data/houses_belgium_500/house_{i}.json"
        errors, naive_errors = get_custom_energyplus_calibration(config_path)
        if errors is None:
            continue
        all_errors.extend(errors)
        all_naive_errors.extend(naive_errors)
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    mean_naive_error = np.mean(all_naive_errors)
    std_naive_error = np.std(all_naive_errors)
    print(f"Overall MAE thermal model: {mean_error}")
    print(f"Overall STD thermal model: {std_error}")
    print(f"Overall MAE naive model: {mean_naive_error}")
    print(f"Overall STD naive model: {std_naive_error}")


def get_average_daily_load():
    load_folder = "data/common/consumption/"
    all_consumption = []
    for file in os.listdir(load_folder):
        print(file)
        file_path = os.path.join(load_folder, file)
        pl.read_csv(file_path)
        df = pl.read_csv(file_path)

        # Only keep loads from 2023
        df = df.filter(pl.col("datetime").str.contains("2023"))
        # print(df)
        load = df["Consumer_0_electric (kW)"]
        avg_load = load.mean()
        yearly_load = avg_load * 24 * 365
        all_consumption.append(yearly_load)
    overall_avg_load = np.mean(all_consumption)
    overall_std_load = np.std(all_consumption)
    print(f"Overall average yearly load: {overall_avg_load} kWh")
    print(f"Overall std yearly load: {overall_std_load} kWh")


def get_stats_day_ahead_prices():
    price_path = "data/common/day_ahead_price/day_ahead_20_24.csv"
    df = pl.read_csv(price_path)
    df = df.with_columns(
        pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ")
    )
    df = df.sort("datetime")

    years = [2020, 2021, 2022, 2023, 2024]
    for year in years:
        df_year = df.filter(pl.col("datetime").dt.year() == year)
        mean_price = df_year["price (€/kWh)"].mean()
        std = df_year["price (€/kWh)"].std()
        min_price = df_year["price (€/kWh)"].min()
        max_price = df_year["price (€/kWh)"].max()
        print(f"Year {year}: Mean price: {mean_price} EUR/MWh, Std: {std} EUR/MWh")
        print(
            f"Year {year}: Min price: {min_price} EUR/MWh, Max price: {max_price} EUR/MWh"
        )


def get_stats_charging_sessions():
    sessions_dir = "data/common/charging_sessions/"
    all_num_sessions = []
    all_detentions = []
    all_soc_diffs = []
    all_std_detentions = []
    all_std_soc_diffs = []
    all_charged_energy = []
    for file in os.listdir(sessions_dir):
        print(file)
        file_path = os.path.join(sessions_dir, file)
        df = pl.read_csv(file_path, schema_overrides={"type_0 (/)": pl.Utf8})

        # Convert the empty string key column to datetime
        df = df.with_columns(
            pl.col("").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ").alias("datetime")
        ).drop("")
        # Only keep sessions from 2023
        df = df.filter(pl.col("datetime").dt.year() == 2023)
        # Put the datetime column as first column
        df = df.select(["datetime"] + [col for col in df.columns if col != "datetime"])

        # Remove all rows where p_max_0 (kW) is 0
        df = df.filter(pl.col("p_max_0 (kW)") > 0)
        print(df)
        num_sessions = df.height
        all_num_sessions.append(num_sessions)
        detentions = df["det_0 (/)"].to_list()
        all_detentions.extend(detentions)
        soc_diff = df["soc_f_0 (/)"] - df["soc_i_0 (/)"]
        all_soc_diffs.extend(soc_diff.to_list())
        all_std_detentions.append(np.std(detentions))
        all_std_soc_diffs.append(np.std(soc_diff.to_list()))
        capacity = df["capa_0 (kWh)"]
        charged_energy = soc_diff * capacity
        all_charged_energy.append(charged_energy.sum())

    mean_num_sessions = np.mean(all_num_sessions)
    std_num_sessions = np.std(all_num_sessions)
    mean_detention = np.mean(all_detentions)
    std_detention = np.std(all_detentions)
    mean_soc_diff = np.mean(all_soc_diffs)
    std_soc_diff = np.std(all_soc_diffs)
    mean_std_detention = np.mean(all_std_detentions)
    mean_std_soc_diff = np.mean(all_std_soc_diffs)
    std_std_detention = np.std(all_std_detentions)
    std_std_soc_diff = np.std(all_std_soc_diffs)

    mean_charged_energy = np.mean(all_charged_energy)
    std_charged_energy = np.std(all_charged_energy)
    print(
        f"Mean number of sessions per EV: {mean_num_sessions}, std: {std_num_sessions}"
    )
    print(f"Mean detention: {mean_detention}, std: {std_detention}")
    print(f"Mean SOC difference: {mean_soc_diff}, std: {std_soc_diff}")
    print(f"Mean std detention: {mean_std_detention}, std of std: {std_std_detention}")
    print(
        f"Mean std SOC difference: {mean_std_soc_diff}, std of std: {std_std_soc_diff}"
    )
    print(
        f"Mean charged energy per EV: {mean_charged_energy} kWh, std: {std_charged_energy} kWh"
    )


if __name__ == "__main__":
    # calc_comfort_stats()
    # execute_calibration()
    # get_average_daily_load()
    # get_stats_day_ahead_prices()
    get_stats_charging_sessions()
    # plt.show()
