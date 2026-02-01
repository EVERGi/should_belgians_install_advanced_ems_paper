import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# from xgboost import XGBRegressor

import numpy as np


def fill_missing_timesteps(df):
    """
    Fill missing values in the dataframe with nan values
    """
    df.index = pd.to_datetime(df.index)

    df = df.dropna()
    df = df.sort_index()

    # Get the index of the dataframe
    index = df.index
    # Get the first and last timestamp
    first_timestamp = index[0]
    last_timestamp = index[-1]
    # Get the frequency of the index
    freq = "15min"
    # Create a new index with all the timesteps
    new_index = pd.date_range(start=first_timestamp, end=last_timestamp, freq=freq)

    # Reindex the dataframe
    df = df.reindex(new_index)

    return df


def fit_imputer(df):
    """
    Fit imputer to the dataframe
    """

    # Input for the imputer
    # Day of the week, minute of the day and previous 96 values

    # Output of the imputer
    # Next value
    first_col_index = df.columns[0]
    df_train = df.copy()

    df_train["day_of_week"] = df_train.index.dayofweek
    df_train["minute_of_day"] = df_train.index.hour * 60 + df_train.index.minute
    for i in range(1, 97):
        df_train[f"value_-{i}"] = df_train[first_col_index].shift(i)

    # Drop rows with NaN values
    df_train = df_train.dropna()

    # Train a forecasting model to predict the next value
    # Use gradient boosting regressor
    X = df_train[
        ["day_of_week", "minute_of_day"] + [f"value_-{i}" for i in range(1, 97)]
    ].values
    Y = df_train[first_col_index].values

    # Fit the model
    model = GradientBoostingRegressor()
    print("Fitting model")
    model.fit(X, Y)
    print("Model fitted")

    return model


def impute_nan_values_with_model(df, forecast_model, no_neg=False):
    # Iterate over the rows of the dataframe with the index
    # For each row, if the value is NaN, predict the value with the model
    # If the value is not NaN, do nothing
    # Return the dataframe with no NaN values

    first_col_index = df.columns[0]
    for index, row in df.iterrows():
        if pd.isna(row.iloc[0]):
            # Predict the value
            # Get the day of the week
            day_of_week = index.dayofweek
            # Get the minute of the day
            minute_of_day = index.hour * 60 + index.minute

            # Get the previous 96 values
            prev_values = df.loc[
                index
                - pd.Timedelta(minutes=15 * 96) : index
                - pd.Timedelta(minutes=15),
                first_col_index,
            ][::-1]

            prev_values = prev_values.values
            prev_values = list(prev_values) + [None] * (96 - len(prev_values))
            # Make the values list length 96 with NaN values
            # Linear interpolate all nan values
            if None in prev_values:
                prev_values = pd.Series(prev_values).interpolate().values
                prev_values = list(prev_values)

            # Create the input
            input = [day_of_week, minute_of_day] + prev_values
            input = np.array(input, dtype=np.float64)
            # Predict the value
            value = forecast_model.predict([input])[0]
            # Set the value in the dataframe

            if no_neg:
                value = max(0, value)
            df.loc[index, first_col_index] = value
    return df


def impute_missing_values(
    file_path, file_path_output=None, plot_results=False, no_neg=False
):
    """
    Impute missing values in the dataframe with gradient boosting regressor
    """
    df = pd.read_csv(file_path, index_col=0)

    df = fill_missing_timesteps(df)
    reindexed_df = df.copy()

    forecast_model = fit_imputer(df)

    df = impute_nan_values_with_model(df, forecast_model, no_neg)

    # Plot on previous pandas plot

    if plot_results:
        file_name = file_path.split("/")[-1]
        plt.plot(df.index, df[df.columns[0]], label="imputed")
        plt.plot(
            reindexed_df.index, reindexed_df[reindexed_df.columns[0]], label="original"
        )
        plt.legend()
        plt.ylabel("Power (kW)")
        plt.xlabel("Time")

        plt.title(f"{file_name}")
        plt.show()

    if file_path_output is not None:
        # Write datetime index in ISO format
        df.index = df.index.strftime("%Y-%m-%dT%H:%M:%SZ")
        # In csv file, write the index header as "datetime"
        df.index.name = "datetime"
        df.to_csv(file_path_output)


if __name__ == "__main__":
    file_path = "data/hdf5_loads/SFH5.csv"
    impute_missing_values(file_path, plot_results=True, no_neg=True)
