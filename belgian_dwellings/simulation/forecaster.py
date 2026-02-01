from simugrid.assets.charger import Charger
from simugrid.assets.energyplus import EnergyPlus
from simugrid.assets.water_heater import WaterHeater
from simugrid.assets.consumer import Consumer
from simugrid.assets.solar_pv import SolarPv

import datetime

from sklearn import neighbors

import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import itertools
import time
from sklearn.preprocessing import StandardScaler


class PerfectForecaster:
    def __init__(self, microgrid):
        self.microgrid = microgrid

        self.chargers = []
        self.energyplus = []
        self.waterheaters = []
        self.consumers = []
        self.solarpvs = []

        self.store_assets()

    def store_assets(self):
        for asset in self.microgrid.assets:

            if isinstance(asset, Charger):
                self.chargers.append(asset)
            elif isinstance(asset, EnergyPlus):
                self.energyplus.append(asset)
            elif isinstance(asset, WaterHeater):
                self.waterheaters.append(asset)
            elif isinstance(asset, Consumer):
                self.consumers.append(asset)
            elif isinstance(asset, SolarPv):
                self.solarpvs.append(asset)

    def get_forecast(self, horizon):

        forec_dict = {}

        charg_forecast = self.charger_forecast(horizon)
        for key, value in charg_forecast.items():
            forec_dict[key] = value

        eplus_forecast = self.energyplus_forecast(horizon)
        for key, value in eplus_forecast.items():
            forec_dict[key] = value

        # Water heater
        waterheat_forecast = self.waterheater_forecast(horizon)
        for key, value in waterheat_forecast.items():
            forec_dict[key] = value

        # Consum load
        forec_dict["pv_and_load"] = self.get_total_load(horizon)

        return forec_dict

    def get_charger_basis(self, charger, horizon):
        time_step = self.microgrid.time_step
        start_dt = self.microgrid.utc_datetime
        end_dt = start_dt + self.microgrid.time_step * horizon
        dt_h = time_step.total_seconds() / 3600
        charger_id = charger.ID

        charger_basis = dict()

        value_name = f"det_{charger_id}"
        charger_basis["det"] = self.get_env_values(value_name, start_dt, end_dt)
        charger_basis["det"] = [int(d / dt_h) for d in charger_basis["det"]]

        value_name = f"soc_i_{charger_id}"
        charger_basis["soc_i"] = self.get_env_values(value_name, start_dt, end_dt)

        value_name = f"soc_f_{charger_id}"
        charger_basis["soc_f"] = self.get_env_values(value_name, start_dt, end_dt)

        value_name = f"p_max_{charger_id}"
        charger_basis["p_max"] = self.get_env_values(value_name, start_dt, end_dt)

        value_name = f"capa_{charger_id}"
        charger_basis["cap"] = self.get_env_values(value_name, start_dt, end_dt)

        if charger.det > 0:
            charger_basis["det"][0] = charger.det
            charger_basis["soc_i"][0] = charger.soc
            charger_basis["soc_f"][0] = charger.soc_f
            charger_basis["p_max"][0] = charger.max_charge
            charger_basis["cap"][0] = charger.size
        return charger_basis

    def charger_forecast(self, horizon):

        charg_forecast = {"det": [], "soc_i": [], "soc_f": [], "p_max": [], "cap": []}

        for charger in self.chargers:
            charger_basis = self.get_charger_basis(charger, horizon)
            det = charger_basis["det"]
            soc_i = charger_basis["soc_i"]
            soc_f = charger_basis["soc_f"]
            p_max = charger_basis["p_max"]
            cap = charger_basis["cap"]

            new_det = []

            for i, d in enumerate(det):
                if i == 0:
                    new_det.append(d)
                    continue
                prev_det = new_det[-1]

                if prev_det > 0 and d == 0:
                    new_det.append(prev_det - 1)
                else:
                    new_det.append(d)

            new_p_max = []
            new_cap = []
            for i, d in enumerate(new_det):
                if p_max[i] > 0:
                    new_p_max.append(p_max[i])
                    new_cap.append(cap[i])
                elif d > 0:
                    new_p_max.append(new_p_max[-1])
                    new_cap.append(new_cap[-1])
                else:
                    new_p_max.append(0)
                    new_cap.append(0)

            new_soc_f = [0 for _ in soc_f]
            for i, s in enumerate(soc_f):
                if s > 0:
                    ind_soc_f = i + det[i] - 1
                    if ind_soc_f < len(soc_f):
                        new_soc_f[ind_soc_f] = s
                    else:
                        equal_soc = (s - soc_i[i]) / det[i]
                        end_soc = soc_i[i] + equal_soc * (len(soc_f) - i)
                        new_soc_f[-1] = end_soc

            charg_forecast["det"].append(new_det)
            charg_forecast["soc_i"].append(soc_i)
            charg_forecast["soc_f"].append(new_soc_f)
            charg_forecast["p_max"].append(new_p_max)
            charg_forecast["cap"].append(new_cap)
        return charg_forecast

    def energyplus_forecast(self, horizon):
        time_step = self.microgrid.time_step

        start_dt = self.microgrid.utc_datetime
        end_dt = start_dt + time_step * horizon
        eplus_forecast = {
            "outdoor_temperature": None,
            "solar_radiation": None,
            "lower_bound": [],
            "upper_bound": [],
        }

        if len(self.energyplus) != 0:

            value_name = "epw_temp_air"
            eplus_forecast["outdoor_temperature"] = self.get_env_values(
                value_name, start_dt, end_dt
            )
            value_name = "epw_dir_norm_rad"
            eplus_forecast["solar_radiation"] = self.get_env_values(
                value_name, start_dt, end_dt
            )

        for eplus in self.energyplus:
            start_pred_bound = start_dt + time_step
            end_pred_bound = end_dt + time_step
            bounds = eplus.get_comfort_sequence(start_pred_bound, end_pred_bound)
            lower_bound = bounds["t_low"]
            upper_bound = bounds["t_up"]
            eplus_forecast["lower_bound"].append(lower_bound)
            eplus_forecast["upper_bound"].append(upper_bound)

        return eplus_forecast
        
    def waterheater_forecast(self, horizon):
        time_step = self.microgrid.time_step

        start_dt = self.microgrid.utc_datetime
        end_dt = start_dt + time_step * horizon

        waterheat_forecast = {"flow": []}

        for waterheater in self.waterheaters:
            value_name = f"{waterheater.name}_flow"
            flow = self.get_env_values(value_name, start_dt, end_dt)
            waterheat_forecast["flow"].append(flow)

        return waterheat_forecast

    def get_total_load(self, horizon):
        time_step = self.microgrid.time_step

        start_dt = self.microgrid.utc_datetime
        end_dt = start_dt + time_step * horizon
        total_load = [0 for _ in range(horizon)]
        for consumer in self.consumers:
            value_name = f"{consumer.name}_electric"
            cons_load = self.get_env_values(value_name, start_dt, end_dt)

            total_load = [l - cons_load[i] for i, l in enumerate(total_load)]

        for pv in self.solarpvs:
            value_name = pv.name
            pv_prod = self.get_env_values(value_name, start_dt, end_dt)

            total_load = [l + pv_prod[i] for i, l in enumerate(total_load)]
        return total_load

    def get_env_values(self, value_name, start_dt, end_dt):
        env_values = self.microgrid.environments[0].env_values

        values = env_values[value_name].get_forecast(start_dt, end_dt)["values"]
        return values


class EasyForcaster(PerfectForecaster):
    def __init__(self, microgrid, ev_forecasting_values):
        super().__init__(microgrid)
        self.soc = []

        self.previous_values = dict()
        self.predicted_values = dict()
        self.prev_forced_charging = False

        self.ev_forecasting_values = ev_forecasting_values

        self.mae_evaluation_log = {
            metric: {"detention": [], "soc_difference": []}
            for metric in ["real", "predicted", "previous"]
        }

    def get_charger_basis(self, charger, horizon):
        time_step = self.microgrid.time_step

        start_dt = self.microgrid.utc_datetime
        end_dt = start_dt + time_step
        dt_h = time_step.total_seconds() / 3600
        charger_id = charger.ID

        pred_det = 0
        pred_soc = 0
        pred_soc_f = 0

        value_name = f"det_{charger_id}"
        init_det = self.get_env_values(value_name, start_dt, end_dt)[0]

        init_det = int(init_det / dt_h)

        if init_det > 0:
            charger_id = int(charger.name.split("_")[-1])
            new_forecasting_values = get_new_ev_forecasting_values(
                self.microgrid, charger
            )
            forcecasting_values = self.ev_forecasting_values[charger_id]
            predictions = get_ev_predictions(
                forcecasting_values, new_forecasting_values
            )
            if (
                new_forecasting_values["arrival_time"]
                not in forcecasting_values["arrival_time"]
            ):
                for key, value in new_forecasting_values.items():
                    forcecasting_values[key].append(value)

            pred_det = int(predictions["detention"] / dt_h)
            pred_soc = 0.95 - float(predictions["soc_difference"])
            pred_soc_f = 0.95

            # Add the predicted values to the mae evaluation log
            # if f"det_{charger_id}" in self.previous_values:
            self.mae_evaluation_log["predicted"]["detention"].append(
                predictions["detention"]
            )
            self.mae_evaluation_log["predicted"]["soc_difference"].append(
                predictions["soc_difference"]
            )
            self.mae_evaluation_log["real"]["detention"].append(init_det * dt_h)
            self.mae_evaluation_log["real"]["soc_difference"].append(
                charger.soc_f - charger.soc
            )
            if f"det_{charger_id}" in self.previous_values:
                self.mae_evaluation_log["previous"]["detention"].append(
                    self.previous_values[f"det_{charger_id}"] * dt_h
                )
                self.mae_evaluation_log["previous"]["soc_difference"].append(
                    self.previous_values[f"soc_f_{charger_id}"]
                    - self.previous_values[f"soc_i_{charger_id}"]
                )
            else:
                self.mae_evaluation_log["previous"]["detention"].append(None)
                self.mae_evaluation_log["previous"]["soc_difference"].append(None)

            # Check if the predicted detention is feasible
            feasible_time = (
                predictions["soc_difference"]
                * charger.size
                / (charger.max_charge * charger.eff)
            )
            feasible_det = int(feasible_time / dt_h)
            if pred_det <= feasible_det:
                print(f"Adjusted pred_det from {pred_det} to {int(feasible_det)+1}")

                pred_det = feasible_det + 1

            self.predicted_values[f"det_{charger_id}"] = pred_det
            self.predicted_values[f"soc_i_{charger_id}"] = pred_soc
            self.predicted_values[f"soc_f_{charger_id}"] = pred_soc_f

            value_name = f"soc_i_{charger_id}"
            init_soc_i = self.get_env_values(value_name, start_dt, end_dt)[0]

            value_name = f"soc_f_{charger_id}"
            init_soc_f = self.get_env_values(value_name, start_dt, end_dt)[0]

            self.previous_values[f"det_{charger_id}"] = init_det
            self.previous_values[f"soc_i_{charger_id}"] = init_soc_i
            self.previous_values[f"soc_f_{charger_id}"] = init_soc_f
        elif charger.det > 0:
            # if charger.det == 16:
            #    print("det 16")
            cur_soc = charger.soc
            charged_percent = cur_soc - self.previous_values[f"soc_i_{charger_id}"]
            cur_det = charger.det
            past_det = self.previous_values[f"det_{charger_id}"] - cur_det

            pred_soc = self.predicted_values[f"soc_i_{charger_id}"] + charged_percent
            pred_soc_f = self.predicted_values[f"soc_f_{charger_id}"]
            pred_det = self.predicted_values[f"det_{charger_id}"] - past_det

            EPSILON = 0.000001
            forced_charging = self.microgrid.management_system.forced_charging
            new_forced_charging = forced_charging and not self.prev_forced_charging
            self.prev_forced_charging = forced_charging

            if charger.soc > charger.soc_f - EPSILON:
                # If the car is full, adjust the predicted values to show it

                # if pred_det > 0:
                #    print(pred_det)
                #    print(f"{self.microgrid.utc_datetime}")
                #    print("Car full, no more charging")
                pred_soc = charger.soc
                pred_soc_f = charger.soc_f
                pred_det = 1

            elif (
                pred_det == 0 or pred_soc > pred_soc_f - EPSILON or new_forced_charging
            ):
                # If the car is still connected but the forecaster predicted it would be full
                # Then schedule the car to charge at half the max possible energy for 2 hours

                time_charge_h = 2
                det_increase = int(time_charge_h / dt_h)

                if forced_charging:
                    # If the car is more than full, this means that there was forced charging
                    # So the car will charge to the max power
                    mean_charge_power = charger.max_charge
                    # print(f"{self.microgrid.utc_datetime}")
                    # print("MPC: Force charging detected in MPC")
                else:
                    # If the car is not more than full, then allow some flexibility
                    mean_charge_power = charger.max_charge
                    # print(f"{self.microgrid.utc_datetime}")
                    # print("End of planned charging, intermediate charging activated")
                soc_i_decrease = (
                    mean_charge_power * time_charge_h * charger.eff / charger.size
                )

                pred_det = det_increase
                pred_soc = pred_soc_f - soc_i_decrease

                self.predicted_values[f"det_{charger_id}"] = past_det + pred_det
                self.predicted_values[f"soc_i_{charger_id}"] = (
                    pred_soc_f - charged_percent - soc_i_decrease
                )

        init_pred = {
            "det": pred_det,
            "soc_i": pred_soc,
            "soc_f": pred_soc_f,
            "p_max": charger.max_charge,
            "cap": charger.size,
        }
        charger_basis = dict()
        for key, value in init_pred.items():
            padded_value = [value] + [0 for _ in range(horizon - 1)]
            charger_basis[key] = padded_value
        return charger_basis

    def energyplus_forecast(self, horizon):
        time_step = self.microgrid.time_step

        start_dt = self.microgrid.utc_datetime
        end_dt = start_dt + time_step * horizon
        eplus_forecast = {
            "outdoor_temperature": None,
            "solar_radiation": None,
            "lower_bound": [],
            "upper_bound": [],
        }

        if len(self.energyplus) != 0:
            value_name = "epw_temp_air"
            eplus_forecast["outdoor_temperature"] = self.get_naive_forecast(
                value_name, horizon
            )
            value_name = "epw_dir_norm_rad"
            eplus_forecast["solar_radiation"] = self.get_naive_forecast(
                value_name, horizon
            )

        for eplus in self.energyplus:
            start_pred_bound = start_dt + time_step
            end_pred_bound = end_dt + time_step
            bounds = eplus.get_comfort_sequence(start_pred_bound, end_pred_bound)
            lower_bound = bounds["t_low"]
            upper_bound = bounds["t_up"]
            eplus_forecast["lower_bound"].append(lower_bound)
            eplus_forecast["upper_bound"].append(upper_bound)

        return eplus_forecast

    def waterheater_forecast(self, horizon):

        waterheat_forecast = {"flow": []}

        for waterheater in self.waterheaters:
            value_name = f"{waterheater.name}_flow"
            flow = self.get_naive_forecast(value_name, horizon)
            waterheat_forecast["flow"].append(flow)

        return waterheat_forecast

    def get_total_load(self, horizon):
        total_load = [0 for _ in range(horizon)]
        for consumer in self.consumers:
            value_name = f"{consumer.name}_electric"
            cons_load = self.get_naive_forecast(value_name, horizon)

            total_load = [l - cons_load[i] for i, l in enumerate(total_load)]

        for pv in self.solarpvs:
            value_name = pv.name
            pv_prod = self.get_naive_forecast(value_name, horizon)

            total_load = [l + pv_prod[i] for i, l in enumerate(total_load)]
        return total_load

    def get_naive_forecast(self, value_name, horizon):
        # This forecast gets the last value and repeats it for an hour
        # The rest is replaced by the values of the previous day
        time_step = self.microgrid.time_step
        start_dt = self.microgrid.utc_datetime

        ONE_HOUR = datetime.timedelta(hours=1)

        start_naive = start_dt - 24 * ONE_HOUR
        end_naive = start_dt

        num_steps_day = (24 * ONE_HOUR) // time_step
        naive_values = self.get_env_values(value_name, start_naive, end_naive)

        naive_day_before = [naive_values[i % num_steps_day] for i in range(horizon)]

        start_prev_step = start_dt - time_step
        end_prev_step = start_dt
        prev_step_value = self.get_env_values(
            value_name, start_prev_step, end_prev_step
        )[0]

        num_prev_step_value = ONE_HOUR // time_step

        for i in range(num_prev_step_value):
            naive_day_before[i] = prev_step_value

        naive_forecast = naive_day_before
        return naive_forecast


def get_ev_predictions(ev_forecasting_values, new_ev_forecasting_values):

    training_df = format_ev_schedule_data(ev_forecasting_values)
    fitted_params = [
        "hour",
        "weekday",
        "prev_soc_difference",
        "prev_detention",
        "time_since_prev_charge",
    ]
    ev_prediction_values = dict()
    for forecasting_keys, forecasting_value in new_ev_forecasting_values.items():
        ev_prediction_values[forecasting_keys] = [
            ev_forecasting_values[forecasting_keys][-1],
            forecasting_value,
        ]

    input_predictions = format_ev_schedule_data(ev_prediction_values)

    best_model = train_ev_forecaster(training_df, fitted_params=fitted_params)

    predicted_values = ["detention", "soc_difference"]
    predictions = dict()
    for predicted_value in predicted_values:
        model = best_model[predicted_value]["model"]
        scaler = best_model[predicted_value]["scaler"]
        params = best_model[predicted_value]["params"]
        to_scale = input_predictions[params].to_numpy()
        input_scaled = scaler.transform(to_scale)
        prediction = model.predict(input_scaled)[0]

        predictions[predicted_value] = prediction

    return predictions


def train_ev_forecaster(ev_schedule_df, fitted_params):
    predicted_values = ["detention", "soc_difference"]
    mae_scores = dict()
    param_candidates = fitted_params

    for random_state in range(5):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            ev_schedule_df[fitted_params],
            ev_schedule_df[predicted_values],
            test_size=0.2,
            random_state=random_state,
        )
        all_param_combinations = []

        for i in range(1, len(param_candidates) + 1):
            all_param_combinations += itertools.combinations(param_candidates, i)

        for params in all_param_combinations:
            for predicted_value in predicted_values:
                if predicted_value not in mae_scores:
                    mae_scores[predicted_value] = dict()

                mae_key = ",".join(params)
                if mae_key not in mae_scores[predicted_value]:
                    mae_scores[predicted_value][mae_key] = []

                model, scaler = train_model(X_train, y_train, params, predicted_value)

                mae = calculate_mae(
                    model, scaler, X_test, y_test, params, predicted_value
                )
                mae_scores[predicted_value][mae_key].append(mae)

    best_params = get_best_params(predicted_values, mae_scores)

    # print("Best parameters:")
    for predicted_value, params in best_params.items():
        param_key = ",".join(params)
        mae_list = mae_scores[predicted_value][param_key]
        mean_mae = sum(mae_list) / len(mae_list)
        # print(f"{predicted_value}: {params} with MAE: {mean_mae:.4f}")

    # Retrain model and scaler with all data
    best_model = {}
    for predicted_value, params in best_params.items():
        model, scaler = train_model(
            ev_schedule_df, ev_schedule_df, params, predicted_value
        )
        best_model[predicted_value] = {
            "model": model,
            "scaler": scaler,
            "params": params,
        }

    return best_model


def train_model(X_train, y_train, params, predicted_value):
    X_train_params = X_train[params].to_numpy()
    y_train_predicted = y_train[predicted_value].to_numpy()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_params)
    neigh = neighbors.KNeighborsRegressor()
    model = neigh.fit(X_train_scaled, y_train_predicted)
    return model, scaler


def calculate_mae(model, scaler, X_test, y_test, params, predicted_value):
    X_test_params = X_test[params].to_numpy()
    y_test_predicted = y_test[predicted_value].to_numpy()

    # Normalise features
    X_test_scaled = scaler.transform(X_test_params)

    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test_predicted, y_pred)
    return mae


def get_best_params(predicted_values, mae_scores):
    best_params = {}
    for predicted_value in predicted_values:
        best_mae = float("inf")
        best_key = None
        for key, mae_list in mae_scores[predicted_value].items():
            mean_mae = sum(mae_list) / len(mae_list)
            if mean_mae < best_mae:
                best_mae = mean_mae
                best_key = key
        best_params[predicted_value] = best_key.split(",")
    return best_params


def format_ev_schedule_data(ev_schedule_data):
    ev_schedule_df = pl.DataFrame(ev_schedule_data)

    # Add a column for time of day in hours, day of week
    ev_schedule_df = ev_schedule_df.with_columns(
        [
            (
                pl.col("arrival_time").dt.hour()
                + pl.col("arrival_time").dt.minute() / 60
            ).alias("hour"),
            pl.col("arrival_time").dt.weekday().alias("weekday"),
        ]
    )
    # Add a column for the previous soc_difference, detention and time since previous charge
    ev_schedule_df = ev_schedule_df.with_columns(
        [
            pl.col("soc_difference").shift(1).alias("prev_soc_difference"),
            pl.col("detention").shift(1).alias("prev_detention"),
            # pl.col("arrival_time").shift(1).alias("prev_arrival_time"),
            (pl.col("arrival_time") - pl.col("arrival_time").shift(1))
            .dt.total_seconds()
            .alias("time_since_prev_charge"),
        ]
    )

    # Remove rows with null values in the new columns
    ev_schedule_df = ev_schedule_df.drop_nulls(
        subset=[
            "prev_soc_difference",
            "prev_detention",
            "time_since_prev_charge",
        ]
    )

    return ev_schedule_df


def get_new_ev_forecasting_values(microgrid, charger):
    env_values = microgrid.environments[0].env_values
    soc_i = env_values[f"soc_i_{charger.ID}"].value
    soc_f = env_values[f"soc_f_{charger.ID}"].value
    det = env_values[f"det_{charger.ID}"].value
    new_forecasting_values = dict()
    new_forecasting_values["soc_difference"] = soc_f - soc_i
    new_forecasting_values["arrival_time"] = microgrid.utc_datetime
    new_forecasting_values["detention"] = det

    return new_forecasting_values


if __name__ == "__main__":
    a = [
        {
            "arrival_time": [
                datetime.datetime(2023, 1, 1, 17, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 2, 21, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 3, 12, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 4, 10, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 4, 22, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 5, 22, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 6, 19, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 7, 12, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 7, 18, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 7, 23, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 8, 20, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 11, 21, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 12, 22, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 13, 20, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 16, 22, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 18, 12, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 19, 23, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 21, 17, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 22, 14, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 24, 23, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 26, 17, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 27, 20, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 28, 10, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 1, 31, 22, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 2, 2, 15, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 2, 2, 22, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 2, 5, 17, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 2, 6, 23, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 2, 9, 23, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 2, 16, 22, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 2, 17, 18, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 2, 18, 23, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 2, 23, 7, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 2, 23, 22, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 2, 24, 19, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 2, 25, 15, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 2, 26, 17, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 2, 27, 18, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 2, 21, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 3, 19, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 5, 0, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 5, 21, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 6, 23, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 9, 22, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 13, 23, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 16, 23, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 17, 19, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 19, 10, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 19, 15, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 20, 22, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 23, 22, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 26, 21, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 30, 23, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 31, 22, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 4, 1, 13, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 4, 2, 20, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 4, 6, 22, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 4, 7, 15, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 4, 8, 20, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 4, 9, 10, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 4, 9, 18, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 4, 10, 9, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 4, 10, 22, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 4, 14, 18, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 4, 23, 17, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 4, 27, 22, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 5, 7, 20, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 5, 8, 10, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 5, 8, 22, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 5, 11, 21, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 5, 14, 16, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 5, 15, 21, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 5, 17, 22, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 5, 20, 12, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 5, 21, 20, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 5, 24, 22, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 5, 29, 3, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 5, 30, 23, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 5, 31, 18, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 6, 3, 1, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 6, 3, 10, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 6, 5, 22, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 6, 10, 7, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 6, 10, 15, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 6, 15, 23, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 6, 16, 8, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 6, 16, 21, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 6, 23, 0, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 6, 23, 21, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 6, 25, 17, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 6, 29, 16, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 7, 2, 20, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 7, 3, 19, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 7, 8, 17, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 7, 10, 22, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 7, 13, 21, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 7, 17, 18, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 7, 25, 20, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 7, 26, 3, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 7, 27, 12, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 7, 27, 17, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 7, 31, 14, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 8, 18, 19, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 8, 19, 11, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 8, 24, 21, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 8, 27, 21, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 8, 30, 16, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 8, 31, 21, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 2, 14, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 3, 20, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 4, 21, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 5, 21, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 7, 21, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 8, 21, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 9, 16, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 9, 20, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 10, 18, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 11, 21, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 23, 18, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 23, 23, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 24, 0, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 24, 13, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 24, 20, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 26, 22, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 28, 21, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 9, 29, 20, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 10, 2, 22, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 10, 12, 21, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 10, 23, 22, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 10, 28, 21, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 10, 29, 21, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 11, 2, 23, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 11, 6, 23, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 11, 9, 22, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 11, 10, 18, 45, tzinfo=datetime.UTC),
                datetime.datetime(2023, 11, 15, 21, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 11, 18, 15, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 11, 25, 17, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 11, 27, 23, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 11, 29, 20, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 12, 2, 20, 15, tzinfo=datetime.UTC),
                datetime.datetime(2023, 12, 4, 20, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 12, 14, 20, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 12, 15, 5, 30, tzinfo=datetime.UTC),
                datetime.datetime(2023, 12, 15, 22, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 12, 18, 23, 0, tzinfo=datetime.UTC),
                datetime.datetime(2023, 12, 21, 23, 30, tzinfo=datetime.UTC),
            ],
            "detention": [
                76,
                56,
                79,
                15,
                61,
                44,
                62,
                21,
                13,
                43,
                67,
                43,
                44,
                73,
                59,
                82,
                51,
                81,
                62,
                37,
                57,
                55,
                97,
                39,
                28,
                45,
                80,
                59,
                27,
                44,
                90,
                55,
                24,
                48,
                66,
                83,
                68,
                84,
                49,
                59,
                53,
                61,
                57,
                48,
                56,
                47,
                81,
                11,
                66,
                55,
                48,
                65,
                46,
                34,
                13,
                63,
                26,
                78,
                47,
                8,
                4,
                6,
                58,
                67,
                68,
                55,
                38,
                7,
                50,
                42,
                73,
                32,
                51,
                74,
                66,
                50,
                22,
                49,
                54,
                27,
                61,
                26,
                29,
                81,
                10,
                12,
                55,
                33,
                28,
                85,
                70,
                64,
                67,
                66,
                62,
                43,
                69,
                22,
                11,
                19,
                69,
                85,
                63,
                90,
                54,
                36,
                78,
                44,
                87,
                65,
                58,
                56,
                54,
                59,
                13,
                60,
                72,
                56,
                8,
                1,
                47,
                22,
                65,
                46,
                53,
                80,
                35,
                48,
                46,
                65,
                52,
                50,
                60,
                48,
                57,
                65,
                66,
                81,
                43,
                70,
                70,
                66,
                35,
                45,
                44,
                58,
                40,
            ],
            "soc_difference": [
                0.015070093457943856,
                0.20946261682243006,
                0.032710280373831835,
                0.05116822429906542,
                0.14532710280373795,
                0.13785046728971995,
                0.18726635514018697,
                0.093107476635514,
                0.026985981308411233,
                0.16845794392523394,
                0.22126168224299103,
                0.10093457943925199,
                0.12546728971962595,
                0.22254672897196304,
                0.13107476635514004,
                0.225233644859813,
                0.21039719626168196,
                0.19450934579439194,
                0.11997663551401905,
                0.19065420560747703,
                0.26857476635514,
                0.20327102803738295,
                0.017757009345794383,
                0.12056074766355096,
                0.167289719626168,
                0.047663551401869175,
                0.21565420560747695,
                0.15607476635514006,
                0.15957943925233598,
                0.20724299065420593,
                0.11051401869158906,
                0.10911214953270998,
                0.15443925233644906,
                0.15221962616822404,
                0.137383177570093,
                0.11074766355140198,
                0.266588785046729,
                0.28983644859813107,
                0.21495327102803696,
                0.11612149532710303,
                0.16098130841121505,
                0.152336448598131,
                0.16355140186915895,
                0.19707943925233595,
                0.14848130841121499,
                0.131542056074766,
                0.140654205607477,
                0.09626168224299059,
                0.19696261682243,
                0.325116822429906,
                0.205724299065421,
                0.14135514018691597,
                0.17371495327102793,
                0.16997663551401898,
                0.08948598130841123,
                0.21565420560747695,
                0.21425233644859798,
                0.19287383177570105,
                0.207710280373832,
                0.07488317757009355,
                0.03235981308411218,
                0.012149532710280408,
                0.188785046728972,
                0.6528037383177571,
                0.6281542056074769,
                0.540654205607477,
                0.374299065420561,
                0.06717289719626174,
                0.48901869158878497,
                0.4120327102803739,
                0.6385514018691589,
                0.30315420560747697,
                0.495443925233645,
                0.7045560747663551,
                0.46588785046729,
                0.491939252336449,
                0.21623831775700897,
                0.47464953271028,
                0.33971962616822404,
                0.264602803738318,
                0.08948598130841123,
                0.25046728971962595,
                0.2875,
                0.45315420560747693,
                0.09485981308411207,
                0.11577102803738304,
                0.5195093457943929,
                0.324532710280374,
                0.268224299065421,
                0.256542056074766,
                0.685397196261682,
                0.520210280373832,
                0.481191588785047,
                0.28796728971962593,
                0.604556074766355,
                0.3933411214953271,
                0.509579439252336,
                0.21273364485981305,
                0.10315420560747701,
                0.17745327102803699,
                0.21238317757009295,
                0.8,
                0.613785046728972,
                0.08726635514018688,
                0.35887850467289706,
                0.34135514018691604,
                0.237967289719626,
                0.190771028037383,
                0.6588785046728969,
                0.5252336448598129,
                0.22605140186915895,
                0.5273364485981309,
                0.528271028037383,
                0.52803738317757,
                0.12686915887850503,
                0.21787383177570097,
                0.642757009345794,
                0.22091121495327104,
                0.07476635514018692,
                0.008995327102803707,
                0.4587616822429911,
                0.21565420560747695,
                0.30525700934579403,
                0.44999999999999996,
                0.46121495327102796,
                0.5047897196261679,
                0.338200934579439,
                0.46810747663551405,
                0.448247663551402,
                0.620093457943925,
                0.4559579439252341,
                0.46892523364486005,
                0.584228971962617,
                0.473364485981308,
                0.37873831775700906,
                0.492056074766355,
                0.11927570093457895,
                0.5914719626168221,
                0.419976635514019,
                0.6880841121495329,
                0.5401869158878511,
                0.644392523364486,
                0.355724299065421,
                0.37231308411215003,
                0.42663551401869193,
                0.567990654205607,
                0.377570093457944,
            ],
        }
    ]
    param_candidates = [
        "hour",
        "weekday",
        "prev_soc_difference",
        "prev_detention",
        "time_since_prev_charge",
    ]

    new_prediction_values = {
        "arrival_time": datetime.datetime(2023, 12, 24, 21, 0, tzinfo=datetime.UTC),
        "soc_difference": 0.37616822429906505,
        "detention": 39,
    }
    print(get_ev_predictions(a[0], new_prediction_values))
