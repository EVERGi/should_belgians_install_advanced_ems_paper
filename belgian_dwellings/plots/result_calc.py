import json
import numpy as np
from sklearn.metrics import mean_absolute_error
import os
import polars as pl


def calculate_mae_house(house_num):
    mae_log_file = f"results/belgium_usefull_500_no_enforcement_mae.json"
    config_file = f"house_{house_num}.json"
    mae_eval_log = json.load(open(mae_log_file, "r"))
    mae_metrics = mae_eval_log[config_file]["MPC_realistic_forecast_no_enforcement"]

    logged_metrics = ["predicted", "previous"]
    predicted_values = ["detention", "soc_difference"]
    log_results = dict()
    for metric in logged_metrics:
        for pred in predicted_values:
            y_true = mae_metrics["real"][pred]
            y_pred = mae_metrics[metric][pred]
            if len(y_true) == 0 or len(y_pred) == 0:
                continue
            if y_pred[0] is None:
                y_pred = y_pred[1:]
                y_true = y_true[1:]
            mae = mean_absolute_error(
                y_true,
                y_pred,
            )
            log_results[f"{metric}_{pred}"] = mae
            # print(f"MAE for {metric} {pred}: {mae:.4f}")
    return log_results


def calculate_mae_all_houses():
    all_results = dict()
    for house_num in range(500):
        print(f"Calculating MAE for house {house_num}")
        log_results = calculate_mae_house(house_num)
        # print(log_results)
        for key, value in log_results.items():
            if key not in all_results:
                all_results[key] = []
            all_results[key].append(value)

    # Print the average results
    for key, values in all_results.items():
        avg_value = sum(values) / len(values)
        print(f"Average {key}: {avg_value:.4f}")
        print(f"Standard deviation {key}: {np.std(values):.4f}")


def calc_elapsed_time_train():
    treec_train_dir = "treec_train_500/"
    all_elapsed_times = []
    for house_dir in sorted(os.listdir(treec_train_dir)):
        print(f"House: {house_dir}")
        house_dir_path = os.path.join(treec_train_dir, house_dir)

        for train_dir in os.listdir(house_dir_path):
            eval_score_path = os.path.join(house_dir_path, train_dir, "eval_score.csv")
            # Read the last line of the eval_score.csv file

            line = pl.read_csv(eval_score_path).tail(1)
            if len(line["elapsed_time"]) == 0:
                continue
            elapsed_time = line["elapsed_time"][0]
            all_elapsed_times.append(elapsed_time)

    print(f"Min elapsed time: {min(all_elapsed_times)}")
    print(f"Max elapsed time: {max(all_elapsed_times)}")
    print(f"Average elapsed time: {sum(all_elapsed_times) / len(all_elapsed_times)}")


if __name__ == "__main__":
    calc_elapsed_time_train()
