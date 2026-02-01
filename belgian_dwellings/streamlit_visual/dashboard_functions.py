import csv
import os

import streamlit as st
from simugrid.misc.log_plot_micro import get_power_hist_data_dict, plot_power_hist
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
import numpy as np
import time
import json


def read_eval_score(eval_file):
    with open(eval_file, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if row[0] == "timestep":
                continue
            else:
                return row[2]


def get_subdivisions(results_folder):
    subdivisions = dict()

    for folder_path in sorted(os.listdir(results_folder)):
        if folder_path.endswith(".csv"):
            continue
        param_file = results_folder + folder_path + "/params_run.json"

        with open(param_file, "r") as jsonfile:
            params_run = json.load(jsonfile)

        common_params = params_run["common_params"]
        if "config_file" in common_params:
            subdivision_name = common_params["config_file"].split("/")[-2]
        elif "gym_env" in common_params.keys():
            subdivision_name = common_params["gym_env"]
        else:
            subdivision_name = common_params["case"]

        if subdivision_name in subdivisions.keys():
            subdivisions[subdivision_name].append(folder_path)
        else:
            subdivisions[subdivision_name] = [folder_path]
    return subdivisions


def csv_to_dict(filepath):
    with open(filepath, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        mydict = {rows[0]: rows[1] for rows in csvreader if len(rows) == 2}
    return mydict


def show_valid_tables(results_folder, shown_run):
    if len(shown_run) != 0:
        st.header("Validation scores")
    valid_scores = dict()
    for folder in shown_run:
        param_file = results_folder + folder + "/params_run.json"

        with open(param_file, "r") as jsonfile:
            params_run = json.load(jsonfile)

        joint_params = {**params_run["common_params"], **params_run["algo_params"]}
        joint_params["algo_type"] = params_run["algo_type"]

        if "start_time_valid" in joint_params:
            start_time_valid = joint_params["start_time"]
        else:
            start_time_valid = ""
        tot_steps_valid = joint_params["tot_steps"]
        if "config_file" in joint_params:
            config_file = joint_params["config_file"].split("/")[-1]
        elif "gym_env" in joint_params.keys():
            config_file = joint_params["gym_env"]
        else:
            config_file = joint_params["case"]
        key_valid = f"{start_time_valid}_{tot_steps_valid}_{config_file}"

        rule_base_score = get_rule_base_score(results_folder, param_file, False)
        str_score = "{:.1f}".format(rule_base_score)
        if key_valid in valid_scores.keys():
            valid_scores[key_valid]["Rule base"] = str_score
        else:
            valid_scores[key_valid] = {"Rule base": str_score}

        start_folder = "_".join(folder.split("_")[:-1])
        validation_folder = (
            results_folder + folder + "/validation/" + start_folder + "_0/"
        )

        valid_eval_file = validation_folder + "eval_score.csv"
        validation_score = read_eval_score(valid_eval_file)
        valid_scores[key_valid][folder] = validation_score

    for keys, scores in valid_scores.items():
        folders = [i for i in scores.keys() if i != "Rule base"]
        valid_folders = [
            i + "/validation/" + "_".join(i.split("_")[:-1]) + "_0/" for i in folders
        ]

        values = [float(value) for key, value in scores.items() if key != "Rule base"]
        names_diff = names_from_differences(results_folder, valid_folders)
        name = [names_diff[i] for i in valid_folders]

        plot_values = pd.DataFrame({"names": folders, "cost": values, "algo": name})

        st.text(keys_to_good_text(keys))
        # st.table(scores.items())
        fig = px.scatter(plot_values, x="algo", y="cost", hover_name="names")

        score_rb = scores["Rule base"]
        x_axis = plot_values["algo"]
        min_x = x_axis[0]
        max_x = x_axis[len(x_axis) - 1]
        fig.add_trace(
            go.Scatter(
                name="Rule base", x=[min_x, max_x], y=[score_rb, score_rb], mode="lines"
            ),
            row=1,
            col=1,
        )

        st.plotly_chart(fig)


def keys_to_good_text(keys):
    splitted = keys.split("_")
    start_time = splitted[0]
    tot_steps = splitted[1]
    config_file = "_".join(splitted[2:])

    text = ""
    text += "Config file: " + config_file + "\n"
    text += "Start time: " + start_time + "\n"
    text += "Total steps: " + tot_steps
    return text


def display_power_train(results_folder, shown_run):
    if len(shown_run) != 0:
        st.header("Training power profiles")
    for folder in shown_run:
        # eval_score[folder] = validation_score

        train_power_dir = results_folder + folder + "/power_profiles/"
        power_hist_files = [i for i in os.listdir(train_power_dir)]
        scores_obtained = sorted(
            set([float(i.replace(".csv", "").split("_")[-1]) for i in power_hist_files])
        )

        selected_set = st.multiselect(folder + " scores", scores_obtained)
        for score in selected_set:
            selected_files = [
                train_power_dir + i
                for i in os.listdir(train_power_dir)
                if i.find(str(score)) != -1
            ]

            power_hist = get_power_hist_data_dict(selected_files)

            start_index = 0
            to_step = 1000

            for i, node_hist in enumerate(power_hist):
                for key, value in node_hist.items():
                    power_hist[i][key] = value[start_index : start_index + to_step]

            col1, col2 = st.columns(2)

            with col1:
                plot_pyplot(power_hist)

            with col2:
                plot_plotly(power_hist)


def plot_pyplot(power_hist):
    fig = plot_power_hist(power_hist)
    st.pyplot(fig[0])


def plot_plotly(power_hist):
    comb_power_hist = dict()

    for node_hist in power_hist:
        for key, value in node_hist.items():
            if key != "datetime":
                value = [i.electrical for i in value]
            comb_power_hist[key] = value

    data = [list(value) for key, value in comb_power_hist.items() if key != "datetime"]
    columns = [key for key in comb_power_hist.keys() if key != "datetime"]

    data = list(map(list, zip(*data)))
    df_bar = pd.DataFrame(data, index=comb_power_hist["datetime"], columns=columns)

    fig = px.bar(df_bar, labels={"value": "Power (kW)", "index": "time"})
    st.plotly_chart(fig)


def display_training_scores(results_folder, shown_run):
    if len(shown_run) != 0:
        st.header("Training evaluation scores")
    train_eval = dict()

    rule_base_scores = dict()

    for folder in shown_run:
        param_file = results_folder + folder + "/params_run.json"
        with open(param_file, "r") as jsonfile:
            params_run = json.load(jsonfile)
        joint_params = {**params_run["common_params"], **params_run["algo_params"]}
        joint_params["algo_type"] = params_run["algo_type"]

        if "start_time_train" in joint_params:
            start_time_valid = joint_params["start_time"]
        else:
            start_time_valid = ""
        tot_steps_valid = joint_params["tot_steps"]

        if "config_file" in joint_params:
            config_file = joint_params["config_file"].split("/")[-1]
        elif "gym_env" in joint_params:
            config_file = joint_params["gym_env"]
        else:
            config_file = joint_params["case"]
        key_valid = f"{start_time_valid}_{tot_steps_valid}_{config_file}"

        eval_score_file = results_folder + folder + "/eval_score.csv"
        df = pd.read_csv(eval_score_file)

        df.rename({"eval_score": folder}, axis=1, inplace=True)

        if key_valid in train_eval.keys():
            train_eval[key_valid].append(df)
        else:
            train_eval[key_valid] = [df]

        rule_base_score = get_rule_base_score(results_folder, param_file, True)
        rule_base_scores[key_valid] = rule_base_score

    names_diff = names_from_differences(results_folder, shown_run)

    for keys, dfs in train_eval.items():
        st.text(keys_to_good_text(keys))
        col1, col2 = st.columns(2)

        folder = dfs[0].columns[1]

        fig_sim = px.line()
        fig_cum = px.line()

        seen_names = list()
        colors = px.colors.qualitative.Plotly
        for df in dfs:
            folder = df.columns[2]
            name = names_diff[folder]
            new_name = name not in seen_names
            if new_name:
                seen_names.append(name)
            color_ind = seen_names.index(name) % len(colors)
            color = colors[color_ind]
            marker = dict(color=color)
            fig_sim.add_scatter(
                x=df["timestep"],
                y=df[folder],
                name=name,
                legendgroup=name,
                marker=marker,
                showlegend=new_name,
            )
            fig_cum.add_scatter(
                x=df["elapsed_time"],
                y=df[folder].cummax(),
                name=name,
                legendgroup=name,
                marker=marker,
                showlegend=new_name,
            )

        score_rb = rule_base_scores[keys]

        for fig in [fig_sim, fig_cum]:
            if fig is fig_sim:
                x_axis = "timestep"
            else:
                x_axis = "elapsed_time"
            fig.add_trace(
                go.Scatter(
                    name="Rule base",
                    x=[df[x_axis].min(), df[x_axis].max()],
                    y=[score_rb, score_rb],
                    mode="lines",
                ),
                row=1,
                col=1,
            )
            if fig is fig_sim:
                with col1:
                    st.plotly_chart(fig)
            else:
                with col2:
                    st.plotly_chart(fig)


def names_from_differences(results_folder, folders_to_compare):
    folder_differences = find_different_params(results_folder, folders_to_compare)

    for folder, diff_params in folder_differences.items():
        diff_params.pop("config_file", None)
        diff_params.pop("gym_env", None)
        diff_params.pop("case", None)
        diff_params.pop("train_steps", None)
        diff_params.pop("gen", None)

        name = ""
        for param, value in diff_params.items():
            if param == "continuous":
                if value == "True":
                    name += "continuous_"
                else:
                    name += "discrete_"
            elif value == "sb3":
                name += "RL_"
            else:
                name += value + "_"

        folder_differences[folder] = name[:-1]
    return folder_differences


def find_different_params(results_folder, folders_to_compare):
    saved_params = list()
    params_folders = list()
    differences_keys = list()

    for folder in folders_to_compare:
        param_file = results_folder + folder + "/params_run.json"
        with open(param_file, "r") as jsonfile:
            params_dict = json.load(jsonfile)
        joint_params = {**params_dict["common_params"], **params_dict["algo_params"]}
        joint_params["algo_type"] = params_dict["algo_type"]

        found_match = False
        for i, param in enumerate(saved_params):
            difference = difference_two_params(joint_params, param)
            if difference == dict():
                params_folders[i].append(folder)
                found_match = True
            else:
                differences_keys += difference.keys()
        if not found_match:
            params_folders.append([folder])
            saved_params.append(joint_params)

    differences_keys = list(set(differences_keys))

    folder_differences = dict()
    for i, folders in enumerate(params_folders):
        params = saved_params[i]
        for folder in folders:
            diff_params = {j: params[j] for j in differences_keys if j in params.keys()}
            folder_differences[folder] = diff_params
    return folder_differences


def difference_two_params(params_1, params_2):
    differences = dict()
    common_elements = list(set(params_1).intersection(params_2))

    for key in common_elements:
        if params_1[key] != params_2[key]:
            differences[key] = [params_1[key], params_2[key]]

    return differences


def display_validation_trees(results_folder, shown_run):
    main_title = False

    for folder in shown_run:
        if folder.find("tree") == -1:
            continue
        if not main_title:
            st.header("Validation trees")
            main_title = True

        st.subheader(folder)

        start_folder = "_".join(folder.split("_")[:-1])
        validation_folder = (
            results_folder + folder + "/validation/" + start_folder + "_0/"
        )

        dot_tree_folder = validation_folder + "dot_trees/"

        for tree_file in sorted(os.listdir(dot_tree_folder)):
            file = open(dot_tree_folder + tree_file, "r")
            dot_tree = file.read()
            st.graphviz_chart(dot_tree)


def display_power_valid(results_folder, shown_run):
    if len(shown_run) != 0:
        st.header("Validation power profiles")
    for folder in shown_run:
        # eval_score[folder] = validation_score
        show = st.checkbox("Show power plot " + folder)
        col1, col2 = st.columns(2)
        if show:
            start_folder = "_".join(folder.split("_")[:-1])
            validation_folder = (
                results_folder + folder + "/validation/" + start_folder + "_0/"
            )
            validation_power_dir = validation_folder + "power_profiles/"

            power_hist_files = [
                validation_power_dir + i for i in os.listdir(validation_power_dir)
            ]

            power_hist = get_power_hist_data_dict(power_hist_files)

            start_index = 0
            to_step = 1000

            for i, node_hist in enumerate(power_hist):
                for key, value in node_hist.items():
                    power_hist[i][key] = value[start_index : start_index + to_step]

            with col1:
                plot_pyplot(power_hist)
            with col2:
                plot_plotly(power_hist)


def get_rule_base_score(results_folder, param_file, train):
    with open(param_file, "r") as jsonfile:
        params = json.load(jsonfile)
    joint_params = {**params["common_params"], **params["algo_params"]}
    joint_params["algo_type"] = params["algo_type"]

    if "config_file" in joint_params:
        config_file = joint_params["config_file"]
    elif "gym_env" in joint_params.keys():
        config_file = joint_params["gym_env"]
    else:
        config_file = joint_params["case"]

    if train:
        if "start_time_train" in joint_params:
            start_time = joint_params["start_time_train"]
        else:
            start_time = 0

        tot_steps = joint_params["tot_steps"]
    else:
        if "start_time_valid" in joint_params:
            start_time = joint_params["start_time_valid"]
        else:
            start_time = 0
        tot_steps = joint_params["tot_steps"]

    rule_base_file = results_folder + "rule_base.csv"

    if os.path.isfile(rule_base_file):
        with open(rule_base_file, "r+") as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if (
                    row[0] == config_file
                    and row[1] == start_time
                    and row[2] == tot_steps
                ):
                    return float(row[3])

    return 0


if __name__ == "__main__":
    start_time = time.time()
    a = [
        "dieteren_results/dieteren_large_diff_sb3_0",
        "dieteren_results/dieteren_large_diff_sb3_1",
        "dieteren_results/dieteren_large_diff_tree_0",
        "dieteren_results/dieteren_large_diff_tree_1",
        "dieteren_results/dieteren_large_diff_tree_2",
        "dieteren_results/dieteren_large_diff_tree_3",
        "dieteren_results/dieteren_medium_diff_sb3_0",
    ]

    b = find_different_params("./", a)

    print("end_time:", time.time() - start_time)
