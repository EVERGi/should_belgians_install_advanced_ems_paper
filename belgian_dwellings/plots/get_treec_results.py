import os, sys

from belgian_dwellings.simulation.run_treec import find_best_tree_for_each_house
import json
import matplotlib.pyplot as plt


def count_features(tree_node, feature_counter):
    feature = tree_node["feature"]
    if "left_node" not in tree_node.keys() and "right_node" not in tree_node.keys():
        feature = "Action: " + feature
    if feature in feature_counter.keys():
        feature_counter[feature] += 1
    else:
        feature_counter[feature] = 1
    if "left_node" in tree_node.keys():
        count_features(tree_node["left_node"], feature_counter)
    if "right_node" in tree_node.keys():
        count_features(tree_node["right_node"], feature_counter)


def print_feature_counter(log_folder):

    best_tree = find_best_tree_for_each_house(log_folder)

    feature_counter = {
        "Heating setpoint": dict(),
        "Cooling setpoint": dict(),
        "WaterHeater": dict(),
        "Charger": dict(),
    }
    for house_str in best_tree.keys():
        model_path = best_tree[house_str]["model_path"]

        with open(model_path, "r") as f:
            model = json.load(f)

        tree_models = [value for key, value in model.items() if key.startswith("tree")]

        if len(tree_models) == 1:
            tree_eval = ["Charger"]
        elif len(tree_models) == 3:
            tree_eval = ["Heating setpoint", "Cooling setpoint", "WaterHeater"]
        elif len(tree_models) == 4:
            tree_eval = [
                "Heating setpoint",
                "Cooling setpoint",
                "WaterHeater",
                "Charger",
            ]
        for i, tree_model in enumerate(tree_models):
            key_counter = tree_eval[i]
            count_features(tree_model, feature_counter[key_counter])
    # Order the features by the number of times they appear
    for key in feature_counter.keys():
        feature_counter[key] = dict(
            sorted(feature_counter[key].items(), key=lambda item: item[1], reverse=True)
        )
    # Seperate the features in action and non action features
    for asset in feature_counter.keys():
        action_features = dict()
        non_action_features = dict()
        for feature in feature_counter[asset].keys():
            if feature.startswith("Action: "):
                action_features[feature] = feature_counter[asset][feature]
            else:
                non_action_features[feature] = feature_counter[asset][feature]
        feature_counter[asset] = dict()
        feature_counter[asset]["action"] = action_features
        feature_counter[asset]["non_action"] = non_action_features
    # Print the results
    for asset in feature_counter.keys():
        print(f"Asset: {asset}")
        print("Non action features:")
        for feature in feature_counter[asset]["non_action"].keys():
            print(f"{feature}: {feature_counter[asset]['non_action'][feature]}")
        print()
        print("Action features:")
        for feature in feature_counter[asset]["action"].keys():
            print(f"{feature}: {feature_counter[asset]['action'][feature]}")
        print()
        print()


def plot_training_curve(log_folder):
    best_tree = find_best_tree_for_each_house(log_folder)

    max_episode = 60200
    training_score = dict()
    for house_str in best_tree.keys():
        training_score[house_str] = list()
        model_path = best_tree[house_str]["model_path"]
        # Base folder is three folders up
        base_folder = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(model_path)))
        )
        model_folder = base_folder + "/models/"
        prev_episode = 0
        prev_score = 0
        sorted_model_files = sorted(
            os.listdir(model_folder), key=lambda x: int(x.split("_")[1])
        )
        for model_file in sorted_model_files:
            split_file = model_file.split("_")
            episode = int(split_file[1])
            score = float(split_file[2].replace(".json", ""))
            score_repeat = episode - prev_episode
            training_score[house_str] += [prev_score for _ in range(score_repeat)]

            prev_score = score
            prev_episode = episode
        score_repeat = max_episode - prev_episode
        training_score[house_str] += [prev_score for _ in range(score_repeat)]

    # Average the training scores
    avg_training_score = list()
    halfway_score = [scores[max_episode // 2] for _, scores in training_score.items()]
    end_score = [scores[max_episode - 1] for _, scores in training_score.items()]
    diff_half_end = [
        end_score[i] - half_score for i, half_score in enumerate(halfway_score)
    ]
    for i in range(max_episode):

        avg_score = 0
        for house_str in training_score.keys():
            avg_score += training_score[house_str][i]
        avg_training_score.append(avg_score / len(training_score.keys()))
    # Plot the training score
    plt.plot(avg_training_score)
    plt.title("Training score")
    plt.xlabel("Episode")
    plt.ylabel("Score")

    plt.figure(figsize=(10, 5))
    # Plot histogram of the difference between the end score and the halfway score
    plt.hist(diff_half_end, bins=20)
    plt.title("Difference between end score and halfway score")
    plt.xlabel("Difference")
    plt.ylabel("Number of houses")
    plt.show()


def compare_best_trees(log_folder_1, log_folder_2):
    best_tree_1 = find_best_tree_for_each_house(log_folder_1)
    best_tree_2 = find_best_tree_for_each_house(log_folder_2)

    for house_str in best_tree_2.keys():
        score_1 = best_tree_1[house_str]["score"]
        score_2 = best_tree_2[house_str]["score"]

        if score_2 > score_1:
            print(f"House {house_str}: {score_1} -> {score_2}")


if __name__ == "__main__":
    log_folder = "treec_train_500"
    log_folder_save = "treec_train_100_save"
    plot_training_curve(log_folder)

    # compare_best_trees(log_folder, log_folder_save)
