import shutil
import timeit
from treec.visualise_tree import binarytree_to_dot
from treec.tree import BinaryTree
import os
import json
from filelock import FileLock
import copy


class GeneralLogger:
    def __init__(self, save_dir, algo_type, common_params, algo_params):
        self.save_dir = save_dir
        self.algo_type = algo_type
        self.common_params = common_params
        self.algo_params = algo_params

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        self.folder_name = self.create_dir()

        self.initialise_dir()

        self.model_dir = self.folder_name + "models/"
        os.mkdir(self.model_dir)

        self.power_prof_dir = self.folder_name + "power_profiles/"
        os.mkdir(self.power_prof_dir)

        self.rewards_dir = self.folder_name + "rewards/"
        os.mkdir(self.rewards_dir)

        self.attributes_dir = self.folder_name + "attributes/"
        os.mkdir(self.attributes_dir)

        self.eval_score_path = self.folder_name + "eval_score.csv"

        file = open(self.eval_score_path, "w+")
        file.write("timestep,elapsed_time,eval_score\n")
        file.close()

        self.episode_score_file = self.folder_name + "episode_score_file.csv"
        self.create_episode_score_file()

        self.start_time = timeit.default_timer()

    def create_dir(self):
        config_file = self.common_params["config_file"]

        microgrid_name = config_file.split("/")[-1].split(".")[0]
        folder_start = microgrid_name + "_" + self.algo_type + "_"

        subfolders = [name for name in os.listdir(self.save_dir)]
        highest_num = -1
        for folder in subfolders:
            if folder.startswith(folder_start):
                folder_num = int(folder.replace(folder_start, ""))
                if folder_num > highest_num:
                    highest_num = folder_num

        folder_name = self.save_dir + "/" + folder_start + str(highest_num + 1) + "/"

        os.mkdir(folder_name)
        return folder_name

    def initialise_dir(self):
        config_file = self.common_params["config_file"]
        logged_config_file = self.folder_name + "microgrid_config.json"

        shutil.copyfile(config_file, logged_config_file)

        json_content = dict()
        json_content["algo_type"] = copy.copy(self.algo_type)
        json_content["common_params"] = copy.copy(self.common_params)
        json_content["algo_params"] = copy.copy(self.algo_params)

        for param_name, param_dict in json_content.items():
            if param_name == "algo_type":
                continue
            for key, value in param_dict.items():
                if callable(value):
                    param_dict[key] = value.__name__

        param_filepath = self.folder_name + "params_run.json"

        file = open(param_filepath, "w+")
        file.write(json.dumps(json_content, indent=4))
        file.close()

    def episode_eval_log(self, model, eval_score):
        with FileLock(self.episode_score_file + str(".lock"), mode=0o664):
            episode_count, _ = self.read_best_score_episode_count()
            self.update_episode_count()
            better_score = self.update_best_score(eval_score)

        score_str = "{0:.1f}".format(eval_score)

        eval_exten = str(episode_count) + "_" + score_str

        time_step = str(episode_count * self.common_params["tot_steps"])

        elapsed_time = str(timeit.default_timer() - self.start_time)

        text_file = open(self.eval_score_path, "a+")
        text_file.write(time_step + "," + elapsed_time + "," + score_str + "\n")
        text_file.close()

        if better_score:
            return eval_exten

        return None

    def save_model(self, model, eval_exten):
        pass

    def create_episode_score_file(self):
        file = open(self.episode_score_file, "w+")
        file.write("episode,0\nscore,\n")
        file.close()

    def update_episode_count(self):
        file = open(self.episode_score_file, "r+")
        file_content = file.read().split("\n")
        file.close()
        try:
            episode_count = int(file_content[0].replace("episode,", ""))
        except (ValueError, IndexError):
            print(file_content)
            return

        new_episode_count = episode_count + 1

        file_content[0] = f"episode,{new_episode_count}"

        file = open(self.episode_score_file, "w+")
        file.write("\n".join(file_content))
        file.close()

    def update_best_score(self, new_best_score):
        file = open(self.episode_score_file, "r+")
        file_content = file.read().split("\n")
        file.close()
        try:
            best_score = file_content[1].replace("score,", "")
        except IndexError:
            print(file_content)
            return False

        if best_score == "" or float(best_score) < new_best_score:
            file_content[1] = f"score,{new_best_score}"

            file = open(self.episode_score_file, "w+")
            file.write("\n".join(file_content))
            file.close()

            return True

        return False

    def read_best_score_episode_count(self):
        file = open(self.episode_score_file, "r+")
        file_content = file.read().split("\n")
        file.close()
        try:
            episode_count = int(file_content[0].replace("episode,", ""))
        except (ValueError, IndexError):
            episode_count = 0
            print(file_content)

        try:
            if file_content[1].replace("score,", "") == "":
                best_score = None
            else:
                best_score = float(file_content[1].replace("score,", ""))
        except (ValueError, IndexError):
            best_score = 100
            print(file_content)

        return episode_count, best_score


class TreeLogger(GeneralLogger):
    def __init__(self, save_dir, algo_type, common_params, algo_params):
        super().__init__(save_dir, algo_type, common_params, algo_params)

        self.dot_files_dir = self.folder_name + "dot_trees/"
        os.mkdir(self.dot_files_dir)

    def save_model(self, model, eval_exten):
        json_model = dict()
        for key, value in model.items():
            if key != "trees":
                json_model[key] = value

        for i, tree in enumerate(model["trees"]):
            tree_model = tree.get_value_dict()
            json_model[f"tree_{i}"] = tree_model

        model_str = json.dumps(json_model)

        model_path = self.model_dir + "model_" + eval_exten + ".json"

        file = open(model_path, "w+")

        file.write(model_str)

        file.close()

    def save_tree_dot(self, trees, all_nodes_visited, eval_exten):
        for i, tree in enumerate(trees):
            leafs_batt = [j[i] for j in all_nodes_visited]
            title = "Tree_" + str(i)
            dot_str = binarytree_to_dot(tree, title, leafs_batt)
            file = open(self.dot_files_dir + eval_exten + "_" + title + ".dot", "w+")
            file.write(dot_str)
            file.close()

    @staticmethod
    def get_best_model(model_folder):
        model_files = [
            f
            for f in os.listdir(model_folder)
            if os.path.isfile(os.path.join(model_folder, f))
        ]

        best_model = model_files[0]

        for model_file in model_files:
            best_score = float(best_model.split("_")[-1].replace(".json", ""))
            model_score = float(model_file.split("_")[-1].replace(".json", ""))

            best_episode = int(best_model.split("_")[-2].replace(".json", ""))
            model_episode = int(model_file.split("_")[-2].replace(".json", ""))

            if model_score >= best_score and model_episode > best_episode:
                best_model = model_file
        best_model_file = model_folder + best_model

        with open(best_model_file) as file:
            model = json.load(file)

        return model

    @staticmethod
    def get_best_trees(model_folder):
        tree_model = TreeLogger.get_best_model(model_folder)
        trees = list()

        trees_num = [
            int(key.replace("tree_", ""))
            for key in tree_model.keys()
            if key.startswith("tree_")
        ]

        for tree_num in sorted(trees_num):
            value_dict = tree_model[f"tree_{tree_num}"]
            tree = BinaryTree(value_dict)
            trees.append(tree)

        return trees
    
    @staticmethod
    def clean_logs(log_folder):
        models_folder = log_folder + "/models/"

        sorted_models = sorted(os.listdir(models_folder), key=lambda x: int(x.split("_")[1]))

        # Keep first and last and 5 models at regular intervals
        if len(sorted_models) <= 7:
            return  # Not enough models to clean
        to_keep = [sorted_models[0], sorted_models[-1]]  # Keep first and last
        interval = len(sorted_models)// 6
        for i in range(1, 6):
            index = i * interval
            
            to_keep.append(sorted_models[index])
        
        for model in sorted_models:
            if model not in to_keep:
                model_path = models_folder + model
                os.remove(model_path)
        
        dot_trees = log_folder + "/dot_trees/"
        for dot_tree in os.listdir(dot_trees):
            keep_file = False
            for model in to_keep:
                start_str = model.split("_")[1]+ "_"
                if dot_tree.startswith(start_str):
                    keep_file = True
                    break
            if not keep_file:
                dot_tree_path = dot_trees + dot_tree
                os.remove(dot_tree_path)