import os
import shutil
import json

def cleanup_treec_training(tot_houses):
    train_folder = f"treec_train_{tot_houses}/"
    for house_folder in sorted(os.listdir(train_folder)):

        if house_folder.startswith("house_"):
            house_path = train_folder + house_folder+"/"
            print(f"Cleaning up {house_path}")
            train_name = house_folder+"_tree_"
            all_tree_train = [f for f in os.listdir(house_path) if f.startswith(train_name)]
            
            # Sort by the last number
            all_tree_train.sort(key=lambda x: int(x.split("_")[-1]))
            to_remove = []
            for tree_train_folder in all_tree_train:
                valid_folder = house_path+tree_train_folder+"/validation/"
                if not os.path.exists(valid_folder):
                    to_remove.append(tree_train_folder)
            for to_remove_folder in to_remove:
                shutil.rmtree(house_path+to_remove_folder)
                all_tree_train.remove(to_remove_folder)
                print(f"Removed {house_path+to_remove_folder}")
            # Rename the remaining folders
            for i, to_rename_folder in enumerate(all_tree_train):
                new_name = f"{train_name}{i}"
                old_path = house_path+to_rename_folder
                new_path = house_path+new_name
                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"Renamed {old_path} to {new_path}")
            print()

def cleanup_treec_charger(tot_houses):
    train_folder = f"treec_train_{tot_houses}/"
    for house_folder in sorted(os.listdir(train_folder)):

        if house_folder.startswith("house_"):
            house_path = train_folder + house_folder+"/"
            print(f"Cleaning up {house_path}")
            train_name = house_folder+"_tree_"
            all_tree_train = [f for f in os.listdir(house_path) if f.startswith(train_name)]
            delete_folder = False
            for tree_train_folder in all_tree_train:
                config_file = house_path+tree_train_folder+"/microgrid_config.json"
                config = json.load(open(config_file, "r"))
                for _, asset in config["Assets"].items():
                    if asset["name"] == "Charger_0":
                        delete_folder = True
                        break
                if delete_folder:
                    shutil.rmtree(house_path+tree_train_folder)
                    print(f"Removed {house_path+tree_train_folder}")

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
            # print(f"Removed {model_path}") 
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
            # print(f"Removed {dot_tree_path}")

def cleanup_all_logs(tot_houses):
    train_folder = f"treec_train_{tot_houses}/"
    for house_folder in sorted(os.listdir(train_folder)):
        if house_folder.startswith("house_"):
            house_path = train_folder + house_folder+"/"
            for log_folder in os.listdir(house_path):
                print(f"Cleaning logs in {house_path + log_folder}")
                log_path = house_path +"/"+ log_folder + "/"
                clean_logs(log_path)


if __name__ == "__main__":
    tot_houses = 500
    # cleanup_all_logs(tot_houses)
    # log_folder = f"treec_train_{tot_houses}/house_2/house_2_tree_0/"
    # clean_logs(log_folder)
    # cleanup_treec_charger(tot_houses)
    cleanup_treec_training(tot_houses)